import numpy as np
from pathlib import Path
import subprocess
import os
import pandas as pd
import sys
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from collections import namedtuple
import hashlib

from tempfile import NamedTemporaryFile
from compyct.templates import SimTemplate
from .backend import Netlister, MultiSimSesh, get_va_path, get_va_paths
from compyct.paramsets import ParamSet, ParamPlace, spicenum_to_float, float_to_spicenum

PseudoAnalysis=namedtuple("PseudoAnalysis",["branches","nodes"])

class NgspiceNetlister(Netlister):
    GND='0'
    _netlist_num=0
    
    def __init__(self,template: SimTemplate):
        self.simtemp: SimTemplate=template
        self._tf = None
        self._analysiscount=0
        
        num=self.__class__._netlist_num
        self.__class__._netlist_num+=1
        self.circuit_name=f"Netlist{num}_{self.simtemp.__class__.__name__}"
        self.analyses=self.simtemp.get_analysis_listing(self)
        self.modelcard_name=f"{self.simtemp.model_paramset.model}_standin"

    def nstr_iabstol(self,abstol):
        return f".option abstol={float_to_spicenum(abstol)}"
    
    def nstr_modeled_xtor(self,name,netd,netg,nets,netb,dt,inst_param_ovrd={}):
        assert len(inst_param_ovrd)==0
        assert dt is None
        ps=self.simtemp.model_paramset
        assert ps.terminals==['d','g','s','b']
        inst_paramstr=' '.join(f'{k}={v}'\
                for k,v in ps.get_values().items()
                               if ps.get_place(k)==ParamPlace.INSTANCE)
        assert ps.terminals[:4]==['d','g','s','b']
        has_dt_terminal='dt' in ps.terminals
        return f"N{name.lower()} {netd} {netg} {nets} {netb} {'dt' if has_dt_terminal else ''}"\
                    f" {self.modelcard_name} {inst_paramstr}"
        
    @staticmethod
    def nstr_VDC(name,netp,netm,dc):
        return f"V{name} {netp} {netm} dc {dc}"
        
    @staticmethod
    def nstr_VAC(name,netp,netm,dc,ac=1):
        return f"V{name} {netp} {netm} dc {dc} ac {ac}"

    @staticmethod
    def nstr_VPulses(name,netp,netm,dc, pulse_width, pulse_period, rise_time, fall_time, vpulses):
        pulse_width,pulse_period,rise_time,fall_time=[spicenum_to_float(x) for x in
                                                      [pulse_width,pulse_period,rise_time,fall_time]]
        times=[t for t0 in np.arange(len(vpulses))*pulse_period
                   for t in [t0,
                             t0+rise_time,
                             t0+rise_time/2+pulse_width-fall_time/2,
                             t0+rise_time/2+pulse_width+fall_time/2]]
        voltages=[v for vpulse in vpulses
                    for v in [dc,vpulse,vpulse,dc]]
        pwlstr=" ".join([f"{float_to_spicenum(t)} {float_to_spicenum(v)}" for t,v in zip(times,voltages)])
        return f"V{name} {netp} {netm} PWL({pwlstr})"
    @staticmethod
    def nstr_VStep(name, netp, netm, dc, rise_time, final_v):
        dc,rise_time,final_v=[spicenum_to_float(x) for x in [dc,rise_time,final_v]]
        return f"V{name} {netp} {netm} PWL(0 {dc} {float_to_spicenum(rise_time)} {final_v})"

    def astr_altervdc(self,whichv, tovalue, name=None):
        return lambda ngss:\
                    (None,ngss.alter_device(f'v{whichv.lower()}',dc=tovalue))
        
    def astr_sweepvdc(self,whichv, start, stop, step, name=None):
        def sweepvdc(ngss):
            ngss.exec_command(f"dc v{whichv.lower()} {start} {stop} {step}")
            ret=name, self.analysis_to_df(ngss.plot(None,ngss.last_plot).to_analysis())
            ngss.destroy()
            return ret
        return sweepvdc
        
    def astr_sweepvac(self, whichv, start, stop, step, freq, name=None):
        def sweepvac(ngss):
            dfs=[]
            for v in np.arange(start,stop+1e-9,step):
                ngss.alter_device(f'v{whichv.lower()}',dc=v)
                ngss.exec_command(f"ac lin 1 {freq} {freq}")
                an=ngss.plot(None,ngss.last_plot).to_analysis()
                dfs.append(self.analysis_to_df(an).assign(**{'v-sweep':v}))
                ngss.destroy()
            return name, pd.concat(dfs)
        return sweepvac

    def astr_idealpulsed(self, vdcs:list, vpulses:dict, rise_time, meas_delay, check_flat=None, name=None):
        def idealpulsed(ngss):
            dfs=[]
            for i in range(len(list(vpulses.values())[0])):
                for whichv,vs in vpulses.items():
                    ngss.exec_command(f"alter v{whichv.lower()} PWL = [ {0} {vdcs[whichv]} {rise_time} {vs[i]} ]")

                ngss.exec_command(f"tran {spicenum_to_float(rise_time)/5}"\
                                  f" {spicenum_to_float(meas_delay)+spicenum_to_float(rise_time)}")
                df=self.analysis_to_df(ngss.plot(None,ngss.last_plot).to_analysis())
                if check_flat is not None:
                    mask=df['time']>spicenum_to_float(meas_delay)/2
                    assert np.sum(mask)>5
                    assert np.allclose(df[check_flat][mask],df[check_flat][-1])
                dfs.append(df.iloc[-1:])
                ngss.destroy()
            return name,pd.concat(dfs).reset_index()
        return idealpulsed

    def get_netlist(self):
        ps:ParamSet=self.simtemp.model_paramset
        netlist=f".title {self.circuit_name}\n"
        if len(ps.va_includes):
            netlist+=".control\n"
            for vaname in ps.va_includes:
                netlist+=f"pre_osdi {get_confirmed_osdi_path(vaname)}\n"
            netlist+=".endc\n"
        if ps is not None:
            paramstr=" ".join([f"{k}={v}"
                               for k,v in ps.get_values().items()
                                   if ps.get_place(k)==ParamPlace.MODEL])
            netlist+=f".model {self.modelcard_name} {ps.model} {paramstr}\n"

        sl=self.simtemp.get_schematic_listing(self)
        # Don't really like collecting this here without having a 
        # safeguard to make sure there's only one netlist
        self._modeled_xtors=[l.split()[0].lower() for l in sl if l.startswith("N")]
        netlist+="\n".join(sl)+"\n"
        netlist+=".end\n"
        return netlist

    def analysis_to_df(self,analysis):
        branches={(f"{k}#p" if "#" not in k else k):b
                             for k,b in analysis.branches.items()}
        nodes={k:n for k,n in analysis.nodes.items()}
        time=({'time':analysis.time} if hasattr(analysis,'time') else {})
        return pd.DataFrame(dict(**branches,**nodes,**time))


class NgspiceMultiSimSesh(MultiSimSesh):
        
    def print_netlists(self):
        for simname, simtemp in self.simtemps.items():
            print(f"###################### {simname} #############")
            print(NgspiceNetlister(simtemp).get_netlist())

    def __enter__(self):
        super().__enter__()

        # Get a NgSpiceShared to commmunicate with
        # Despite the name, the returned object may be reused
        # since we're not supplyed an ngspice_id
        self._ngspice=NgSpiceShared.new_instance()

        # Make sure its clear of any previous circuits
        # by loading a blank circuit and then clearing
        # all circuits.  [Loading the blank prevents
        # NgSpiceShared from error-logging about setcirc
        # if there really are no prior circuits.]
        self._ngspice.load_circuit(".title Nothing\n.end")
        sc=self._ngspice.exec_command('setcirc')
        for i in range(len(sc.split("\n"))-1):
            self._ngspice.remove_circuit()
        
        for simname, simtemp in self.simtemps.items():
            nl=NgspiceNetlister(simtemp)
            self._sessions[simname]=nl
            self._ngspice.load_circuit(nl.get_netlist())

        self._circuit_numbers={simname:i for i,(simname,_) in\
                enumerate(reversed(self.simtemps.items()),start=1)}
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            del self._ngspice
            self._sessions={}
            
    def __del__(self):
        super().__del__()
        if hasattr(self,'_ngspice'):
            del self._ngspice
            
    def run_with_params(self,params={}):
        results={}
        #import time
        for (simname, simtemp) in self.simtemps.items():
            #print(f"Running {simname}", time.time())
            unparsed_result={}
            nl=self._sessions[simname]
            self._ngspice.set_circuit(self._circuit_numbers[simname])
            ps=simtemp.model_paramset
            nv=ps.update_and_return_changes(params)
            #print("Changes: ",nv, time.time())
            self._ngspice.alter_model(nl.modelcard_name,
                **{k:v for k,v in nv.items() if ps.get_place(k)==ParamPlace.MODEL})
            #print("Made model changes: ",time.time())
            for dev in nl._modeled_xtors:
                self._ngspice.alter_device(dev,
                    **{k:v for k,v in nv.items() if ps.get_place(k)==ParamPlace.INSTANCE})
            #print("Made inst changes: ",time.time())
            for an in nl.analyses:
                name,subresult=an(self._ngspice)
                if name is not None:
                    unparsed_result[name]=subresult    
                #print(f"Ran analysis {name}",time.time())
            #print("About to parse: ",time.time())
            #results[simname]=simtemp.parse_return(nl.preparse_return(unparsed_result))
            results[simname]=simtemp.parse_return(unparsed_result)

            #print("Done: ",time.time())
        return results            
        

def get_unconfirmed_osdi_path(vaname) -> Path:
    COMPYCT_OSDI_PATH=Path(os.environ['COMPYCT_OSDI_PATH'])
    vapath=get_va_path(vaname=vaname)

    # Read the file in 1kb chunks, updating the hash for each
    hasher=hashlib.sha1()
    with open(vapath,'rb') as input:
        chunk = 0
        while chunk != b'':
            chunk = input.read(1024)
            hasher.update(chunk)

    # Return an osdi filename incorporating the hex digest of the va file
    hash=hasher.hexdigest()
    osdipath=COMPYCT_OSDI_PATH/str(vaname).replace(".va","_"+hash+".osdi")
    return osdipath

def compile_va_to_osdi(vaname=None):
    COMPYCT_OSDI_PATH=Path(os.environ['COMPYCT_OSDI_PATH'])
    COMPYCT_OPENVAF_PATH=Path(os.environ['COMPYCT_OPENVAF_PATH'])

    # Delete any old compilation files that match this vaname
    for osdipath in COMPYCT_OSDI_PATH.glob("*"):
        if vaname is None or str(vaname).replace(".va","") in osdipath.name:
            if not osdipath.name.endswith(".va"):
                osdipath.unlink()

    # Compile anew
    for vapath in get_va_paths():
        if vaname is None or vapath.name==vaname:
            osdipath=get_unconfirmed_osdi_path(vaname)
            print(f"Compiling {vapath} to {osdipath}")
            result=subprocess.run([str(COMPYCT_OPENVAF_PATH/"openvaf"),str(vapath),"-o",str(osdipath)],
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if result.returncode!=0:
                raise Exception("Compilation seems to have failed: "+result.stdout.decode(sys.stdout.encoding))

def get_confirmed_osdi_path(vaname) -> Path:
    path=get_unconfirmed_osdi_path(vaname=vaname)
    if not path.exists():
        compile_va_to_osdi(vaname=vaname)
    return path
