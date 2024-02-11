import warnings
import numpy as np
from pathlib import Path
import subprocess
import os
import pandas as pd
import sys

from numpy import ComplexWarning

from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from collections import namedtuple
import hashlib
from contextlib import redirect_stderr
from io import StringIO

from compyct.templates import SimTemplate
from .backend import Netlister, MultiSimSesh, get_va_path, get_va_paths, SimulatorCommandException
from compyct.paramsets import ParamSet, ParamPlace, spicenum_to_float, float_to_spicenum
from ..util import catch_stderr, logger

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
        self.modelcard_name=f"{self.simtemp._patch.model}_standin"

    def nstr_iabstol(self,abstol):
        return f".option abstol={float_to_spicenum(abstol)}"

    def nstr_temp(self, temp=27, tnom=27):
        return f".option temp={float_to_spicenum(temp)} tnom={float_to_spicenum(tnom)}"

    def nstr_modeled_xtor(self,name,netd,netg,nets,netb,dt,inst_param_ovrd={},internals_to_save=[]):
        assert len(inst_param_ovrd)==0
        assert dt is None
        patch=self.simtemp._patch
        assert patch.terminals[:4]==['d','g','s','b']
        inst_paramstr=' '.join(f'{k}={v}'\
                for k,v in patch.filled().to_base().break_into_model_and_instance()[1].items())
        has_dt_terminal='dt' in patch.terminals
        intstr=f"\n.save all "+\
                " ".join([f"@n{name.lower()}[{k}]" for k in internals_to_save])
        return f"N{name.lower()} {netd} {netg} {nets} {netb} {'dt' if has_dt_terminal else ''}"\
                    f" {self.modelcard_name} {inst_paramstr}"+ intstr

    @staticmethod
    def nstr_VDC(name,netp,netm,dc):
        return f"V{name} {netp} {netm} dc {dc}"

    @staticmethod
    def nstr_R(name,netp,netm,r):
        return f"R{name} {netp} {netm} {r}"

    @staticmethod
    def nstr_IDC(name,netp,netm,dc):
        return f"I{name} {netp} {netm} dc {dc}"

    @staticmethod
    def nstr_VAC(name,netp,netm,dc,ac=1):
        return f"V{name} {netp} {netm} dc {dc} ac {ac}"

    @staticmethod
    def nstr_port(name,netp,netm,dc,portnum,z0=50):
        #return f"V{name} {netp} {netm} dc {dc} portnum {portnum} z0 {z0}"
        return f"V{name} port_{name}_ac {netm} portnum {portnum} z0 {z0} dc 0\n"\
               f"Cport{name} {netp} port_{name}_ac 1\n" \
               f"Lport{name} {netp} port_{name}_dc 1meg\n"\
               f"Vdc_{name} port_{name}_dc {netm} dc {dc}\n"
    def astr_altervportdc(self,whichv, tovalue, portnum, name=None):
        return lambda ngss: \
            (None,ngss.alter_device(f'vdc_{whichv.lower()}',dc=tovalue))

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
            ngss.exec_command(f"dc v{whichv.lower()} {start:.5g} {stop:.5g} {step:.5g}")
            with catch_stderr(["Unit is None"]):
                ret=name, self.analysis_to_df(ngss.plot(None,ngss.last_plot).to_analysis())
            ngss.destroy()
            return ret
        return sweepvdc

    def astr_sweepidc(self,whichi, start, stop, step, name=None):
        def sweepidc(ngss):
            ngss.exec_command(f"dc i{whichi.lower()} {start:.5g} {stop:.5g} {step:.5g}")
            with catch_stderr(["Unit is None"]):
                ret=name, self.analysis_to_df(ngss.plot(None,ngss.last_plot).to_analysis())
            ngss.destroy()
            return ret
        return sweepidc
        
    def astr_sweepvac(self, whichv, start, stop, step, freq, name=None):
        def sweepvac(ngss):
            dfs=[]
            for v in np.arange(start,stop+1e-9,step):
                try:
                    ngss.alter_device(f'v{whichv.lower()}',dc=v)
                    ngss.exec_command(f"ac lin 1 {freq} {freq}")
                except Exception as e:
                    raise SimulatorCommandException(original_error=e)
                with warnings.catch_warnings():
                    # Todo: debug and remove this.. only became necessary when I was
                    # including device internal parameters (eg "gmi") in output via ngspice ".save all ..." command
                    warnings.filterwarnings('ignore',category=ComplexWarning)
                    with catch_stderr(["Unit is None"]):
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

    def astr_spar(self, fstart, fstop, pts_per_dec=None, points=None, sweep_option='dec', name=None):
        def spar(ngss):
            narg={'lin':points,'dec':pts_per_dec}[sweep_option]
            assert narg is not None
            ngss.exec_command(f"sp {sweep_option} {narg} {fstart} {fstop}")
            # PySpice doesn't have an s-parameter analysis class yet so we'll extract it differently
            #df=self.analysis_to_df(ngss.plot(None,ngss.last_plot).to_analysis())
            plot=ngss.plot(None,ngss.last_plot)
            df=pd.DataFrame({
                'freq': plot['frequency'].to_waveform(to_real=True),

                'S11':  plot['S_1_1'].to_waveform(),
                'S12':  plot['S_1_2'].to_waveform(),
                'S21':  plot['S_2_1'].to_waveform(),
                'S22':  plot['S_2_2'].to_waveform(),

                'Y11':  plot['Y_1_1'].to_waveform(),
                'Y12':  plot['Y_1_2'].to_waveform(),
                'Y21':  plot['Y_2_1'].to_waveform(),
                'Y22':  plot['Y_2_2'].to_waveform(),
            })
            ngss.destroy()
            # If we want, there are also Y parameters, Z parameters here ripe for picking
            return name, df
        return spar

    def get_netlist(self):
        patch=self.simtemp._patch
        netlist=f".title {self.circuit_name}\n"
        if len(patch.param_set.va_includes):
            netlist+=".control\n"
            for vaname in patch.param_set.va_includes:
                netlist+=f"pre_osdi {get_confirmed_osdi_path(vaname)}\n"
            netlist+=".endc\n"
        if patch is not None:
            paramstr=" ".join([f"{k}={v}"
                               for k,v in patch.filled().to_base().break_into_model_and_instance()[0].items()])
            netlist+=f".model {self.modelcard_name} {patch.model} {paramstr}\n"

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
        internals={k:n for k,n in analysis.internal_parameters.items()}
        df=pd.DataFrame(dict(**branches,**nodes,**time,**internals))
        return df


class NgspiceMultiSimSesh(MultiSimSesh):

    singleton_in_use=False

    def print_netlists(self):
        for simname, simtemp in self.simtemps.items():
            print(f"###################### {simname} #############")
            print(NgspiceNetlister(simtemp).get_netlist())

    def _ensure_clear_ngspice(self, if_circuits='warn'):
        # Make sure its clear of any previous circuits
        # by loading a blank circuit and then clearing
        # all circuits.  [Loading the blank prevents
        # NgSpiceShared from error-logging about setcirc
        # if there really are no prior circuits.]
        self._ngspice.load_circuit(".title Nothing\n.end")
        sc=self._ngspice.exec_command('setcirc')
        for i in range(len(sc.split("\n"))-1):
            self._ngspice.remove_circuit()
            if if_circuits=='warn' and i>0:
                print("ALERT, REMOVING OLD CIRCUIT")

    def __enter__(self):
        assert not NgspiceMultiSimSesh.singleton_in_use
        NgspiceMultiSimSesh.singleton_in_use=True
        super().__enter__()

        ngs_warnings=StringIO()
        with redirect_stderr(ngs_warnings):
            self._ngspice=NgSpiceShared.new_instance()
        ngs_warnings.seek(0)
        ngs_warnings=ngs_warnings.readlines()
        # TODO: make this use the catch_stderr function in compyct.util
        for w in ngs_warnings:
            cms=['spice2poly','analog','digital','xtradev','xtraevt','table']
            if any((f"{cm}.cm couldn't be loaded" in w) for cm in cms):
                continue
            if "Unsupported Ngspice version 41" in w:
                continue
            raise Exception("New error while trying to load Ngspice... "+w)

        self._ensure_clear_ngspice()

        for simname, simtemp in self.simtemps.items():
            nl=NgspiceNetlister(simtemp)
            self._sessions[simname]=nl
            self._ngspice.load_circuit(nl.get_netlist())

        self._circuit_numbers={simname:i for i,(simname,_) in\
                enumerate(reversed(list(self.simtemps.items())),start=1)}
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._ensure_clear_ngspice(if_circuits='ignore')
        assert NgspiceMultiSimSesh.singleton_in_use
        NgspiceMultiSimSesh.singleton_in_use=False
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            del self._ngspice
            self._sessions={}
            
    def __del__(self):
        super().__del__()
        if hasattr(self,'_ngspice'):
            del self._ngspice

    def run_with_params(self, params=None, full_resync=False, only_temps=None):
        results={}
        import time
        for (simname, simtemp) in self.simtemps.items():
            if only_temps is not None and simname not in only_temps: continue
            logger.debug(f"Running {simname}")
            unparsed_result={}
            nl=self._sessions[simname]
            self._ngspice.set_circuit(self._circuit_numbers[simname])
            if params is not None:

                with simtemp.tentative_base_deltas(params) as deltas:
                    if full_resync: deltas=simtemp._patch
                    dmodl,dinst=deltas.break_into_model_and_instance()
                    if len(dmodl):
                        logger.debug(f"Model param changes: {dmodl}")
                        self._ngspice.alter_model(nl.modelcard_name,**dmodl)
                        logger.debug("Done model param changes")
                    if len(dinst):
                        for dev in nl._modeled_xtors:
                            logger.debug(f"Instance param changes: {dinst}")
                            self._ngspice.alter_device(dev,**dinst)
                            logger.debug("Done instance param changes")

            for an in nl.analyses:
                logger.debug(f"Running an analysis")
                name,subresult=an(self._ngspice)
                if name is not None:
                    unparsed_result[name]=subresult
            logger.debug(f"Done analyses")
                #print(f"Ran analysis {name}",time.time())
            #print("About to parse: ",time.time())
            #results[simname]=simtemp.parse_return(nl.preparse_return(unparsed_result))
            results[simname]=simtemp.postparse_return(simtemp.parse_return(unparsed_result))

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
