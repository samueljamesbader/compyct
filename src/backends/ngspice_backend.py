import numpy as np
from pathlib import Path
import subprocess
import os
import pandas as pd
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from collections import namedtuple

from tempfile import NamedTemporaryFile
from compyct.templates import SimTemplate
from .backend import Netlister, MultiSimSesh
from compyct.paramsets import ParamPlace
from compyct import COMPYCT_VA_PATH, COMPYCT_OSDI_PATH

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
        
    
    def nstr_modeled_xtor(self,name,netd,netg,nets,netb,dt,inst_param_ovrd={}):
        assert len(inst_param_ovrd)==0
        assert dt is None
        ps=self.simtemp.model_paramset
        inst_paramstr=' '.join(f'{k}={v}'\
                for k,v in ps.get_values().items()
                               if ps.get_place(k)==ParamPlace.INSTANCE)
        #inst_paramstr=""
        return f"N{name.lower()} {netd} {netg} {nets} {netb} dt"\
                    f" {self.modelcard_name} {inst_paramstr}"
        
    @staticmethod
    def nstr_VDC(name,netp,netm,dc):
        return f"V{name} {netp} {netm} dc {dc}"
        
    @staticmethod
    def nstr_VAC(name,netp,netm,dc,ac=1):
        return f"V{name} {netp} {netm} dc {dc} ac {ac}"

    def astr_altervdc(self,whichv, tovalue, name=None):
        return lambda ngss:\
                    (None,ngss.alter_device(f'v{whichv.lower()}',dc=tovalue))
        
    def astr_sweepvdc(self,whichv, start, stop, step, name=None):
        def sweepvdc(ngss):
            ngss.exec_command(f"dc v{whichv.lower()} {start} {stop} {step}")
            return name, ngss.plot(None,ngss.last_plot).to_analysis()
        return sweepvdc
        
    def astr_sweepvac(self, whichv, start, stop, step, freq, name=None):
        def sweepvac(ngss):
            branches,nodes={},{}
            for v in np.arange(start,stop+1e-9,step):
                ngss.alter_device(f'v{whichv.lower()}',dc=v)
                ngss.exec_command(f"ac lin 1 {freq} {freq}")
                an=ngss.plot(None,ngss.last_plot).to_analysis()
                for k,b in an.branches.items():
                    branches[k]=branches.get(k,[])+[b.as_ndarray()[0]]
                for k,n in an.nodes.items():
                    nodes[k]=nodes.get(k,[])+[n.as_ndarray()[0]]
                nodes['v-sweep']=nodes.get('v-sweep',[])+[v]
            return name,PseudoAnalysis(
                branches={k:np.array(b) for k,b in branches.items()},
                nodes={k:np.array(n) for k,n in nodes.items()})
        return sweepvac

    def get_netlist(self):
        ps=self.simtemp.model_paramset
        netlist=f".title {self.circuit_name}\n"
        if len(ps.osdi_includes):
            netlist+=".control\n"
            for i in ps.osdi_includes:
                netlist+=f"pre_osdi {i}\n"
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


    def preparse_return(self,result):
        branches={sweepname:{(f"{k}#p" if "#" not in k else k):b
                                 for k,b in an.branches.items()}
                      for sweepname,an in result.items()}
        nodes={sweepname:{k:n for k,n in an.nodes.items()}
                      for sweepname,an in result.items()}
        return {sweepname:pd.DataFrame(dict(**bdict,**ndict))
                    for (sweepname,bdict),ndict
                        in zip(branches.items(),nodes.values())}
        
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
        for (simname, simtemp) in self.simtemps.items():
            #print(f"Running {simname}")
            unparsed_result={}
            nl=self._sessions[simname]
            self._ngspice.set_circuit(self._circuit_numbers[simname])
            ps=simtemp.model_paramset
            nv=ps.update_and_return_changes(params)
            #print("Changes: ",nv)
            self._ngspice.alter_model(nl.modelcard_name,
                **{k:v for k,v in nv.items() if ps.get_place(k)==ParamPlace.MODEL})
            for dev in nl._modeled_xtors:
                self._ngspice.alter_device(dev,
                    **{k:v for k,v in nv.items() if ps.get_place(k)==ParamPlace.INSTANCE})
            for an in nl.analyses:
                name,subresult=an(self._ngspice)
                if name is not None:
                    unparsed_result[name]=subresult
            results[simname]=simtemp.parse_return(nl.preparse_return(unparsed_result))
        return results            
        

def compile_va_to_osdi(vaname=None):
    for osdipath in COMPYCT_OSDI_PATH.glob("*"):
        if vaname is None or osdipath.name.replace(".osdi",".va")==vaname:
            osdipath.unlink()
    for vapath in va_dir.glob("*.va"):
        if vaname is None or vapath.name==vaname:
            osdipath=osdi_dir/vaname.replace(".va",".osdi")
            subprocess.run(["openvaf",str(vapath),"-o",str(osdipath)])
            