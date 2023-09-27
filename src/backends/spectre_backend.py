import pyspectre as psp
from tempfile import NamedTemporaryFile
from compyct.templates import SimTemplate
from .backend import Netlister, MultiSimSesh
from compyct.paramsets import ParamPlace

class SpectreNetlister(Netlister):
    GND='0'
    
    def __init__(self,template: SimTemplate):
        self.simtemp: SimTemplate=template
        self._tf = None
        self._analysiscount=0

    #@staticmethod
    #def nstr_param(params):
    #    return f"parameters "+\
    #        " ".join([f"{k}={v}" for k,v in params.items()])
        
    def nstr_modeled_xtor(self,name,netd,netg,nets,netb,dt,inst_param_ovrd={}):
        assert len(inst_param_ovrd)==0
        assert dt is None
        ps=self.simtemp.model_paramset
        inst_paramstr=' '.join(f'{k}=instparam_{k}'\
                for k in ps if ps.get_place(k)==ParamPlace.INSTANCE)
        return f"X{name} ({netd} {netg} {nets} {netb})"\
                    f" {ps.model}_standin {inst_paramstr}"
        
    @staticmethod
    def nstr_VDC(name,netp,netm,dc):
        return f"V{name} ({netp} {netm}) vsource dc={dc} type=dc"
        
    @staticmethod
    def nstr_VAC(name,netp,netm,dc,ac=1):
        return f"V{name} ({netp} {netm}) vsource dc={dc} mag={ac} type=dc"

    def astr_altervdc(self,whichv, tovalue, name=None):
        if name is None:
            name=f"alter{self._analysiscount}"
            self._analysiscount+=1
        return f"{name} alter dev=V{whichv} param=dc value={tovalue}"
        
    def astr_sweepvdc(self,whichv, start, stop, step, name=None):
        if name is None:
            name=f"sweep{self._analysiscount}"
            self._analysiscount+=1
        return f"{name} dc dev=V{whichv} param=dc start={start} stop={stop} step={step}"
        
    def astr_sweepvac(self,whichv, start, stop, step, freq, name=None):
        if name is None:
            name=f"sweep{self._analysiscount}"
            self._analysiscount+=1
        return f"{name} ac dev=V{whichv} param=dc start={start} stop={stop} step={step} freq={freq}"
        
    # def nstr_alter(dev,param,value,name=None):
    #     if name is None:
    #         name=f"alter{self._altercount}"
    #         self._altercount+=1
    #     return f"{name} alter dev={dev} param={param} value={value}"

    def get_netlist_file(self):
        """ Creates a temp netlist file if needed and returns its path.

        Warning: as soon as this SpiceNetlister object is garbage-collected,
        the system may delete the netlist file.  So hold on to your Netlister!
        """
        if self._tf is None:
            self._tf=NamedTemporaryFile(prefix=self.simtemp.__class__.__name__,mode='w')
            self._tf.write(f"// {self.simtemp.__class__.__name__}\n")
            self._tf.write(f"simulator lang = spectre\n")
            ps=self.simtemp.model_paramset
            if ps is not None:
                for i in ps.scs_includes:
                    if type(i)==str:
                        self._tf.write(f"include \"{i}\"\n")
                    else:
                        self._tf.write(
                            f"include \"{i[0]}\" {' '.join(i[1:])}\n")
                for i in ps.va_includes:
                    assert type(i)==str
                    self._tf.write(f"ahdl_include \"{i}\"\n")
                        
                self._tf.write(f"model {ps.model}_standin {ps.model}\n")
                paramlinedict={("modparam_"+k):ps.get_value(k) for k in ps}
                self._tf.write(
                    f"parameters "+\
                    ' '.join([f'{k}={v}' for k,v in paramlinedict.items()])\
                    +"\n\n")
                instance_params={k:ps.get_value(k)
                    for k in ps if ps.get_place(k)==ParamPlace.INSTANCE}
                if len(instance_params):
                    self._tf.write(f"parameters "+\
                                       ' '.join(f'instparam_{k}={v}'
                                            for k,v in instance_params.items())+\
                                   "\n")
            self._tf.write("\n".join(self.simtemp.get_schematic_listing(self))+"\n")
            if ps is not None:                
                self._tf.write(
                    f"set_modparams altergroup {{\nmodel {ps.model}_standin"\
                    f" {ps.model} "+\
                    " ".join((f"{k}=modparam_{k}" for k in ps))\
                    +"\n}\n\n")
            self._tf.write("\n".join(self.simtemp.get_analysis_listing(self))+"\n")
            self._tf.flush()
        return self._tf.name

    def get_spectre_names_for_param(self,param):
        prefix={ParamPlace.MODEL.value:'mod',ParamPlace.INSTANCE.value:'inst'}\
                    [self.simtemp.model_paramset.get_place(param).value]+'param_'
        return prefix+param
        
    def preparse_return(self,result):
        def standardize_col(k):
            if k=='dc':
                return 'v-sweep'
            else:
                return (k.lower().replace(":","#"))
        def standardize_swname(k):
            #print("SWEEP NAME: ",k)
            return k.split("`")[1].split("'")[0]
        return {standardize_swname(sweepname):sweepdata.rename(columns=standardize_col)
                    for sweepname,sweepdata in result.items()} 
    

class SpectreMultiSimSesh(MultiSimSesh):

    def print_netlists(self):
        for simname,simtemp in self.simtemps.items():
            print(f"###################### {simname} #############")
            nl=SpectreNetlister(simtemp)
            with open(nl.get_netlist_file(),'r') as f:
                print(f.read())
            
    
    def __enter__(self):
        super().__enter__()
        for simname,simtemp in self.simtemps.items():
            try:
                nl=SpectreNetlister(simtemp)
                sesh=psp.start_session(net_path=nl.get_netlist_file())
            except Exception as myexc:
                args=next((k for k in myexc.value.split("\n") if k.startswith("args:")))
                print(" ".join(eval(args.split(":")[1])))
                raise
            self._sessions[simname]=(nl,sesh)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            for simname in list(self._sessions.keys()):
                psp.stop_session(self._sessions.pop(simname)[1])


    def __del__(self):
        print("Deleting S MSS")
        super().__del__()

    def run_with_params(self, params={}):
        #print(f"Running with params {params}")
        results={}
        #import time
        for simname,(nl,sesh) in self._sessions.items():
            #print(f"Running {simname}")
            simtemp=self.simtemps[simname]
            nv=simtemp.model_paramset.update_and_return_changes(params)
            re_p_changed={nl.get_spectre_names_for_param(k):v for k,v in nv.items()}
            #print('setting params',re_p_changed,time.time())
            psp.set_parameters(sesh,re_p_changed)
            #print('running', time.time())
            results[simname]=simtemp.parse_return(nl.preparse_return(psp.run_all(sesh)))
            #print('done', time.time())
        return results
        
