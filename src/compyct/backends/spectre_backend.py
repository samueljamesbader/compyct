import pyspectre as psp
from tempfile import NamedTemporaryFile
from compyct.templates import SimTemplate
from .backend import Netlister, MultiSimSesh, get_va_path
from compyct.paramsets import ParamPlace
from ..util import logger
from .spectre_util import n2scs


class SpectreNetlister(Netlister):
    GND='0'
    
    def __init__(self,template: SimTemplate):#, additional_includes=[], override_model_subckt=None):
        self.simtemp: SimTemplate=template
        self._tf = None
        self._analysiscount=0
        self.modelcard_name=f"{self.simtemp._patch.model}_standin"
        #self.additional_includes=additional_includes
        #self.override_model_subckt=override_model_subckt

    #@staticmethod
    #def nstr_param(params):
    #    return f"parameters "+\
    #        " ".join([f"{k}={v}" for k,v in params.items()])
        
    def nstr_modeled_xtor(self,name,netd,netg,nets,netb,dt,inst_param_ovrd={},internals_to_save=[]):
        assert len(inst_param_ovrd)==0
        #assert len(internals_to_save)==0, "Haven't implemented internal saving for spectre backend"
        assert dt is None
        patch=self.simtemp._patch
        terms=[t for t in patch.param_set.terminals if t!='dt']
        if True:
        #    assert ps.terminals==['d','g','s','b']
            terms=" ".join([{'d':netd,'g':netg,'s':nets,'b':netb}[k] for k in terms])
        #if self.override_model_subckt is None:
            inst_paramstr=' '.join(f'{k}=instparam_{k}'\
                    for k in patch.filled().break_into_model_and_instance()[1])
            return f"X{name} ({terms})"\
                        f" {self.modelcard_name} {inst_paramstr}"
        #else:
        #    return f"X{name} ({netd} {netg} {nets} {netb})"\
        #                f" {self.override_model_subckt}"
        
    @staticmethod
    def nstr_VDC(name,netp,netm,dc):
        return f"V{name} ({netp} {netm}) vsource dc={n2scs(dc)} type=dc"
        
    @staticmethod
    def nstr_VAC(name,netp,netm,dc,ac=1):
        return f"V{name} ({netp} {netm}) vsource dc={n2scs(dc)} mag={n2scs(ac)} type=dc"

    @staticmethod
    def nstr_port(name,netp,netm,dc,portnum,z0=50):
        return f"PORT{portnum} ({netp} {netm} portdc{portnum}) port r={n2scs(z0)}\n"\
               f"VPort{portnum} (portdc{portnum} {netm}) vsource dc={n2scs(dc)}"
    
    def nstr_iabstol(self,abstol):
        return f"simulatorOptions options iabstol={n2scs(abstol)}"

    def nstr_temp(self, temp=27, tnom=27):
        return f"simulatorOptions options temp={n2scs(temp)} tnom={n2scs(tnom)}"

    def astr_altervdc(self,whichv, tovalue, name=None):
        if name is None:
            name=f"alter{self._analysiscount}"
            self._analysiscount+=1
        return f"{name} alter dev=V{whichv} param=dc value={n2scs(tovalue)}"
        
    def astr_sweepvdc(self,whichv, start, stop, step, name=None):
        if name is None:
            name=f"sweep{self._analysiscount}"
            self._analysiscount+=1
        return f"{name} dc dev=V{whichv} param=dc"\
                f" start={n2scs(start)} stop={n2scs(stop)} step={n2scs(step)}"
        
    def astr_sweepvac(self,whichv, start, stop, step, freq, name=None):
        if name is None:
            name=f"sweep{self._analysiscount}"
            self._analysiscount+=1
        return f"{name} ac dev=V{whichv} param=dc start={n2scs(start)}"\
            f" stop={n2scs(stop)} step={n2scs(step)} freq={n2scs(freq)}"
        
    #def astr_spar(self, pts_per_dec, fstart, fstop, name=None):
    def astr_spar(self, fstart, fstop, pts_per_dec=None, points=None, sweep_option='dec', name=None):
        if name is None:
            name=f"sweepspar{self._analysiscount}"
            self._analysiscount+=1
        narg = {'lin': points, 'dec': pts_per_dec}[sweep_option]
        return f"{name} sp ports=[PORT1 PORT2] start={n2scs(fstart)} stop={n2scs(fstop)} {sweep_option}={narg} annotate=status"

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
            patch=self.simtemp._patch
            # for i in self.additional_includes:
            #     if type(i)==str:
            #         self._tf.write(f"include \"{i}\"\n")
            #     else:
            #         self._tf.write(
            #             f"include \"{i[0]}\" {' '.join(i[1:])}\n")
            if patch is not None:
                for i in patch.param_set.scs_includes:
                    if type(i)==str:
                        self._tf.write(f"include \"{i}\"\n")
                    else:
                        self._tf.write(
                            f"include \"{i[0]}\" {' '.join(i[1:])}\n")
                for i in patch.param_set.va_includes:
                    assert type(i)==str
                    self._tf.write(f"ahdl_include \"{get_va_path(i)}\"\n")
                        
                self._tf.write(f"model {self.modelcard_name} {patch.model}\n")
                paramlinedict={f"modparam_{k}":v for k,v in patch.filled().break_into_model_and_instance()[0].items()}
                self._tf.write(
                    f"parameters "+\
                    ' '.join([f'{k}={n2scs(v)}' for k,v in paramlinedict.items()])\
                    +"\n\n")
                instance_params = {k: v for k, v in patch.filled().break_into_model_and_instance()[1].items()}
                if len(instance_params):
                    self._tf.write(f"parameters "+\
                                       ' '.join(f'instparam_{k}={n2scs(v)}'
                                            for k,v in instance_params.items())+\
                                   "\n")
            self._tf.write("\n".join(self.simtemp.get_schematic_listing(self))+"\n")
            if patch is not None:
                self._tf.write(
                    f"set_modparams altergroup {{\nmodel {self.modelcard_name}"\
                    f" {patch.model} "+\
                    " ".join((f"{k}=modparam_{k}" for k in patch.filled().break_into_model_and_instance()[0]))\
                    +"\n}\n\n")
            self._tf.write("\n".join(self.simtemp.get_analysis_listing(self))+"\n")
            self._tf.flush()
        return self._tf.name

    def get_spectre_names_for_param(self,param):
        prefix={ParamPlace.MODEL.value:'mod',ParamPlace.INSTANCE.value:'inst'}\
                    [self.simtemp._patch.param_set.get_place(param).value]+'param_'
        return prefix+param
        
    def preparse_return(self,result):
        def standardize_col(k):
            if k=='dc':
                return 'v-sweep'
            elif k in ['s11','s12','s21','s22']:
                return k.upper()
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
            nl=SpectreNetlister(simtemp, **self._netlist_kwargs)
            with open(nl.get_netlist_file(),'r') as f:
                print(f.read())
            
    
    def __enter__(self):
        super().__enter__()
        for simname,simtemp in self.simtemps.items():
            try:
                logger.debug(f"  {simname}")
                nl=SpectreNetlister(simtemp, **self._netlist_kwargs)
                sesh=psp.start_session(net_path=nl.get_netlist_file(), timeout=600) # longer timeout in-case compiling ahdl
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

    def run_with_params(self, params=None, full_resync=False, only_temps=None):
        raise NotImplementedError("Need to update this to match base_delta stuff")
        results={}
        #import time
        for simname,(nl,sesh) in self._sessions.items():
            if only_temps is not None and simname not in only_temps: continue
            logger.debug(f"Running {simname}")
            simtemp=self.simtemps[simname]
            if params is not None:
                sparams=params.translated_to(simtemp._patch.param_set)

                with simtemp.tentative_deltas(sparams) as deltas:
                    if full_resync: deltas=simtemp._patch
                    logger.debug(f"Param changes: {deltas}")
                    logger.debug(f"Done param changes")
                    psp.set_parameters(sesh,{nl.get_spectre_names_for_param(k):n2scs(v)
                                             for k,v in deltas.items()})
            #print('running', time.time())
            results[simname]=simtemp.postparse_return(simtemp.parse_return(nl.preparse_return(psp.run_all(sesh))))
            #print('done', time.time())
        return results
        
