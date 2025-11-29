from functools import partial
from scipy.constants import k as kb

import pyspectre as psp
from tempfile import NamedTemporaryFile
from compyct.templates import SimTemplate
from .backend import Netlister, MultiSimSesh, get_va_path
from compyct.paramsets import ParamPlace
from ..util import logger
from .spectre_util import n2scs


class SpectreNetlister(Netlister):
    GND='0'

    spectre_format:str = None
    
    def __init__(self,template: SimTemplate):#, additional_includes=[], override_model_subckt=None):
        self.simtemp: SimTemplate=template
        self._tf = None
        self._analysiscount=0
        self.modelcard_name=f"{self.simtemp._patch.model}_standin" if self.simtemp._patch else "generic_model_standin"
        #self.additional_includes=additional_includes
        #self.override_model_subckt=override_model_subckt

    #@staticmethod
    #def nstr_param(params):
    #    return f"parameters "+\
    #        " ".join([f"{k}={v}" for k,v in params.items()])

    def nstr_modeled_xtor(self,name,netmap,inst_param_ovrd={},internals_to_save=[]):
        assert len(inst_param_ovrd)==0
        #assert len(internals_to_save)==0, "Haven't implemented internal saving for spectre backend"
        patch=self.simtemp._patch
        for tterm in ['t','dt']:
            if tterm in patch.terminals: assert netmap.get(tterm,None) is None
        #terms=[t for t in patch.param_set.terminals if t!='dt']
        terms = " ".join([netmap.get(t,self.unique_term()) for t in patch.terminals])
        if True:
        #    assert ps.terminals==['d','g','s','b']
            #terms=" ".join([{'d':netd,'g':netg,'s':nets,'b':netb}[k] for k in terms])
        #if self.override_model_subckt is None:
            inst_paramstr=' '.join(f'{k}=instparam_{k}'\
                    for k in patch.filled().to_base().break_into_model_and_instance()[1])
            return f"X{name} ({terms})"\
                        f" {self.modelcard_name} {inst_paramstr}"
        #else:
        #    return f"X{name} ({netd} {netg} {nets} {netb})"\
        #                f" {self.override_model_subckt}"
        
    @staticmethod
    def nstr_VDC(name,netp,netm,dc):
        return f"V{name} ({netp} {netm}) vsource dc={n2scs(dc)} type=dc"
    
    @staticmethod
    def nstr_IDC(name,netp,netm,dc):
        return f"I{name} ({netp} {netm}) isource dc={n2scs(dc)} type=dc"
        
    @staticmethod
    def nstr_VAC(name,netp,netm,dc,ac=1):
        return f"V{name} ({netp} {netm}) vsource dc={n2scs(dc)} mag={n2scs(ac)} type=dc"

    @staticmethod
    def nstr_port(name,netp,netm,dc,portnum,z0=50,ac=0):
        #if ac!=0: print(f"Setting useless AC on port {portnum}")
        return f"PORT{portnum} ({netp} {netm} portdc{portnum}) port r={n2scs(z0)}\n"\
               f"VPort{portnum} (portdc{portnum} {netm}) vsource dc={n2scs(dc)} mag={n2scs(ac)}"

    def astr_altervportdc(self, whichv, tovalue, portnum, name=None):
        if name is None:
            name = f"alter{self._analysiscount}"
            self._analysiscount += 1
        return f"{name} alter dev=VPort{portnum} param=dc value={n2scs(tovalue)}"

    def nstr_iabstol(self,abstol):
        return f"simulatorOptions options iabstol={n2scs(abstol)}"

    def nstr_temp(self, temp=27, tnom=27):
        return f"simulatorOptions options temp={n2scs(temp)} tnom={n2scs(tnom)}"

    def nstr_res(self,name,netp,netm,r):
        return f"R{name} ({netp} {netm}) resistor r={n2scs(r)}"
    
    def nstr_cap(self,name,netp,netm,c):
        return f"C{name} ({netp} {netm}) capacitor c={n2scs(c)}"

    def nstr_iprobe(self,name,netp,netm):
        return f"{name} ({netp} {netm}) iprobe"

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
    
    def astr_sweepidc(self,whichi, start, stop, step, name=None):
        if name is None:
            name=f"sweep{self._analysiscount}"
            self._analysiscount+=1
        return f"{name} dc dev=I{whichi} param=dc"\
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
            name = f"sweepspar{self._analysiscount}"
            self._analysiscount += 1
        if fstart == fstop:
            return f"{name} sp ports=[PORT1 PORT2] freq={n2scs(fstart)} annotate=status"
        else:
            narg = {'lin': points, 'dec': pts_per_dec}[sweep_option]
            return f"{name} sp ports=[PORT1 PORT2] start={n2scs(fstart)} stop={n2scs(fstop)} {sweep_option}={narg} annotate=status"
    def astr_sparnoise(self, fstart, fstop, pts_per_dec=None, points=None, sweep_option='dec', name=None):
        self.spectre_format='psfascii'
        if name is None:
            name=f"sweepspar{self._analysiscount}"
            self._analysiscount+=1
        if fstart==fstop:
            #raise Exception("Spectre+Nutmeg fails for s-parameter noise for newer spectre versions (once 20.10.382.isr12 would complain but mostly work(?) for single-freq)")
            return f"dc_{name} dc\n{name} sp ports=[PORT1 PORT2] freq={n2scs(fstart)} annotate=status donoise=yes noisedata=cy iprobe=PORT1 oprobe=PORT2"
        else:
            #raise Exception("Spectre+Nutmeg fails for multi-freq s-parameter noise")
            narg = {'lin': points, 'dec': pts_per_dec}[sweep_option]
            return f"dc_{name} dc\n{name} sp ports=[PORT1 PORT2] start={n2scs(fstart)} stop={n2scs(fstop)} {sweep_option}={narg} annotate=status donoise=yes noisedata=cy iprobe=PORT1 oprobe=PORT2"

    def astr_noise(self, outprobe, vsrc, fstart, fstop, pts_per_dec=None, points=None, sweep_option='dec', name=None):
        if name is None:
            name=f"sweepnoise{self._analysiscount}"
            self._analysiscount+=1
        if fstart==fstop:
            return f"dc_{name} dc\n{name} noise iprobe={vsrc} oprobe={outprobe} freq={n2scs(fstart)} annotate=status"
        else:
            narg = {'lin': points, 'dec': pts_per_dec}[sweep_option]
            return f"dc_{name} dc\n{name} noise iprobe={vsrc} oprobe={outprobe} start={n2scs(fstart)} stop={n2scs(fstop)} {sweep_option}={narg} annotate=status"
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
            # Reset the format to None
            self.spectre_format=None

            self._tf=NamedTemporaryFile(prefix=self.simtemp.__class__.__name__,mode='w')
            self._tf.write(f"// {self.simtemp.__class__.__name__}\n")
            self._tf.write(f"simulator lang = spectre\n")
            # for i in self.additional_includes:
            #     if type(i)==str:
            #         self._tf.write(f"include \"{i}\"\n")
            #     else:
            #         self._tf.write(
            #             f"include \"{i[0]}\" {' '.join(i[1:])}\n")
            for i in self.simtemp.scs_includes:
                if type(i)==str: self._tf.write(f"include \"{i}\"\n")
                else: self._tf.write(f"include \"{i[0]}\" {' '.join(i[1:])}\n")
            for i in self.simtemp.va_includes:
                assert type(i)==str; self._tf.write(f"ahdl_include \"{get_va_path(i)}\"\n")


            from compyct.backends.spectre_template import SpectreSimTemplate
            if isinstance(self.simtemp,SpectreSimTemplate):
                self._tf.write(self.simtemp.get_netlist_template(self))
                self._tf.flush()
            else:
                patch=self.simtemp._patch
                if patch is not None:
                            
                    self._tf.write(f"model {self.modelcard_name} {patch.model}\n")
                    paramlinedict={f"modparam_{k}":v for k,v in patch.filled().to_base().break_into_model_and_instance()[0].items()}
                    self._tf.write(
                        f"parameters "+\
                        ' '.join([f'{k}={n2scs(v)}' for k,v in paramlinedict.items()])\
                        +"\n\n")
                    instance_params = {k: v for k, v in patch.filled().to_base().break_into_model_and_instance()[1].items()}
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
                        " ".join((f"{k}=modparam_{k}" for k in patch.filled().to_base().break_into_model_and_instance()[0]))\
                        +"\n}\n\n")
                self._tf.write("\n".join(self.simtemp.get_analysis_listing(self))+"\n")
                self._tf.flush()
        # If nothing has demanded 'psfascii', use nutbin
        if self.spectre_format is None: self.spectre_format='nutbin'
        return self._tf.name

    def get_spectre_names_for_param(self,param):
        prefix={ParamPlace.MODEL.value:'mod',ParamPlace.INSTANCE.value:'inst'}\
                    [self.simtemp._patch.to_base().param_set.get_place(param).value]+'param_'
        return prefix+param
        
    def preparse_return(self,result):
        def standardize_col(k,sweepname=None):
            if '`noise' in sweepname:
                return {'out':'onoise [A/sqrt(Hz)]','in':'inoise [V/sqrt(Hz)]','gain':'gain [A/V]'}.get(k,k)
            else:
                if k=='dc':
                    return 'v-sweep'
                elif k in ['s11','s12','s21','s22']:
                    return k.upper()
                else:
                    return (k.lower().replace(":","#"))
        def standardize_swname(k):
            #print("SWEEP NAME: ",k)
            return k.split("`")[1].split("'")[0]
        def standardize_norms(df):
            # Spectre normalizes the noise correlation matrix by the available noise power of the source port: 4*kb*(290K)
            # But I have not found this documented anywhere. So I'll unnormalize it here before template sees it
            # For noise fig calculations, make sure source port noisetemp is set to 16.85C=290K! [default for spectre]
            for c in ['cy11','cy12','cy21','cy22']:
                if c in df.columns:
                    df[c]=df[c]*(4*kb*290)
            return df
        return {standardize_swname(sweepname):standardize_norms(sweepdata.rename(columns=partial(standardize_col,sweepname=sweepname)))
                    for sweepname,sweepdata in result.items()} 
    

class SpectreMultiSimSesh(MultiSimSesh):

    def print_netlists(self, only=None, file=None):
        for simname,simtemp in self.simtemps.items():
            if only is not None and simname not in only: continue
            print(f"###################### {simname} #############")
            nl=SpectreNetlister(simtemp, **self._netlist_kwargs)
            with open(nl.get_netlist_file(),'r') as f:
                print(f.read(), file=file)
            
    
    def __enter__(self):
        super().__enter__()
        logger.debug("Entering Spectre MultiSimSesh")
        for simname,simtemp in self.simtemps.items():
            logger.debug(f"  {simname}")
            nl=SpectreNetlister(simtemp, **self._netlist_kwargs)
            net_path=nl.get_netlist_file()
            assert nl.spectre_format is not None, "Spectre format not set by netlister"
            try:
                sesh=psp.start_session(net_path=net_path, timeout=600, format=nl.spectre_format, keep_log=True) # longer timeout in-case compiling ahdl
            except Exception as myexc:
                with open(net_path,'r') as f:
                    from compyct import CACHE_DIR
                    of=CACHE_DIR/"fails/last_failed_spectre_netlist.scs"
                    of.parent.mkdir(parents=True,exist_ok=True)
                    with open(of,'w') as ff:
                        ff.write(f.read())
                    print(f"Failed netlist available in {of}:")
                if hasattr(myexc,'value'):
                    args=next((k for k in myexc.value.split("\n") if k.startswith("args:")))
                    print(" ".join(eval(args.split(":")[1])))
                else:
                    print(myexc)
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
        results={}
        #import time
        for simname,(nl,sesh) in self._sessions.items():
            if only_temps is not None and simname not in only_temps: continue
            logger.debug(f"Running {simname}")
            simtemp=self.simtemps[simname]
            if params is not None:

                with simtemp.tentative_base_deltas(params) as deltas:
                    if full_resync: deltas=simtemp._patch
                    logger.debug(f"Param changes: {deltas}")
                    psp.set_parameters(sesh,{nl.get_spectre_names_for_param(k):n2scs(v)
                                             for k,v in deltas.items()})
                    logger.debug(f"Done param changes")
            #print('running', time.time())
            results[simname]=simtemp.postparse_return(simtemp.parse_return(nl.preparse_return(psp.run_all(sesh))))
            #print('done', time.time())
        return results
        
