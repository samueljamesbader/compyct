import typing
if typing.TYPE_CHECKING:
    from compyct.templates import TemplateGroup
import importlib
from pathlib import Path
import os
import importlib.resources as irsc
    
#from compyct.python_models import python_compact_models

class Netlister():
        
    @staticmethod
    def nstr_modeled_xtor(self,name,netd,netg,nets,netb,dt,inst_param_ovrd={}):
        raise NotImplementedError
        
    @staticmethod
    def nstr_VDC(name,netp,netm,dc):
        raise NotImplementedError
        
    @staticmethod
    def nstr_VAC(name,netp,netm,dc,ac=1):
        raise NotImplementedError

    def astr_altervdc(self,whichv, tovalue, name=None):
        raise NotImplementedError
        
    def astr_sweepvdc(self,whichv, start, stop, step, name=None):
        raise NotImplementedError
        
    def astr_sweepvac(self,whichv, start, stop, step, freq, name=None):
        raise NotImplementedError
        

class MultiSimSesh():
    @staticmethod
    def get_with_backend(simtemps: 'TemplateGroup', backend:str='ngspice', **kwargs) -> 'MultiSimSesh':
        try:
            backend_module=importlib.import_module('.'+backend+"_backend",package=__package__)
        except Exception as e:
            backends=[f.name.split("_backend")[0] for f in Path(__file__).parent.glob("*_backend.py")]
            raise Exception(f"Unrecognized backend {backend}, valid options are: {','.join(backends)}")
        return next(getattr(backend_module,k)(simtemps,**kwargs) for k in dir(backend_module)
             if k.lower()==(backend.lower()+"multisimsesh")
                 and issubclass(getattr(backend_module,k),MultiSimSesh))

    def __init__(self, simtemps: 'TemplateGroup'):
        self.simtemps: TemplateGroup=simtemps
        self._sessions: dict[str,typing.Any]={}
        
    def __enter__(self):
        print("Opening simulation session(s)")
        assert len(self._sessions)==0, "Previous sessions exist somehow!!"
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing sessions")
        
    def __del__(self):
        if len(self._sessions):
            print("Somehow deleted MultiSimSesh without closing sessions."\
                  "  That's bad but I can try to handle it.")
            self.__exit__(None,None,None)

    def run_with_params(self, params={}):
        raise NotImplementedError

def get_va_path(vaname):
    vapath=Path(os.environ['COMPYCT_VA_PATH'])/vaname
    if vapath.exists():
        return vapath
    else:
        # Okay to assume these va are in a real filesystem because compyct is marked as not zip-safe
        with irsc.as_file(irsc.files('compyct.examples')) as example_files:
            if (example_va:=(example_files/"standard_models/vacode"/vaname)).exists():
                return example_va
            else:
                raise Exception(f"Can't find {vaname} in {os.environ['COMPYCT_VA_PATH']} or built-in examples.")

def get_va_paths():
    user_va=list(Path(os.environ['COMPYCT_VA_PATH']).glob("*.va"))

    # Okay to assume these va are in a real filesystem because compyct is marked as not zip-safe
    with irsc.as_file(irsc.files('compyct.examples')) as example_files:
        ex_va=list((example_files/"standard_models/vacode").glob("*.va"))

    return user_va+ex_va

# class PythonMultiSimSesh(MultiSimSesh):
#     def run_with_params(self,params={}):
#         results={}
#         for simname,simtemp in self.simtemps.items():
#             re_p_changed=simtemp.update_paramset_and_return_spectre_changes(params)
#             results[simname]=python_compact_models[simtemp.model_paramset.model].run_all()
#         return results
            