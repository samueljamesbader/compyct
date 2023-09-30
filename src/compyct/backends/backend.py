import typing
if typing.TYPE_CHECKING:
    from compyct.templates import TemplateGroup
    
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
    def __init__(self, simtemps: 'TemplateGroup'):
        self.simtemps: TemplateGroup=simtemps
        self._sessions: dict[str,psp.Session]={}
        
    def __enter__(self):
        print("Opening simulation session(s)")
        assert len(self._sessions)==0, "Previous sessions exist somehow!!"
        
    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing sessions")
        
    def __del__(self):
        if len(self._sessions):
            print("Somehow deleted MultiSimSesh without closing sessions."\
                  "  That's bad but I can try to handle it.")
            self.__exit__(None,None,None)

    def run_with_params(self, params={}):
        raise NotImplementedError

# class PythonMultiSimSesh(MultiSimSesh):
#     def run_with_params(self,params={}):
#         results={}
#         for simname,simtemp in self.simtemps.items():
#             re_p_changed=simtemp.update_paramset_and_return_spectre_changes(params)
#             results[simname]=python_compact_models[simtemp.model_paramset.model].run_all()
#         return results
            