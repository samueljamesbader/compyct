from pathlib import Path
import csv
import io
import re
import os
from copy import copy as _copy
import numpy as np

def spicenum_to_float(spicenum):
    try:
        return float(spicenum)
    except:
        return float(spicenum[:-1])*\
            {'f':1e-15,'p':1e-12,'n':1e-9,
             'u':1e-6,'m':1e-3,'k':1e3,'M':1e6}\
                    [spicenum[-1]]
        
def float_to_spicenum(fl):
    ord=np.floor(np.log10(fl)/3)*3
    si={-12:'p',-9:'n',-6:'u',-3:'m',0:'',3:'k',6:'M',9:'G'}[ord]
    return f'{(fl/10**ord):g}{si}'

from enum import Enum
class ParamPlace(Enum):
    MODEL = 1
    INSTANCE =2
    
class ParamSet():
    _shared_paramdict=None
    includes=[]
    
    def __init__(self,model,**kwargs):
        self.model=model
        self._values={}
        for k,v in kwargs.items():
            if k in self._shared_paramdict:
                self._values[k]=v
            else:
                raise Exception(f"Tried to set parameter {k} in {self.__class__.__name__} but it's not in paramset")
        
    def __iter__(self):
        return iter(self._shared_paramdict)
    def get_dtype(self,param) -> type:
        raise NotImplementedError
    def get_place(self,param) -> ParamPlace:
        raise NotImplementedError

    def get_units(self,param) -> str:
        return self._shared_paramdict[param]['units']
    def get_default(self,param):
        if param not in self._shared_paramdict:
            raise Exception(f"Param {param} not in {self.__class__.__name__}")
        return self._shared_paramdict[param]['default']
    def get_value(self,param):
        return self._values.get(param,self.get_default(param))
    def copy(self):
        cp=_copy(self)
        cp._values=self._values.copy()
        return cp
        
    def update_and_return_changes(self,new_values):
        changed={}
        for k,v in new_values.items():
            if v!=self.get_value(k):
                self._values[k]=v
                changed[k]=v
        return changed
                
    def copy_with_changes(self,new_values):
        other=self.copy()
        other.update_and_return_changes(new_values)
        return other

    def get_values(self):
        return {k:self.get_value(k) for k in self}
    
    def get_non_default_values(self):
        return {k:v for k,v in self._values.items()
                            if v!=self.get_default(k)}

    def __repr__(self):
        return f"<{self.model}:"\
                    +",".join(f"{k}={v}"
                        for k,v in self.get_non_default_values().items() )\
                +">"
                
        
class GuessedInstanceValueParamSet(ParamSet):
    _shared_paramdicts={}
    def __init__(self, model, file, section="",**kwargs):
        super().__init__(model=model,**kwargs)
        self.file=file
        self.section=section
        self.includes=[(str(file),"section="+section)]
        if (model,file,section) in GuessedInstanceValueParamSet._shared_paramdicts:
            return
        in_section=""
        with open(file) as f:
            for l in f:
                if l.startswith("section"):
                    in_section=l.strip().split(" ")[1]
                
                if l.startswith(f"subckt {model}"):
                    if in_section==self.section:
                        l=next(f).strip()
                        assert l.startswith("parameters ")
                        defaults=dict([eq.split("=") for eq in l.split()[1:]])
        GuessedInstanceValueParamSet._shared_paramdicts[(model,file,section)]=\
            {k:{'default':v,'units':'A.U.'} for k,v in defaults.items()}
    @property
    def _shared_paramdict(self):
        return self._shared_paramdicts[(self.model,self.file,self.section)]
        
    def get_dtype(self, param):
        return self._shared_paramdict[param].get('dtype',float)
    def get_place(self, param):
        return ParamPlace.INSTANCE

class CMCParamSet(ParamSet):
    def __init__(self,model,vapath,**kwargs):
        super().__init__(model=model,**kwargs)
        if self._shared_paramdict is not None:
            return
        self.includes=[str(vapath)]
        
        with open(vapath,'r') as f:        
            # Collect the model parameter and instance parameter definition lines
            # Tweak them slightly so they can be read by csv reader with the macro
            # name as the first entry in each line (easier than trying to split
            # by commas since descriptions may contain commas)
            lines=[]
            for l in (l.strip() for l in f):
                
                if l.startswith("`MP") or l.startswith("`IP"):
                    assert l[-1]==')'
        
                    # Make the macro part another comma-separated part of the line
                    macropart,rest=l[1:].split("(",maxsplit=1)
                    l=macropart+","+rest
        
        
                    if "$" in l:
                        #print(l)
                        depth=0
                        for i,c in enumerate(l[l.find("$"):]):
                            if c=="(": depth+=1
                            if c==")": depth-=1
                            if c=="," and depth==0: break
                        l=l.replace(l[l.find("$"):][:i],"???")
                    
                    # Remove double-whitespace
                    # (helps CSV reader when there's mixed quoting)
                    lines.append(" ".join(l.split()).replace(", \"",",\""))
                    
            f=io.StringIO("\n".join(lines))
        
            paramset={}
            for line in csv.reader(f):
                match len(line):
                   case 5:
                       macro,name,default,units,desc=[x.strip() for x in line]
                       paramset[name]=\
                           {'macro':macro,'default':default,'units':units,'desc':desc}
                   case 7:
                       macro,name,default,units,lower,upper,desc=\
                           [x.strip() for x in line]
                       paramset[name]=\
                           {'macro':macro,'default':default,'units':units,
                            'desc':desc,'lower':lower,'upper':upper}
                   case _:
                       raise Exception("Can't read line, best guess is "+",".join(l))
            self.__class__._shared_paramdict=paramset
            
    def get_dtype(self,param) -> type:
        return {'':float}[self._shared_paramdict[param]['macro']]
    def get_place(self,param) -> ParamPlace:
        return [ParamPlace.INSTANCE,ParamPlace.MODEL]\
                    [self._shared_paramdict[param]['macro'].startswith('M')]
    
    def get_bounds(self,param):        
        deets=self._shared_paramdict[param]
        match deets['macro']:
            case 'MPRco':
                lower=spicenum_to_float(deets['lower'])
                upper=None
            case 'MPRcc':
                lower=spicenum_to_float(deets['lower'])
                upper=spicenum_to_float(deets['upper'])
            case 'MPRoo':
                lower=spicenum_to_float(deets['lower'])
                upper=spicenum_to_float(deets['upper'])
            case 'MPRoz':
                lower=spicenum_to_float(deets['default'])*.01
                upper=None
            case 'MPRcz':
                lower=0
                upper=None
            case _:
                print(f"Not sure what to do with macro {deets['macro']} for param {param}")
                lower=None
                upper=None
        if upper is not None and np.isinf(upper): upper=None
        if deets['macro']=='MPRnb':
            step=.1
        else:
            step=np.abs(spicenum_to_float(deets['default']))*.1
            if step==0:
                step=.1
        return (lower,step,upper)
        
class MVSGParamSet(CMCParamSet):
    def __init__(self,**kwargs):
        vapath=Path(os.environ['MODELFITPATH'])/\
                    "standard_models/vacode/mvsg_cmc_3.0.0.va"
        super().__init__(model='mvsg_cmc',vapath=vapath,**kwargs)
        
    def get_total_device_width(self):
        return spicenum_to_float(self.get_value('w'))*\
                spicenum_to_float(self.get_value('ngf'))
        
class ASMHEMTParamSet(CMCParamSet):
    def __init__(self,**kwargs):
        vapath=Path(os.environ['MODELFITPATH'])/\
                    "standard_models/vacode/asmhemt.va"
        super().__init__(model='asmhemt',vapath=vapath,**kwargs)
        
    def get_total_device_width(self):
        return spicenum_to_float(self.get_value('w'))*\
                spicenum_to_float(self.get_value('nf'))
        