from collections import UserDict
from pathlib import Path
import csv
import io
import re
import os
from copy import copy as _copy
import numpy as np

from compyct.backends.backend import get_va_path


def spicenum_to_float(spicenum):
    try:
        return float(spicenum)
    except:
        assert spicenum[-1]!="M",\
            f"'M' is an ambiguous unit between spice (case-insensitive)"\
            f"and spectre (case-sensitive)"
        spicenum=spicenum.lower().replace("meg","M")
        return float(spicenum[:-1])*\
            {'f':1e-15,'p':1e-12,'n':1e-9,
             'u':1e-6,'m':1e-3,'k':1e3,'M':1e6}\
                    [spicenum[-1]]

def float_to_spicenum(fl):
    if type(fl) is str:
        fl=spicenum_to_float(fl)
    if fl==0: return '0'
    ord=np.clip(np.floor(np.log10(np.abs(fl))/3)*3,-12,9)
    si={-12:'p',-9:'n',-6:'u',-3:'m',0:'',3:'k',6:'meg',9:'G'}[ord]
    return f'{(fl/10**ord):g}{si}'

from enum import Enum
class ParamPlace(Enum):
    MODEL = 1
    INSTANCE =2

class ParamSet():
    _shared_paramdict=None
    scs_includes=[]
    va_includes=[]

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
    def get_display_units(self,param) -> str:
        d=self._shared_paramdict[param]
        return d.get('display_units',d['units'])
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

    def get_scale(self,param):
        if 'scale' in (pd:=self._shared_paramdict[param]):
            return pd['scale']
        else:
            return 1

    def update_and_return_changes(self, new_values):
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
        ndf=self.get_non_default_values()
        return f"<{self.model}:"\
                    +",".join(f"{k}={ndf[k]}" for k in sorted(ndf) )\
                +">"

    def translate_to(self,param_dict,other):
        # TODO: some (eg Guessed) allow multiple types from same class so this logic is bad
        if other.__class__==self.__class__:
            return param_dict
        else:
            raise NotImplementedError

    def as_patch(self, ignore_keys=[]):
        return ParamPatch(self,{k:v for k,v in self.get_values().items() if k not in ignore_keys})

class ParamPatch(UserDict):

    def __init__(self, ps_example:ParamSet, *args, **kwargs):
        self._ps_example=ps_example
        super().__init__(*args,**kwargs)

    def patch_paramset_and_return_changes(self, ps: ParamSet):
        pd=self._ps_example.translate_to(self, ps)
        return ps.update_and_return_changes(pd)
    def patch_paramset_and_return_it(self, ps: ParamSet):
        pd=self._ps_example.translate_to(self, ps)
        ps.update_and_return_changes(pd)
        return ps

    #def generate_from_example(self):
    #    self._ps_example.copy_with_changes()

class GuessedInstanceValueParamSet(ParamSet):
    _shared_paramdicts={}
    terminals=['d','g','s','b']

    def __init__(self, model, file, section="",**kwargs):
        super().__init__(model=model,**kwargs)
        self.file=file
        self.section=section
        self.scs_includes=[(str(file),"section="+section)]
        if (model,file,section) in GuessedInstanceValueParamSet._shared_paramdicts:
            return
        in_section=""
        defaults=None
        with open(file) as f:
            for l in f:
                if l.startswith("section"):
                    in_section=l.strip().split(" ")[1]

                if l.startswith(f"subckt {model}"):
                    if in_section==self.section:
                        l=next(f).strip()
                        assert l.startswith("parameters ")
                        assert defaults is None
                        defaults=dict([eq.split("=") for eq in l.split()[1:]])
        defaults['m']=1
        GuessedInstanceValueParamSet._shared_paramdicts[(model,file,section)]=\
            {k:{'default':v,'units':'A.U.'} for k,v in defaults.items()}
    @property
    def _shared_paramdict(self):
        return self._shared_paramdicts[(self.model,self.file,self.section)]

    def get_dtype(self, param):
        return self._shared_paramdict[param].get('dtype',float)
    def get_place(self, param):
        return ParamPlace.INSTANCE



class GuessedDSPFParamSet(ParamSet):
    _shared_paramdicts={}
    _terminals={}

    def __init__(self, subckt, supply_model, file, **kwargs):
        super().__init__(model=subckt, **kwargs)
        self.file=file
        self.scs_includes=supply_model.scs_includes+[str(file)]
        self.va_includes=supply_model.va_includes
        self.supply_model=supply_model
        if (self.model,file) in GuessedDSPFParamSet._shared_paramdicts:
            return
        defaults=None
        with open(file) as f:
            for l in f: 
                if l.startswith(f".SUBCKT") and l.split()[1]==subckt:
                    GuessedDSPFParamSet._terminals[(self.model,file)]=l.split()[2:]
                    assert defaults is None
                    defaults={}
        assert defaults is not None
        GuessedDSPFParamSet._shared_paramdicts[(self.model,file)]=defaults

    @property
    def _shared_paramdict(self):
        return self._shared_paramdicts[(self.model,self.file)]
        
    def get_total_device_width(self):
        return self.supply_model.get_total_device_width()
        
    @property
    def terminals(self):
        return self._terminals[(self.model,self.file)]


class CMCParamSet(ParamSet):
    """ Abstract superclass for Verilog-A standard models, so far tried only on CMC models (MVSG and ASMHEMT).

    To use this class, you *must* subclass it, because the default parameters dictionary is stored at the class level,
    shared between all instances.

    # Todo: make this work more like GuessInstanceValueParamSet where it is safe to use without subclassing
    """
    def __init__(self,model,vaname,**kwargs):
        super().__init__(model=model,**kwargs)
        if self._shared_paramdict is not None:
            return

        vapath=get_va_path(vaname=vaname)
        self.__class__.scs_includes=[]
        self.__class__.va_includes=[vaname]

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
        return {'R':float,'I':int}[self._shared_paramdict[param]['macro'][2]]
    def get_place(self,param) -> ParamPlace:
        return [ParamPlace.INSTANCE,ParamPlace.MODEL]\
                    [self._shared_paramdict[param]['macro'].startswith('M')]
    
    def get_bounds(self, param, null=None):
        if null is np.inf:
            upper_null=null
            lower_null=-null
        else:
            upper_null=lower_null=null
        deets=self._shared_paramdict[param]
        match deets['macro'][3:]:
            case 'co':
                lower=spicenum_to_float(deets['lower'])
                upper=upper_null
            case 'cc':
                lower=spicenum_to_float(deets['lower'])
                upper=spicenum_to_float(deets['upper'])
            case 'oo':
                lower=spicenum_to_float(deets['lower'])
                upper=spicenum_to_float(deets['upper'])
            case 'oz':
                lower=spicenum_to_float(deets['default'])*.01
                upper=upper_null
            case 'cz':
                lower=0
                upper=upper_null
            case 'sw':
                lower=0
                upper=1
            case _:
                print(f"Not sure what to do for bounds with macro {deets['macro']} for param {param}")
                #if 'lower' in deets and deets['lower'] is not None:
                lower=lower_null
                upper=upper_null
        if null is None:
            if upper is not null and np.isinf(upper): upper=null
        if deets['macro']=='MPRnb':
            step=.1
        else:
            step=np.abs(spicenum_to_float(deets['default']))*.01
            if step==0:
                step=.1
        return (lower,step,upper)
        
class MVSGParamSet(CMCParamSet):
    terminals=['d','g','s','b']
    def __init__(self,**kwargs):
        super().__init__(model='mvsg_cmc',vaname="mvsg_cmc_3.0.0.va",**kwargs)

    def get_total_device_width(self):
        return spicenum_to_float(self.get_value('w'))*\
                spicenum_to_float(self.get_value('ngf'))
        
class ASMHEMTParamSet(CMCParamSet):
    terminals=['d','g','s','b','dt']
    def __init__(self,**kwargs):
        super().__init__(model='asmhemt',vaname="asmhemt.va",**kwargs)

        # Overrides where ASMHEMT 101.2.0 documentation is incorrect
        self._shared_paramdict['cgso']['units']='F/m' # Documentation says F
        self._shared_paramdict['cgdo']['units']='F/m' # Documentation says F
        self._shared_paramdict['ua']['units']='m/V' # Documentation says 1/V
        self._shared_paramdict['ub']['units']='m^2/V^2' # Documentation says 1/V^2
        self._shared_paramdict['ns0accs']['units']='1/m^2' # Documentation says C/m^2
        self._shared_paramdict['ns0accd']['units']='1/m^2' # Documentation says C/m^2
        self._shared_paramdict['vsataccs']['units']='m/s' # Documentation says cm/s

        # Pint needs 'W' (for Watt) to be capitalized
        self._shared_paramdict['rth0']['units']=self._shared_paramdict['rth0']['units'].replace('w','W')
        self._shared_paramdict['cth0']['units']=self._shared_paramdict['cth0']['units'].replace('w','W')

    def get_total_device_width(self):
        return spicenum_to_float(self.get_value('w'))*\
                spicenum_to_float(self.get_value('nf'))
        
# TODO: actually assess terminals instead of defining here