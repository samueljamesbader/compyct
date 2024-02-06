from collections import UserDict
from functools import lru_cache
from pathlib import Path
import csv
import io
import re
import os
from copy import copy as _copy, deepcopy
from typing import Any, Union

import numpy as np
import yaml
from pint import DimensionalityError

from compyct import logger, ureg
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

    def __init__(self, model:str, terminals:list[str],
                 pdict:dict[str,dict[str,Any]], display_yaml:Union[Path,str,dict]=None,
                 scs_includes: list=[], va_includes: list[str]=[]
                 ):

        # direct features from model code
        self._pdict: dict[str,dict[str,Any]] = pdict

        # content to reference in netlist
        self.model:str = model
        self.terminals:list[str] = terminals
        self.scs_includes:list = scs_includes
        self.va_includes:list[str] = va_includes

        ###
        # display configurations
        ###
        if isinstance(display_yaml,Path) or type(display_yaml)==str:
            with open(display_yaml,'r') as f:
                self._disp_yaml=yaml.safe_load(f)
        else:
            self._disp_yaml=display_yaml or {}

        # Display units and overriden units
        for p,v in self._disp_yaml.get("unit_overrides",{}).items(): self._pdict[p]['units']=v
        for p,disp_units in self._disp_yaml.get('display_units',{}).items():
            assert p in self._pdict, f"Display units provided for unknown parameter {p}"
            self._pdict[p]['display_units']=disp_units
        for p in self._pdict:
            try:
                units=self._pdict[p]['units']
            except KeyError:
                raise Exception(f"No units provided for {p}")
            disp_units=self._pdict[p].get('display_units',self._pdict[p]['units'])
            try:
                disp_scale=(ureg.parse_expression(units)/ureg.parse_expression(disp_units)).to("").magnitude
            except DimensionalityError as e:
                raise Exception(f"Units of {units} and Display Units of {disp_units}" \
                                f" for {p} are not compatible. Pint says \"{str(e)}\"")
            except Exception as e:
                raise Exception(f"Unit error on {p}. Pint says \"{str(e)}\"")
            self._pdict[p]['display_scale']=disp_scale

        # Categorization
        self._cat_to_params: dict[str,list[str]] = self._disp_yaml.get('categories',{})
        categorized_params=[p for v in self._cat_to_params.values() for p in v if v!='Uncategorized']
        #assert 'Uncategorized' not in self._cat_to_params, "Category 'Uncategorized' will be created, do not supply"
        uncategorized=list(sorted([p for p in self._pdict if p not in categorized_params]))
        for p in list(uncategorized):
            if (cat:=self._pdict[p].get('category','Uncategorized'))!='Uncategorized':
                self._cat_to_params[cat].append(p)
                uncategorized.remove(p)
        self._cat_to_params['Uncategorized']=uncategorized
        assert len(set(categorized_params))==len(categorized_params), "A param is listed in multiple categories"
        for cat, ps in self._cat_to_params.items():
            for p in ps:
                self._pdict[p]['category']=cat

        # Irrelevancies
        self._irrelevancies: dict[str,dict[str,list[str]]] = self._disp_yaml.get('irrelevancies',{})
        for p,i in self._irrelevancies.items():
            assert p in self._pdict, f"Specified an irrelevancy using unknown controller param {p}"
            for v, p2s in list(i.items()):
                if type(p2s) is str:
                    i[v]=[p2s]
                for p2 in i[v]:
                    assert p2 in self._pdict, f"Specified an irrelevancy of unknown param {p2} with controller {p}"

    def get_dtype(self,param) -> type:
        raise NotImplementedError

    def get_place(self,param) -> ParamPlace:
        raise NotImplementedError

    def get_units(self,param) -> str:
        return self._pdict[param]['units']

    def get_description(self,param) -> str:
        return self._pdict[param].get('desc','')

    def get_display_units(self,param) -> str:
        return self._pdict[param].get('display_units',None) or self._pdict[param]['units']

    def get_display_scale(self,param) -> str:
        return self._pdict[param].get('display_scale',1)

    def get_default(self,param):
        return self._pdict[param]['default']

    def get_scale(self,param):
        return self._pdict[param].get('scale',1)

    def get_categorized(self,params):
        categorized={cat:[p for p in cparams if p in params] for cat, cparams in self._cat_to_params.items()}
        return {cat:cparams for cat,cparams in categorized.items() if len(cparams)}

    def is_hidden(self,param):
        return self._pdict[param].get('hidden',False)

    def __copy__(self):
        raise Exception("You shouldn't be copying a ParamSet")

    def get_defaults_patch(self, ignore_keys=None, only_keys=None):
        assert not (only_keys and ignore_keys), "Can only supply one of ignore_keys or only_keys"
        if only_keys:
            return ParamPatch(self,{k:self._pdict[k]['default'] for k in only_keys})
        else:
            return ParamPatch(self,{k:v['default'] for k,v in self._pdict.items() if k not in (ignore_keys or [])})

    def make_complete_patch(self, **kwargs):
        for k in kwargs: assert k in self._pdict, f"Unknown parameter {k}"
        return ParamPatch(self,{k:(kwargs[k] if k in kwargs else self.get_default(k)) for k in self._pdict})

    def mcp_(self, **kwargs): return self.make_complete_patch(**kwargs)

    def make_partial_patch(self, **kwargs):
        return ParamPatch(self,**kwargs)

    def mpp_(self, **kwargs): return self.make_partial_patch(**kwargs)


    @lru_cache
    def get_potential_irrelevancies_for(self,param):
        return [(c,int(v)) for c, vdict in self._irrelevancies.items()
                    for vs,irrs in vdict.items()
                        for v in str(vs).split('+')
                            if param in irrs]

    def translate_patch(self, patch: Union['ParamPatch',dict], other_param_set: 'ParamSet', affected_only: bool = False) -> 'ParamPatch':
        if hasattr(patch,'param_set'):
            assert patch.param_set is self, "Can only translate patches that belong to this ParamSet"
        else:
            for p in patch: assert p in self._pdict,\
                f"Supplied invalid dict to translate: contains unknown parameter {p}"
        if other_param_set is self:
            return patch.copy()
        else:
            ret=self._sub_translate_patch(patch, other_param_set, affected_only=affected_only)
            if isinstance(ret,ParamPatch):
                assert ret.param_set is other_param_set, "_sub_translate_patch returned a patch of the wrong ParamSet"
            elif type(ret) is dict:
                return ParamPatch(other_param_set,**ret)
            else:
                raise Exception(f"_sub_translate_patch returned something ({type(ret)} that's not a ParamPatch or dict"\
                                f" when trying to go from {patch.param_set} to {other_param_set}")
            return ret

    def _sub_translate_patch(self, patch: Union['ParamPatch',dict], other_param_set: 'ParamSet', affected_only: bool = False)\
            -> Union['ParamPatch',dict]:
        raise NotImplementedError

    def get_bounds(self, param):
        deets=self._pdict[param]
        lower=spicenum_to_float(deets['lower'])
        upper=spicenum_to_float(deets['upper'])
        step=np.abs(spicenum_to_float(deets['default']))*.01
        return (lower,step,upper)

    @staticmethod
    def get_total_device_width_for_patch(patch):
        raise NotImplementedError

class ParamPatch(UserDict):

    def __init__(self, param_set:ParamSet, *args, **kwargs):
        self.param_set: ParamSet=param_set
        super().__init__(*args,**kwargs)
        for k in self:
            try:
                assert k in param_set._pdict, f"Unknown parameter {k}"
            except:
                import pdb; pdb.set_trace()
                raise

    def translated_to(self, param_set: ParamSet, affected_only: bool = False):
        return self.param_set.translate_patch(self,param_set, affected_only=affected_only)

    def with_updates(self, other:Union['ParamPatch',dict]):
        n=self.copy()
        if hasattr(other,'translated_to'):
            n.update(**other.translated_to(self.param_set))
        else:
            n.update(**other)
        return n

    def filled(self):
        return self.param_set.make_complete_patch(**self)

    def update_inplace_and_return_changes(self, other: 'ParamPatch'):
        other=other.translated_to(self.param_set)
        changes={}
        for k,v in other.items():
            if (k not in self) or (self[k]!=other[k]):
                self[k]=v
                changes[k]=v
        return ParamPatch(self.param_set,**changes)

    def update_inplace_and_return_base_changes(self, other: 'ParamPatch'):
        if not hasattr(self.param_set,'base'):
            return self.update_inplace_and_return_changes(other)
        base_1=self.translated_to(self.param_set.base)
        self.update_inplace_and_return_changes(other)
        base_2=self.translated_to(self.param_set.base)
        return base_1.update_inplace_and_return_changes(base_2)

    def to_base(self, affected_only: bool = False):
        if hasattr(self.param_set,'base'):
            return self.translated_to(self.param_set.base, affected_only=affected_only)
        else:
            return self


    def get_nondefault_values(self):
        return {k:v for k,v in self.items() if self.param_set.get_default(k)!=v}

    def __repr__(self):
        ndf=self.get_nondefault_values()
        return f"<{self.param_set.model}:"\
                    +",".join(f"{k}={ndf[k]}" for k in sorted(ndf) )\
                +">"

    def is_param_irrelevant(self,param):
        cvs=[(c,v) for c,v in self.param_set.get_potential_irrelevancies_for(param) if c in self]
        return any((float(self[c])==int(v) for c,v in cvs))

    def __getattr__(self, item):
        if item in ['get_scale','get_dtype','get_description','get_bounds','get_units',
                    'model','terminals','get_place','get_display_scale','is_hidden']:
            return getattr(self.param_set,item)
        else:
            raise AttributeError(item)

    def break_into_model_and_instance(self):
        modl_params=ParamPatch(self.param_set,{k:v for k,v in self.items()
                                               if self.param_set.get_place(k)==ParamPlace.MODEL})
        inst_params=ParamPatch(self.param_set,{k:v for k,v in self.items()
                                               if self.param_set.get_place(k)==ParamPlace.INSTANCE})
        return modl_params,inst_params

    def get_total_device_width(self):
        return self.param_set.get_total_device_width_for_patch(self)

class GuessedSubcktParamSet(ParamSet):

    def __init__(self, model, file, section="", display_yaml=None):
        self.file=file
        self.section=section
        scs_includes=[(str(file),"section="+section)]
        in_section=""
        defaults=None
        with open(file) as f:
            for l in f:
                if l.startswith("section"):
                    in_section=l.strip().split(" ")[1]

                if l.startswith(f"subckt {model}"):
                    terminals=[x.strip() for x in l.strip().split()[2:]]
                    # TODO: this check is temporary, remove it
                    assert terminals==['d','g','s','b']

                    if in_section==self.section:
                        l=next(f).strip()
                        assert l.startswith("parameters ")
                        assert defaults is None
                        defaults=dict([eq.split("=") for eq in l.split()[1:]])
        defaults['m']=1
        pdict= {k:{'default':v,'units':'A.U.'} for k,v in defaults.items()}
        super().__init__(model=model, terminals=terminals, pdict=pdict,
                         display_yaml=display_yaml, scs_includes=scs_includes)

    def get_dtype(self, param):
        return self._pdict[param].get('dtype',float)

    def get_place(self, param):
        return ParamPlace.INSTANCE



#class GuessedDSPFParamSet(ParamSet):
#    _shared_paramdicts={}
#    _terminals={}
#
#    def __init__(self, subckt, supply_model, file, **kwargs):
#        super().__init__(model=subckt, **kwargs)
#        self.file=file
#        self.scs_includes=supply_model.scs_includes+[str(file)]
#        self.va_includes=supply_model.va_includes
#        self.supply_model=supply_model
#        if (self.model,file) in GuessedDSPFParamSet._shared_paramdicts:
#            return
#        defaults=None
#        with open(file) as f:
#            for l in f:
#                if l.startswith(f".SUBCKT") and l.split()[1]==subckt:
#                    GuessedDSPFParamSet._terminals[(self.model,file)]=l.split()[2:]
#                    assert defaults is None
#                    defaults={}
#        assert defaults is not None
#        GuessedDSPFParamSet._shared_paramdicts[(self.model,file)]=defaults
#
#    @property
#    def _shared_paramdict(self):
#        return self._shared_paramdicts[(self.model,self.file)]
#
#    def get_total_device_width(self):
#        return self.supply_model.get_total_device_width()
#
#    @property
#    def terminals(self):
#        return self._terminals[(self.model,self.file)]


class CMCParamSet(ParamSet):
    """ Abstract superclass for Verilog-A standard models, so far tried only on CMC models (MVSG and ASMHEMT).
    """
    def __init__(self,vaname,display_yaml=None):

        vapath=get_va_path(vaname=vaname)
        if display_yaml is None:
            if (potential_yaml:=Path(str(vapath).replace("vacode","display").replace(".va",".yaml"))).exists():
                display_yaml=potential_yaml

        with open(vapath,'r') as f:
            # Collect the model parameter and instance parameter definition lines
            # Tweak them slightly so they can be read by csv reader with the macro
            # name as the first entry in each line (easier than trying to split
            # by commas since descriptions may contain commas)
            lines=[]
            for l in (l.strip() for l in f):
                l=l.replace('`P_CELSIUS0','273.15')

                if l.startswith('module'):
                    model=l.split()[1].split("(")[0]
                    terminals=[x.strip() for x in l.split('(')[1].split(')')[0].split(',')]

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
                           {'macro':macro,'default':default,'units':units,'desc':desc[:-1].strip()}
                   case 7:
                       macro,name,default,units,lower,upper,desc=\
                           [x.strip() for x in line]
                       paramset[name]=\
                           {'macro':macro,'default':default,'units':units,
                            'desc':desc[:-1].strip(),'lower':lower,'upper':upper}
                   case _:
                       raise Exception("Can't read line, best guess is "+",".join(l))



        super().__init__(model=model,terminals=terminals,pdict=paramset,
                         display_yaml=display_yaml,va_includes=[vaname])

    def get_dtype(self,param) -> type:
        return {'R':float,'I':int}[self._pdict[param]['macro'][2]]
    def get_place(self,param) -> ParamPlace:
        return [ParamPlace.INSTANCE,ParamPlace.MODEL]\
                    [self._pdict[param]['macro'].startswith('M')]
    
    def get_bounds(self, param, null=None):
        if null is np.inf:
            upper_null=null
            lower_null=-null
        else:
            upper_null=lower_null=null
        deets=self._pdict[param]
        if 'macro' not in deets:
            #return super().get_bounds(param)
            # ^^ This causes issues because currently CMCParamSet.get_bounds is
            # sometimes called where self=SimplifierParamSet, which is not an instance of CMCParamSet
            # so let's spell it out
            return ParamSet.get_bounds(self,param)
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

class SimplifierParamSet(ParamSet):

    def __init__(self, base_param_set:ParamSet, trans_code,
                 additional_parameters={},
                 overrides={},
                 constants={},
                 display_yaml=None):
        self._constants=constants
        self.base=base_param_set
        pdict=deepcopy(self.base._pdict)
        self._translations=[]
        unconnected_base_params=set(self.base._pdict)
        used_from_this_pset={}
        adds=[]
        drops=[]
        self._homonyms=[]
        for l in trans_code.split("\n"):
            l=l.strip()
            if l=='' or l.startswith("#"): continue
            for_this_pset,for_base_psets=[x.strip() for x in l.split("->")]
            for_base_psets=[x.strip() for x in for_base_psets.split(',') if len(x)]
            if len(for_base_psets)==0:
                for i in [x.strip() for x in for_this_pset.split(',')]:
                    print(f"Grabbing do-nothing {i}")
                    if (i not in pdict) and (i not in additional_parameters):
                        pdict[i]={'hidden':True}
                        adds.append(i)
                        used_from_this_pset[i]=[]
                continue

            if for_this_pset=="":
                just_default=True
            else:
                just_default=False
                involved=[x.strip() for x in
                          for_this_pset.replace("("," ").replace(")"," ").replace("*"," ") \
                              .replace("/"," ").replace("+"," ").replace("-"," ").split()]
                def is_numerical(x):
                    try: float(x)
                    except: return False
                    else: return True
                involved=[i for i in involved if i not in self._constants and not is_numerical(i)]
                if len(involved)==0:
                    for_this_pset=eval(for_this_pset,self._constants)
                elif len(involved)==1:
                    i=involved[0]
                    if (i not in pdict) and (i not in additional_parameters):
                        #logger.debug(f"New parameter {i} based on {for_base_psets[0]}")
                        pdict[i]=deepcopy(self.base._pdict[for_base_psets[0]])
                        #logger.debug(str(pdict[i]))
                        adds.append(i)
                        used_from_this_pset[i]=used_from_this_pset.get(i,[])+for_base_psets
                else:
                    for i in involved:
                        assert (i in pdict) or (i in additional_parameters), f"Provide more info about {i}"

                if len(involved) and (for_this_pset!=involved[0]):
                    for_this_pset=(involved,
                                   eval("lambda "+",".join(involved)+": "+for_this_pset,self._constants),
                                   for_this_pset)
                    for i in involved:
                        used_from_this_pset[i]=used_from_this_pset.get(i,[])+for_base_psets

            for for_base_pset in for_base_psets:
                if for_base_pset in used_from_this_pset:
                    self._homonyms.append(for_base_pset)
                else:
                    del pdict[for_base_pset]
                    drops.append(for_base_pset)
                if just_default:
                    self._translations.append([self.base.get_default(for_base_pset),for_base_pset])
                else:
                    self._translations.append([for_this_pset,for_base_pset])
                unconnected_base_params.remove(for_base_pset)

        for p in unconnected_base_params:
            self._translations.append([p,p])

        pdict.update(additional_parameters)
        adds+=list(additional_parameters)
        for i in used_from_this_pset:
            assert i in pdict, f"Not finding {i} in this paramset, was it dropped earlier in translation code?"

        if display_yaml is None:
            display_yaml=deepcopy(self.base._disp_yaml)
            for p in adds:
                for c,v2i in display_yaml['irrelevancies'].items():
                    for v,ips in v2i.items():
                        if all(b in ips for b in used_from_this_pset[p]):
                            ips.append(p)

            for p in drops:
                for cat, ps in display_yaml['categories'].items():
                    if p in ps: ps.remove(p)
                if p in display_yaml['display_units']:
                    del display_yaml['display_units'][p]
                if p in display_yaml['unit_overrides']:
                    del display_yaml['unit_overrides'][p]
                for c,v2i in display_yaml['irrelevancies'].items():
                    if p==c: raise NotImplementedError("Dropping a control variable")
                    for v,ips in v2i.items():
                        if p in ips: ips.remove(p)


        for p,pd in overrides.items():
            assert p in pdict, f"Override provided for unknown parameter {p}"
            pdict[p].update(**pd)
        super().__init__(model=self.base.model, terminals=self.base.terminals, pdict=pdict,
                         scs_includes=self.base.scs_includes, va_includes=self.base.va_includes,
                         display_yaml=display_yaml)
        for p,pd in overrides.items():
            pdict[p].update(**pd)

    def _sub_translate_patch(self, patch: Union['ParamPatch',dict], other_param_set: 'ParamSet', affected_only=False):
        assert other_param_set is self.base, f"{self} can only translate to its underlying model {self.base}"
        d={}
        for for_this_pset, for_base_pset in self._translations:
            if type(for_this_pset) is str and for_this_pset in self._pdict:
                #print(for_this_pset.rjust(35),'    ->    ',for_base_pset)
                if for_this_pset in patch:
                    d[for_base_pset]=patch[for_this_pset]
            elif type(for_this_pset) is str and for_this_pset[0] in '+-0123456789.':
                if not affected_only:
                    d[for_base_pset]=for_this_pset
            elif type(for_this_pset) in [float,int]:
                #print(f'{for_this_pset:35g}','    ->    ',for_base_pset)
                if not affected_only:
                    d[for_base_pset]=for_this_pset
            elif type(for_this_pset) is tuple:
                #print((f"{','.join(for_this_pset[0])}: {for_this_pset[2]}").rjust(35),'    ->    ',for_base_pset)
                if any(k in patch for k in for_this_pset[0]):
                    assert all(k in patch for k in for_this_pset[0]), f"Must have all or none of {for_this_pset[0]}"
                    d[for_base_pset]=for_this_pset[1](**{k:spicenum_to_float(patch[k]) for k in for_this_pset[0]})
            else:
                raise Exception(f"What is {for_this_pset}")
        return ParamPatch(other_param_set,**d)

    def minimal_completion_of_pcell(self):
        assert hasattr(self,'pcell_params'), "Must have pcell_params defined"
        prev_reqs=set()
        reqs=set(self.pcell_params)
        while prev_reqs!=reqs:
            prev_reqs=reqs
            reqs_sing=set([for_this_pset for for_this_pset, for_base_pset in self._translations
                      if type(for_this_pset) is str and for_this_pset in prev_reqs ])
            reqs_mult=set([r for for_this_pset, for_base_pset in self._translations
                                if (type(for_this_pset) is tuple and any( k in for_this_pset[0] for k in prev_reqs))
                           for r in for_this_pset[0]
                           ])
            reqs=reqs_sing.union(reqs_mult).union(prev_reqs)
        return reqs

    def check_param_placements(self):
        simple_inst_reqs=self.minimal_completion_of_pcell()
        base_must_be_inst=list(self.get_defaults_patch(only_keys=simple_inst_reqs).to_base(affected_only=True))
        trouble=[p for p in base_must_be_inst if self.base.get_place(p)!=ParamPlace.INSTANCE]
        assert not len(trouble), f"In order to have a subcircuit control this model, {trouble} should be made instance parameters!"

    # TODO: regularize pdict to move more of this into ParamSet superclass and not use this hack
    def get_bounds(self, param, *args, **kwargs):
        return self.base.__class__.get_bounds(self, param, *args, **kwargs)
    def get_dtype(self, param):
        return self.base.__class__.get_dtype(self, param)

    @classmethod
    def from_yaml(cls,yaml_path,base):
        from scipy.constants import epsilon_0, elementary_charge as q
        with open(yaml_path,'r') as f:
            yl=yaml.safe_load(f)
        psSimple=cls(
            base_param_set=base,
            trans_code=yl['trans_code'],
            overrides=yl['overrides'],
            constants={'EPS_SIO2':3.9*epsilon_0, 'Q':q},
            additional_parameters=yl.get('additional_parameters',{}),
        )
        for attr,val in yl.get('attributes',{}).items():
            print(f"Setting Attribute {attr}={val}")
            setattr(psSimple,attr,val)
        return psSimple

    @staticmethod
    def get_total_device_width_for_patch(patch):
        return patch.to_base().get_total_device_width()

class MVSGParamSet(CMCParamSet):

    def __init__(self,vaname):
        super().__init__(vaname=vaname)
        assert self.terminals==['d','g','s','b'], f"MVSG Terminals are {self.terminals}"

    @staticmethod
    def get_total_device_width_for_patch(patch):
        return spicenum_to_float(patch['w'])*\
                spicenum_to_float(patch['ngf'])


class ASMHEMTParamSet(CMCParamSet):

    def __init__(self, vaname):
        super().__init__(vaname=vaname)
        assert self.terminals==['d','g','s','b','dt']

    @staticmethod
    def get_total_device_width_for_patch(patch):
        return spicenum_to_float(patch['w'])*\
                spicenum_to_float(patch['nf'])

