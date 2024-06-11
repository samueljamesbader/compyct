import numpy as np
from io import StringIO

from compyct.paramsets import ParamPlace, ParamPatch
import textwrap

def wrap_scs(paragraph,indent):
    return '\n'.join(textwrap.wrap(paragraph,width=80,initial_indent=indent,
                          subsequent_indent=indent+'\t+ ',break_long_words=False))

def n2scs(num):
    if type(num) is str:
        return num.replace("meg","M")
    else:
        if num==0: return '0'
        ord=np.clip(np.floor(np.log10(np.abs(num))/3)*3,-18,15)
        si={-18:'a',-15:'f',-12:'p',-9:'n',-6:'u',-3:'m',
            0:'',
            3:'k',6:'M',9:'G',12:'T',15:'P'}[ord]
        return f'{(num/10**ord):g}{si}'


def simplifier_patch_to_scs(self:ParamPatch, pdk_model_name, netmap, use_spectre_builtin=False):
    #print(f"In p_to_scs: {pdk_model_name} with {use_spectre_builtin}")
    ps=self.param_set
    f=StringIO()

    term_order=['d','g','s','b']
    print(f"inline subckt {pdk_model_name} "+' '.join(term_order),file=f)
    pdk_params=ps.get_defaults_patch(only_keys=ps.pcell_params)
    print(wrap_scs(f"parameters "+" ".join([f"{k}={v}" for k,v in pdk_params.items()]),indent='\t'),file=f)
    def int_name(basename):
        return basename+"_modelcard" if basename.lower() in ps._lowercase_homonyms else basename
    inst_reqs=[k for k in ps.minimal_completion_of_pcell() if k not in ps.pcell_params]
    for i in inst_reqs:
        print(f"\tparameters {int_name(i)}={self[i]}",file=f)
    for for_this_pset, for_base_pset in ps._translations:
        if ps.base.get_place(for_base_pset)==ParamPlace.INSTANCE:
            if type(for_this_pset) in [str,float,int] and for_this_pset in ps._pdict:
                if for_this_pset!=for_base_pset:
                    print(f"\tparameters {int_name(for_base_pset)}={for_this_pset}",file=f)
            elif type(for_this_pset) is tuple:
                print(f"\tparameters {int_name(for_base_pset)}={for_this_pset[2]}",file=f)
    base_params_manip=ps.get_defaults_patch(only_keys=ps.minimal_completion_of_pcell()).to_base(affected_only=True)
    base_params_all=ps.get_defaults_patch(only_keys=ps.minimal_completion_of_pcell()).to_base(affected_only=False)
    inst_passings1=[f"{k.lower() if use_spectre_builtin else k}={int_name(k)}" for k in base_params_manip]
    inst_passings2=[f'{k.lower() if use_spectre_builtin else k}={v}' for k,v in base_params_all.items()
                    if ps.base.get_place(k)==ParamPlace.INSTANCE and k not in base_params_manip]
    inst_passings=" ".join(inst_passings1+inst_passings2)

    #assert self.param_set.terminals==['d','g','s','b','dt']
    term_part=' '.join([netmap.get(t,t) for t in ps.terminals])
    print(wrap_scs(f"{pdk_model_name} {term_part} {pdk_model_name}_inner {inst_passings}",indent='\t'),file=f)
    if ps.extra_subckt_text is not None:
        print('\n'.join(["\t"+l for l in ps.extra_subckt_text['spectre'].split('\n') if l!='']),file=f)
    print(f"ends {pdk_model_name}",file=f)

    assert len(self.param_set.va_includes)==1
    vafile=self.param_set.va_includes[0]
    #print(f"ahdl_include \"./{vafile}\"",file=f)
    modelstr=f"model {pdk_model_name}_inner {use_spectre_builtin or self.param_set.model} "+\
             ' '.join([f"{k.lower() if use_spectre_builtin else k}={v}" for k,v in self.filled().to_base().items()
                       if (k!='m' and ps.base.get_place(k)==ParamPlace.MODEL and ps.base.get_default(k)!=v)])
    print(wrap_scs(modelstr,indent=''),file=f)

    f.seek(0)
    return f.read(), [f"ahdl_include \"./{vafile}\""]