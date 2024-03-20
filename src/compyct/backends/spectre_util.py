import numpy as np
from io import StringIO

from compyct.paramsets import ParamPlace


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

def simplifier_patch_to_scs(self, pdk_model_name):
    ps=self.param_set
    f=StringIO()

    assert self.param_set.terminals==['d','g','s','b','dt']
    print(f"inline subckt {pdk_model_name} d g s b",file=f)
    pdk_params=ps.get_defaults_patch(only_keys=ps.pcell_params)
    print(f"\tparameters "+" ".join([f"{k}={v}" for k,v in pdk_params.items()]),file=f)
    def int_name(basename):
        return basename+"_modelcard" if basename in ps._homonyms else basename
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
    inst_passings=" ".join([f"{k}={int_name(k)}" for k in
                            ps.get_defaults_patch(only_keys=ps.minimal_completion_of_pcell()).to_base(affected_only=True)])

    assert self.param_set.terminals==['d','g','s','b','dt']
    print(f"\t{pdk_model_name} d g s b dt {pdk_model_name}core {inst_passings}",file=f)
    print(f"ends {pdk_model_name}",file=f)

    assert len(self.param_set.va_includes)==1
    vafile=self.param_set.va_includes[0]
    #print(f"ahdl_include \"./{vafile}\"",file=f)
    print(f"model {pdk_model_name}core {self.param_set.model} "+' '.join([f"{k}={v}" for k,v in self.filled().to_base().items()]),file=f)

    f.seek(0)
    return f.read(), [f"ahdl_include \"./{vafile}\""]