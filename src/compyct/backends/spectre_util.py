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

# simplifier_patch_to_scs moved out to notebook temporarily