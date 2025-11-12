from typing import TypeVar, Callable
from functools import wraps
import pickle
import sys
from contextlib import redirect_stderr, contextmanager
from functools import cache
from io import StringIO
import re
import logging
from copy import copy
import pandas as pd
import numpy as np


# Copied from https://stackoverflow.com/a/22424821
# under license https://creativecommons.org/licenses/by-sa/4.0/
@cache
def is_notebook() -> bool:
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


@contextmanager
def catch_stderr(ignore_regexes,unmatched='stderr'):
    sio=StringIO()
    with redirect_stderr(sio):
        yield
    sio.seek(0)
    for l in sio.readlines():
        if l.strip()=="": continue
        if any((re.match(ir,l.strip())  for ir in ignore_regexes)):
            continue
        else:
            if unmatched=='stderr':
                print(l,file=sys.stderr)
            elif unmatched=='raise':
                raise Exception(l)
            else:
                raise Exception(f"What is '{unmatched}'?")

class MyHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(sys.stdout)
        self.unset_current_callback()

        # See https://stackoverflow.com/a/7517430
        self.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s', '%H:%M:%S'))

    def set_current_callback(self,callback):
        self._callback=callback

    def unset_current_callback(self):
        self._callback=(lambda record, formatter: True)

    def emit(self, record):
        if self._callback(record,self.formatter):
            super().emit(record)

logger=logging.getLogger('compyct')
logger.setLevel('DEBUG')
log_handler=MyHandler()
logger.handlers.clear()
logger.addHandler(log_handler)

def set_logging_callback(callback):
    log_handler.set_current_callback(callback)
def unset_logging_callback():
    log_handler.unset_current_callback()

def s2y(s11,s12,s21,s22, z0=50):
    y0=1/z0
    deltas=(1+s11)*(1+s22)-s12*s21
    y11=((1-s11)*(1+s22)+s12*s21)/deltas * y0
    y12=-2*s12/deltas * y0
    y21=-2*s21/deltas * y0
    y22=((1+s11)*(1-s22)+s12*s21)/deltas * y0
    return y11,y12,y21,y22

def y2s(y11,y12,y21,y22, z0=50):
    delta=(1+z0*y11)*(1+z0*y22)-z0**2*y12*y21
    s11=((1-z0*y11)*(1+z0*y22)+z0**2*y12*y21)/delta
    s12=-2*z0*y12/delta
    s21=-2*z0*y21/delta
    s22=((1+z0*y11)*(1-z0*y22)+z0**2*y12*y21)/delta
    return s11,s12,s21,s22

def s2z(s11,s12,s21,s22, z0=50):
    delta=(1-s11)*(1-s22)-s12*s21
    z11=((1+s11)*(1-s22)+s12*s21)/delta *z0
    z12=2*s12/delta * z0
    z21=2*s21/delta * z0
    z22=((1-s11)*(1+s22)+s12*s21)/delta *z0
    return z11, z12, z21, z22

def z2s(z11,z12,z21,z22, z0=50):
    delta=(z11+z0)*(z22+z0)-z12*z21
    s11=((z11-z0)*(z22+z0)-z12*z21)/delta
    s12=2*z0*z12/delta
    s21=2*z0*z21/delta
    s22=((z11+z0)*(z22-z0)-z12*z21)/delta
    return s11, s12, s21, s22

# Pickle-able string eval lambda function
class ExprFunc:
    def __init__(self, expr: str, involved: list[str], consts: dict[str,float]):
        self._expr=expr
        self._involved=involved
        self._consts=consts
        self._lambda=eval("lambda "+",".join(involved)+": "+expr,copy(self._consts))
    def __call__(self, *args, **kwargs):
        return self._lambda(*args,**kwargs)
    def __str__(self):
        return f"<ExprFunc: {self._expr}>"
    def __getstate__(self):
        return self._expr, self._involved, self._consts
    def __setstate__(self,args):
        self.__init__(*args)

T = TypeVar('T')
def pickle_cache(func: Callable[...,T], cache_path=None) -> Callable[...,T]:
    if cache_path is None:
        return func
    else:
        @wraps(func)
        def wrapper(*args, force_rerun=False, **kwargs):
            if not force_rerun:
                try:
                    with open(cache_path,'rb') as f:
                        val=pickle.load(f)
                        print("Using picked cache")
                        return val
                except Exception as e:
                    print("Couldn't use pickled cache")
                    print(str(e))
            val=func(*args,**kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(val,f)
            print("Dumped values into pickle cache")
            return val
        return wrapper

def only(lst):
    lst=list(lst)
    assert len(lst)==1
    return lst[0]

def form_multisweep(point_results,outeri,inneri,inner_name,queryvar,querytarget=np.nan):
    if point_results is None: return point_results
    outers=list(sorted(set([k[outeri] for k in point_results])))
    inners=list(sorted(set([k[inneri] for k in point_results])))
    sweep_results={}
    assert (queryvar is None)==(querytarget is None), "Must give queryvar and querytarget together"
    #assert (queryvar is None)!=(collapser is None), "Give EITHER queryvar OR collapser"
    for outer in outers:
        rows=[]
        rel_inners=[]
        for inner in inners:
            key=(outer,inner) if outeri<inneri else (inner,outer)
            if key in point_results: rel_inners.append(inner)
            else: continue
            pt_df=point_results[key]
            #if queryvar is not None:
            if np.isnan(querytarget):
                pt_df=pt_df[pd.isna(pt_df[queryvar])].copy()
            else:
                pt_df=pt_df[np.isclose(pt_df[queryvar],querytarget)].copy()
            #elif collapser is not None:
            #    pt_df=collapser(pt_df.copy())
            assert len(pt_df)==1, "Query or collapser must result in a len()==1 table"
            rows.append(pt_df)
        sweep_results[(outer,'f')]=pd.concat(rows).assign(**{inner_name:rel_inners})
    return sweep_results

from typing import TypeVar
T=TypeVar('T')
def unnest_dict(d:dict[str,dict[str,T]])->dict[str,T]:
    return {f"{k}|||{kk}":vv for k,v in d.items() for kk,vv in v.items()}