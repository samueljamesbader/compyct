from __future__ import annotations
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

from compyct.backends.spectre_util import wrap_scs
from myrtusfit.first_fit_study import MODELFITPATH
#from compyct.backends.spectre_util import simplifier_patch_to_scs
from compyct.backends.backend import ModelCardWriter, get_va_path
from compyct.paramsets import ParamPatch, ParamPlace, SimplifierParamSet

if TYPE_CHECKING:
    from compyct.model_suite import ModelSuite

def chunk_keys(d:dict, by_first_n_char: int, max_in_chunk: int)->list[dict]:
    """Chunk a dict into a list of dicts, each with at most max_in_chunk items,
    grouped by the first by_first_n_char characters of the keys.
    """
    chunks=[]
    current_chunk={}
    current_prefix=None
    for k in sorted(d.keys()):
        prefix=k[:by_first_n_char]
        if (current_prefix is None) or (prefix!=current_prefix) or (len(current_chunk)>=max_in_chunk):
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk={k:d[k]}
            current_prefix=prefix
        else:
            current_chunk[k]=d[k]
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

class SpectreModelCardWriter(ModelCardWriter):
    backend = 'spectre'

    def simplifier_patch_to_modelcard_string(self,
                patch:ParamPatch[SimplifierParamSet], element_name:str, out_to_in_netmap:OrderedDict[str,str|None], pcell_params:list[str],
                extra_text:str, use_builtin:bool=False, inner_name=None):
        ps=patch.param_set
        f=StringIO()

        inner_name=inner_name or f"{element_name}_inner"
        innards={(k.lower() if use_builtin else k):v
                     for k,v in sorted(patch.filled().to_base().items(),key=(lambda x: x[0]))
                           if (k != 'm' and ps.base.get_place(k) == ParamPlace.MODEL)}
        by_first_n_char=2 if len(innards)>300 else 1
        modelstr=f"\nmodel {inner_name} {use_builtin or patch.param_set.model}\n"+\
                 '\n'.join([f"  + "+' '.join([f"{k}={v}" for k,v in chunk.items()])
                            for chunk in chunk_keys(innards, by_first_n_char=by_first_n_char, max_in_chunk=5)])
                           #if (k!='m' and ps.base.get_place(k)==ParamPlace.MODEL and ps.base.get_default(k)!=v)])
        print(modelstr+'\n',file=f)



        term_order=out_to_in_netmap.keys()
        inline=False
        if inline:
            print(f"\ninline subckt {element_name} "+' '.join(term_order),file=f)
        else:
            print(f"\nsubckt {element_name} "+' '.join(term_order),file=f)
        pdk_params=ps.get_defaults_patch(only_keys=pcell_params)
        print(wrap_scs(f"parameters "+" ".join([f"{k}={v}" for k,v in pdk_params.items()]),indent='\t'),file=f)
        def int_name(basename):
            return basename+"_modelcard" if basename.lower() in ps._lowercase_homonyms else basename
        inst_reqs=[k for k in ps.minimal_completion_of_pcell() if k not in pcell_params]
        for i in inst_reqs:
            print(f"\tparameters {int_name(i)}={patch[i]}",file=f)
        for for_this_pset, for_base_pset in ps._translations:
            if ps.base.get_place(for_base_pset)==ParamPlace.INSTANCE:
                if type(for_this_pset) in [str,float,int] and for_this_pset in ps._pdict:
                    if for_this_pset!=for_base_pset:
                        print(f"\tparameters {int_name(for_base_pset)}={for_this_pset}",file=f)
                elif type(for_this_pset) is tuple:
                    print(f"\tparameters {int_name(for_base_pset)}={for_this_pset[2]}",file=f)
        base_params_manip=ps.get_defaults_patch(only_keys=ps.minimal_completion_of_pcell()).to_base(affected_only=True)
        base_params_all=ps.get_defaults_patch(only_keys=ps.minimal_completion_of_pcell()).to_base(affected_only=False)
        inst_passings1=[f"{k.lower() if use_builtin else k}={int_name(k)}" for k in base_params_manip]
        inst_passings2=[f'{k.lower() if use_builtin else k}={v}' for k,v in base_params_all.items()
                        if ps.base.get_place(k)==ParamPlace.INSTANCE and k not in base_params_manip]
        inst_passings=" ".join(inst_passings1+inst_passings2)

        in_to_out_netmap={v:k for k,v in out_to_in_netmap.items() if v is not None}
        term_part=' '.join([in_to_out_netmap.get(t,t+'_inner') for t in ps.terminals])
        print(wrap_scs(f"{element_name if inline else 'X0'} {term_part} {inner_name} {inst_passings}",indent='\t'),file=f)
        if extra_text is not None and extra_text!='':
            print('\n'.join(["\t"+l for l in extra_text.split('\n') if l!='']),file=f)
        print(f"ends {element_name}",file=f)


        f.seek(0)
        return f.read()

    def get_wrapper_modelcard_string(self, element_name:str, inner_element_name:str,
                                     pass_parameters:dict, eat_parameters:dict, inner_term_order:list[str],
                                     out_to_in_netmap:OrderedDict[str,str|None]={}, extra_text:str='')->str:
        f=StringIO()
        term_order=out_to_in_netmap.keys()
        print(f"\nsubckt {element_name} "+' '.join(term_order),file=f)
        print(wrap_scs(f"parameters "+" ".join([f"{k}={v}" for k,v in (pass_parameters|eat_parameters).items()]),
                       indent='\t'),file=f)
        in_to_out_netmap={v:k for k,v in out_to_in_netmap.items() if v is not None}
        core_terms=' '.join([in_to_out_netmap.get(t,t+'_inner') for t in inner_term_order])
        inst_passings=' '.join([f"{k}={k}" for k in pass_parameters])
        print(wrap_scs(f"Xcore {core_terms} {inner_element_name} {inst_passings}",indent='\t'),file=f)
        if extra_text is not None:
            print('\n'.join(["\t"+l for l in extra_text.split('\n') if l!='']),file=f)
        print(f"ends {element_name}",file=f)
        f.seek(0)
        return f.read()


    def model_suite_to_modelcard_string(self, model_suite:ModelSuite,):
        from compyct.model_suite import FittableModelSuite
        if isinstance(model_suite, FittableModelSuite):
            if len(model_suite.submodel_split)==1:
                return model_suite.get_submodel_modelcard_text(
                    submodel_split_name=next(iter(model_suite.submodel_split)),mcw=self)

            #ps=list(patch_group.values())[0].param_set
            #assert all( (patch.param_set is ps) for patch in patch_group.values()), \
            #    "All patches in a patch_group must have the same paramset"
            f=StringIO()

            term_order=list(model_suite.get_out_to_in_netmap().keys())
            print((f"\ninline subckt {model_suite.element_name} "+' '.join(term_order)),file=f)
            pdk_params=model_suite.param_set.get_defaults_patch(only_keys=
                [k for k in model_suite.param_set.pcell_params if k not in model_suite.kill_parameters()])
            print(wrap_scs(f"parameters "+" ".join([f"{k}={v}" for k,v in pdk_params.items()]),indent='\t'),file=f)

            for patchnum,(submodel_name,patch_cond) in enumerate(model_suite.submodel_split.items()):
                if patch_cond is not None:
                    print(f"\t{'if' if patchnum==0 else 'else if'} ({patch_cond}) {{",file=f)
                else:
                    print(f" else {{",file=f)
                term_part=' '.join(term_order)
                cond_inst_str=f"{model_suite.element_name} {term_part} {model_suite.element_name}_cond{submodel_name} "+\
                    " ".join([f'{k}={k}' for k in pdk_params])
                print(wrap_scs(cond_inst_str,indent='\t\t'),file=f)
                print("\t}",file=f,end='')
            print(f"\nends {model_suite.element_name}",file=f)
            
            for patchnum,(submodel_name,patch_cond) in enumerate(model_suite.submodel_split.items()):
                print(model_suite.get_submodel_modelcard_text(
                    submodel_split_name=submodel_name,mcw=self),file=f)

            f.seek(0)
            return f.read()
        else:
            return model_suite.get_modelcard_text(mcw=self)
    
    def write_modelcard_file(self, filepath:Path, header:str, model_suites:list[ModelSuite]):
        with open(filepath,'w') as f:
            print(header.replace("$FILENAME$",filepath.name),file=f)
            print("section tttt",file=f)
            
            vafiles=[]
            for ms in model_suites: vafiles+=(ms.va_includes)
            for vafile in sorted(set(vafiles)):
                print(f"ahdl_include \"./{vafile}\"",file=f)
            for ms in model_suites:
                print(self.model_suite_to_modelcard_string(ms),file=f)

            print("endsection tttt",file=f)

def make_bundle(patchgroups,version,netmaps={},builtins={},share_inner={},extra_scs=None, extra_code=None, process_suffix='',header=''):
    from compyct import OUTPUT_DIR
    BUNDLE_DIR=OUTPUT_DIR/'output/bundles'
    BUNDLE_DIR.mkdir(parents=True,exist_ok=True)
    version=version+process_suffix

    # Make the bundle folder
    if (MODELFITPATH/f"output/bundle_{version}").exists():
        for f in (MODELFITPATH/f"output/bundle_{version}").glob("*"): f.unlink()
        (MODELFITPATH/f"output/bundle_{version}").rmdir()
    (MODELFITPATH/f"output/bundle_{version}").mkdir(parents=True)
    
    # Get all the va files
    vafiles=[]
    for modelname, patchgroup in patchgroups.items():
        if modelname not in builtins:
            for patch in patchgroup.values():
                vafiles+=(patch.param_set.va_includes)
    for vafile in set(vafiles):    
        (MODELFITPATH/f"output/bundle_{version}/{vafile}").write_text(get_va_path(vafile).read_text())

    # Check consistency of share inner
    for k,ms in share_inner.items():
        for m in ms:
            assert dict(**patchgroups[ms[0]][None])==dict(**patchgroups[m][None])
    
    with open(MODELFITPATH/f"output/bundle_{version}/p1231_2.scs",'w') as f:
        print(header.replace("$VERSION$",version),file=f)
        if extra_scs: print(f'include "{extra_scs[0]}" section={extra_scs[1]}',file=f)
        for vafile in set(vafiles):
            print(f"ahdl_include \"./{vafile}\"",file=f)
        for modelname,patchgroup in patchgroups.items():
            inner_name=next((k for k,v in share_inner.items() if modelname in v),None)
            scs_code=simplifier_patch_group_to_spectre_string(patchgroup, modelname,
                                                        netmap=netmaps.get(modelname,{}),
                                                        use_spectre_builtin=builtins.get(modelname,False),
                                                        inner_name=inner_name,
                                                        print_inner=(True if (not inner_name) else share_inner[inner_name][-1]==modelname))
            print(scs_code,file=f)
            # if len(patchgroup)>1:
            #     pass
            # for patchnum,(patchcond, patch) in enumerate(patchgroup.items()):
            #     scs_code,incs=simplifier_patch_group_to_scs(patch, f'{modelname}_cond{patchnum}')
            #     #print("".join(incs),file=f)
            #     print(scs_code,file=f)
            
        if extra_code: print(extra_code+"\n",file=f)
        print("endsection tttt",file=f)
        