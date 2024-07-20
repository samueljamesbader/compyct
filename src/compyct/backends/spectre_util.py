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
        #ord=np.clip(np.floor(np.log10(np.abs(num))/3)*3,-18,15)
        ord = np.floor(np.log10(np.abs(num)) / 3) * 3
        if ord>=-18 and ord<=15:
            si={-18:'a',-15:'f',-12:'p',-9:'n',-6:'u',-3:'m',
                0:'',
                3:'k',6:'M',9:'G',12:'T',15:'P'}[ord]
            return f'{(num/10**ord):g}{si}'
        else:
            return f'{num:g}'


def simplifier_patch_to_scs(self:ParamPatch, pdk_model_name, netmap, use_spectre_builtin=False, inner_name=None, print_inner=True):
    #print(f"In p_to_scs: {pdk_model_name} with {use_spectre_builtin}")
    ps=self.param_set
    f=StringIO()

    term_order=['d','g','s','b']
    inline=False
    if inline:
        print(f"inline subckt {pdk_model_name} "+' '.join(term_order),file=f)
    else:
        print(f"subckt {pdk_model_name} "+' '.join(term_order),file=f)
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
    inner_name=inner_name or f"{pdk_model_name}_inner"
    print(wrap_scs(f"{pdk_model_name if inline else 'X0'} {term_part} {inner_name} {inst_passings}",indent='\t'),file=f)
    if ps.extra_subckt_text is not None:
        print('\n'.join(["\t"+l for l in ps.extra_subckt_text['spectre'].split('\n') if l!='']),file=f)
    print(f"ends {pdk_model_name}",file=f)

    assert len(self.param_set.va_includes)==1
    vafile=self.param_set.va_includes[0]
    #print(f"ahdl_include \"./{vafile}\"",file=f)
    modelstr=f"model {inner_name} {use_spectre_builtin or self.param_set.model} "+\
             ' '.join([f"{k.lower() if use_spectre_builtin else k}={v}" for k,v in self.filled().to_base().items()
                       #if (k!='m' and ps.base.get_place(k)==ParamPlace.MODEL and ps.base.get_default(k)!=v)])
                       if (k != 'm' and ps.base.get_place(k) == ParamPlace.MODEL)])
    if print_inner:
        print(wrap_scs(modelstr,indent=''),file=f)

    f.seek(0)
    return f.read(), [f"ahdl_include \"./{vafile}\""]

#################

from pathlib import Path
import textwrap
import os
import shutil
import subprocess

typical_scs_code = """
simulatorOptions options psfversion="1.4.0" reltol=1e-3 vabstol=1e-6 \\
    iabstol=1e-12 temp=27 tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 \\
    maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 \\
    sensfile="../psf/sens.output" checklimitdest=psf 
saveOptions options save=allpub
"""


def export_netlist(library, cell, view='schematic', design_variables=[], scs_includes="", additional_code="",
                   include_typical=True):
    WARD = Path(os.environ['WARD'])
    rundir = WARD / 'spyctre' / f"{library}__{cell}"
    # Check if the directory exists
    if rundir.exists():
        # Remove all files and subdirectories in the directory
        for item in rundir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        # Create the directory if it does not exist
        rundir.mkdir(parents=True, exist_ok=True)

    with open(rundir / "si.env", 'w') as f:
        dvarskill = " ".join([f'"_EXPR_{i}" "{var}"' for i, var in enumerate(design_variables)])
        print(textwrap.dedent(f"""
        simLibName = "{library}"
        simCellName ="{cell}"
        simViewName ="{view}"
        simSimulator = "spectre"
        simNotIncremental = 't
        simReNetlistAll = nil
        simViewList = '("spectre" "cmos_sch" "cmos.sch" "schematic" "veriloga")
        simStopList = '("spectre")
        simNetlistHier = 't
        nlFormatterClass = 'spectreFormatter
        nlCreateAmap = 't
        nlDesignVarNameList = '({dvarskill})
        """), file=f)

    shutil.copy(WARD / "cds.lib", rundir / "cds.lib")

    process = subprocess.run(["si", "-batch", "-command", "nl"], capture_output=True, text=True, cwd=rundir)
    stdout = process.stdout
    stderr = process.stderr
    print(stdout)
    print(stderr)
    if ('ERRROR' in stdout) or len(stderr):
        raise Exception("Not sure what's wrong, but check it out")

    nl = ""
    with open(rundir / "netlistHeader", 'r') as f:
        nl += f.read()
    nl += ''.join(
        [f'include "{scsfile}"\n' if type(scsfile) is str else f'include "{scsfile[0]}" section=tttt\n' for scsfile in
         scs_includes])
    nl += ''.join([f'parameters {k}={v}\n' for k, v in design_variables.items()])
    with open(rundir / "netlist", 'r') as f:
        nl += f.read()
    with open(rundir / "netlistFooter", 'r') as f:
        nl += f.read()
    if include_typical:
        nl += typical_scs_code + "\n"
    nl += additional_code
    with open(rundir / "input.scs", 'w') as f:
        f.write(nl)
    return rundir

