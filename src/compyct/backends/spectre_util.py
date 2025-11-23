import numpy as np

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


def export_netlist(library, cell, view='schematic', design_variables={},
                   scs_includes:str|list[str]="", additional_code="",
                   include_typical=True, rundir:Path|None=None):
    WARD = Path(os.environ['WARD'])
    if rundir is None:
        from compyct import CACHE_DIR
        rundir = CACHE_DIR / 'spyctre' / f"{library}__{cell}"
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
        simCellName = "{cell}"
        simViewName = "{view}"
        simSimulator = "spectre"
        simNotIncremental = 't
        simReNetlistAll = nil
        simViewList = '("spectre" "cmos_sch" "cmos.sch" "schematic" "veriloga")
        simStopList = '("spectre")
        simNetlistHier = 't
        nlFormatterClass = 'spectreFormatter
        nlCreateAmap = 't
        nlDesignVarNameList = {f"'({dvarskill})" if len(dvarskill) else "nil"}
        simNetlistHier = t
        """).strip(), file=f)

    shutil.copy(WARD / "cds.lib", rundir / "cds.lib")

    process = subprocess.run(["si", "-batch", "-command", "nl"], capture_output=True, text=True, cwd=rundir)
    stdout = process.stdout
    stderr = process.stderr
    print(stdout)
    print(stderr)

    stderr_lines = stderr.splitlines()
    ok_msgs=['System is not a supported distribution','We don\'t recognize OS','This OS does not appear to be','For more info']
    filtered_stderr='\n'.join([l for l in stderr_lines if not any(msg in l for msg in ok_msgs)])
    if ('ERRROR' in stdout) or len(filtered_stderr):
        raise Exception("Not sure what's wrong, but check it out")

    nl = ""
    with open(rundir / "netlistHeader", 'r') as f:
        nl += f.read()
    nl += ''.join(
        [f'include "{scsfile}"\n' if type(scsfile) is str else f'include "{scsfile[0]}" section=tttt\n' for scsfile in
         scs_includes])
    print(f"################ DESIGN VARIABLES {design_variables}")
    nl += ''.join([f'parameters {k}={v}\n' for k, v in design_variables.items() if v is not None])
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

