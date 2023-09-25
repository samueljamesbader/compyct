from pathlib import Path
import subprocess
import os

def compile_va_to_osdi(va_dir, osdi_dir):
    osdi_dir=Path(osdi_dir)
    va_dir=Path(va_dir)
    for osdipath in osdi_dir.glob("*"):
        osdipath.unlink()
    for vapath in va_dir.glob("*.va"):
        osdipath=osdi_dir/vapath.name.replace(".va",".osdi")
        subprocess.run(["openvaf",str(vapath),"-o",str(osdipath)])