from __future__ import annotations
from typing import TYPE_CHECKING
from compyct.templates import SimTemplate
from compyct.backends.spectre_util import export_netlist

if TYPE_CHECKING:
    from compyct.backends.spectre_backend import SpectreNetlister

class SpectreSimTemplate(SimTemplate):

    def __init__(self, netlist_file:str, scs_includes:list[str]=[], additional_code="", **kwargs):
        super().__init__(**kwargs)
        self._netlist_file = netlist_file
        self._scs_includes = scs_includes
        self._additional_code = additional_code

    @property
    def scs_includes(self) -> list[str]: return self._scs_includes
    @property
    def va_includes(self) -> list[str]: return []
    @property
    def additional_code(self) -> str:
        return self._additional_code

    def get_netlist(self, netlister: SpectreNetlister):
        with open(self._netlist_file,'r') as f:
            return f.read()+'\n'+self.additional_code


class CadenceSimTemplate(SpectreSimTemplate):
    
    def __init__(self, library, cell, view='schematic', design_variables={},
                 scs_includes=[], additional_code="", include_typical=True,
                 force_reexport=False, **kwargs):
        self.library = library
        self.cell = cell
        self.view = view
        self.design_variables = design_variables
        self.include_typical = include_typical

        from compyct import CACHE_DIR
        rundir = CACHE_DIR / 'spyctre' / f"{self.library}__{self.cell}__{self.view}"
        if force_reexport or (not rundir.exists()):
            print(f"Exporting netlist for {self.library}::{self.cell} to {rundir}")
            rundir=export_netlist(
                self.library,
                self.cell,
                view=self.view,
                design_variables=self.design_variables,
                scs_includes=[], # handled by Netlister
                additional_code="", # handled by SpectreSimTemplate,
                include_typical=self.include_typical,
                rundir=rundir
            )
            assert (rundir/'input.scs').exists(), f"Netlist export failed for {self.library}::{self.cell}"
            print(f"Successfully exported netlist")
        else:
            print(f"Using cached netlist for {self.library}::{self.cell}")# from {rundir}")
            
        super().__init__(str(rundir/'input.scs'),scs_includes=scs_includes,additional_code=additional_code,**kwargs)
