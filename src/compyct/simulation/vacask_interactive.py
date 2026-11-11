"""Interactive VACASK backend for InSpice.

Currently a stub that behaves exactly like InSpice's VACASK backend.

Importing this module registers ``simulator='vacask_interactive'`` with ``Simulator.factory``.
"""

import logging

from InSpice.Spice.Simulator import Simulator
from InSpice.Spice.Vacask.Simulation import VacaskSimulation
from InSpice.Spice.Vacask.Simulator import VacaskSimulator

_module_logger = logging.getLogger(__name__)


class VacaskInteractiveSimulation(VacaskSimulation):

    _logger = _module_logger.getChild('VacaskInteractiveSimulation')


class VacaskInteractiveSimulator(VacaskSimulator):

    _logger = _module_logger.getChild('VacaskInteractiveSimulator')

    SIMULATOR = 'vacask_interactive'

    def simulation(self, circuit, **kwargs):
        return VacaskInteractiveSimulation(self, circuit, **kwargs)


_original_factory = Simulator.factory.__func__


@classmethod
def _factory(cls, *args, **kwargs):
    if kwargs.get('simulator') == VacaskInteractiveSimulator.SIMULATOR:
        kwargs.pop('simulator')
        return VacaskInteractiveSimulator(*args, **kwargs)
    return _original_factory(cls, *args, **kwargs)


if VacaskInteractiveSimulator.SIMULATOR not in Simulator.SIMULATORS:
    Simulator.factory = _factory # type:ignore
    Simulator.SIMULATORS += (VacaskInteractiveSimulator.SIMULATOR,) # type:ignore
