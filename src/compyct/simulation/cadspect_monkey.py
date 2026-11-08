"""Monkey-patched Cadence Spectre ("CadSpect") backend for InSpice.

InSpice's own "Spectre" code (``InSpice.Spice.Spectre`` and the ``to_spectre()`` methods) targets
VACASK's Spectre-like dialect.  This module reuses the element/model rendering from that code but
writes a true Spectre top level (``simulator lang=spectre``, ``global 0``, top-level analyses, no
``control`` block or OSDI ``load`` lines) and runs the ``spectre`` executable.

Importing this module registers ``simulator='cadspect'`` with ``Simulator.factory``.
"""

import logging
import os
import shutil
import subprocess
import tempfile

from InSpice.Spice.Simulation import Simulation
from InSpice.Spice.Simulator import Simulator
from InSpice.Spice.Spectre import SpectreContext, format_spectre_value, DEFAULT_MODELS
from InSpice.Spice.AnalysisParameters import (
    OperatingPointAnalysisParameters, DCAnalysisParameters,
    ACAnalysisParameters, TransientAnalysisParameters,
)
from InSpice.Spice.Vacask.Simulation import VacaskSimulation
from InSpice.Spice.Vacask.RawFile import VacaskRawFile

_module_logger = logging.getLogger(__name__)

# VACASK module names (from InSpice's DEFAULT_MODELS) -> Spectre primitives they alias to
CADSPECT_PRIMITIVES = {
    'sp_resistor': 'resistor',
    'sp_capacitor': 'capacitor',
    'sp_inductor': 'inductor',
}


def _strip_v(node):
    """'V(out)' -> 'out'"""
    node = str(node)
    if node.startswith('V(') and node.endswith(')'):
        return node[2:-1]
    return node


class CadSpectSimulation(Simulation):

    _logger = _module_logger.getChild('CadSpectSimulation')

    def _translate_options(self):
        parts = []
        for key, value in self._options.items():
            k = key.lower()
            if k == 'savecurrents':
                self._logger.warning('SAVECURRENTS is not supported by CadSpect, ignored')
                continue
            if value is not None:
                parts.append(f"{k}={format_spectre_value(value)}")
            else:
                parts.append(k)
        return 'cadspectOptions options ' + ' '.join(parts)

    def _translate_analysis(self, analysis):
        """Return Spectre lines for one InSpice analysis."""
        if isinstance(analysis, OperatingPointAnalysisParameters):
            return ['op1 dc']

        if isinstance(analysis, DCAnalysisParameters):
            # InSpice/SPICE order: first sweep is innermost; Spectre nests outer sweeps around it
            params = analysis.parameters
            sweeps = []
            for i in range(0, len(params), 4):
                src = str(params[i]).lower()
                start, stop, step = (format_spectre_value(p) for p in params[i + 1:i + 4])
                target = 'param=temp' if src == 'temp' else f'dev={src} param=dc'
                sweeps.append(f'{target} start={start} stop={stop} step={step}')
            lines = [f'dc1 dc {sweeps[0]}']
            for n, sweep in enumerate(sweeps[1:], start=1):
                lines = [f'sw{n} sweep {sweep} {{'] + ['  ' + l for l in lines] + ['}']
            return lines

        if isinstance(analysis, ACAnalysisParameters):
            return [f'ac1 ac start={format_spectre_value(analysis.start_frequency)} '
                    f'stop={format_spectre_value(analysis.stop_frequency)} '
                    f'{analysis.variation}={analysis.number_of_points}']

        if isinstance(analysis, TransientAnalysisParameters):
            parts = ['tran1 tran',
                     f'step={format_spectre_value(analysis.step_time)}',
                     f'stop={format_spectre_value(analysis.end_time)}']
            if analysis.max_time is not None:
                parts.append(f'maxstep={format_spectre_value(analysis.max_time)}')
            if analysis.use_initial_condition:
                parts.append('skipdc=yes')
            return [' '.join(parts)]

        raise NotImplementedError(f"{type(analysis).__name__} is not supported by CadSpect")

    def to_spectre(self):
        """Generate the complete Spectre netlist."""
        context = SpectreContext()
        circuit_body = self._circuit.to_spectre(context)

        lines = [f"// {self._circuit.title}", 'simulator lang=spectre', 'global 0', '']

        directives = self._circuit._include_directives + self._circuit._lib_directives
        for directive in directives:
            lines.append(VacaskSimulation._format_include(directive))
        if directives:
            lines.append('')

        # Alias VACASK's default R/L/C modules to Spectre primitives.  Builtins (vsource, ...)
        # are already Spectre primitives and need no model line.
        for prefix in sorted(context.default_models):
            _, module = DEFAULT_MODELS[prefix]
            lines.append(f"model {module} {CADSPECT_PRIMITIVES[module]}")
        if context.osdi_files - {DEFAULT_MODELS[p][0] for p in context.default_models}:
            raise NotImplementedError(
                f"CadSpect cannot use VACASK OSDI models: {sorted(context.osdi_files)}")

        if self._circuit._parameters:
            params = ' '.join(f'{k}={v}' for k, v in self._circuit._parameters.items())
            lines.append(f'parameters {params}')

        if circuit_body:
            lines.append('')
            lines.extend(circuit_body.split('\n'))
        lines.append('')

        if self._node_set:
            lines.append('nodeset ' + ' '.join(
                f'{_strip_v(k)}={format_spectre_value(v)}' for k, v in self._node_set.items()))
        if self._initial_condition:
            lines.append('ic ' + ' '.join(
                f'{_strip_v(k)}={format_spectre_value(v)}' for k, v in self._initial_condition.items()))

        lines.append(self._translate_options())
        if self._saved_nodes:
            lines.append('save ' + ' '.join(sorted(self._saved_nodes)))

        for analysis_parameters in self._analyses.values():
            lines.extend(self._translate_analysis(analysis_parameters))

        return '\n'.join(lines) + '\n'

    def to_spice(self):
        raise NotImplementedError("CadSpect uses Spectre format, not SPICE. Use to_spectre().")

    def __str__(self):
        return self.to_spectre()


class CadSpectSimulator(Simulator):

    _logger = _module_logger.getChild('CadSpectSimulator')

    SIMULATOR = 'cadspect'
    CADSPECT_COMMAND = 'spectre'

    def __init__(self, **kwargs):
        self._cadspect_command = kwargs.get('cadspect_command') or self.CADSPECT_COMMAND
        self._AS_SIMULATOR = self.SIMULATOR

    @property
    def version(self):
        return ''

    def simulation(self, circuit, **kwargs):
        return CadSpectSimulation(self, circuit, **kwargs)

    def run(self, simulation, *args, **kwargs):
        tmp_dir = tempfile.mkdtemp()
        try:
            input_filename = os.path.join(tmp_dir, 'input.scs')
            raw_filename = os.path.join(tmp_dir, 'output.raw')
            with open(input_filename, 'w') as f:
                f.write(str(simulation))

            command = (self._cadspect_command, '-format', 'nutbin', '-raw', raw_filename,
                       input_filename)
            self._logger.info('Run {}'.format(' '.join(command)))
            process = subprocess.run(command, capture_output=True, cwd=tmp_dir)
            output = (process.stdout + process.stderr).decode('utf-8', errors='replace')
            self._logger.info(os.linesep + output)
            if process.returncode != 0:
                raise RuntimeError(f"CadSpect exited with code {process.returncode}:\n{output}")

            with open(raw_filename, 'rb') as f:
                raw_file = VacaskRawFile(f.read())
        finally:
            shutil.rmtree(tmp_dir)

        raw_file.simulation = simulation
        return raw_file.to_analysis()


_original_factory = Simulator.factory.__func__


@classmethod
def _factory(cls, *args, **kwargs):
    if kwargs.get('simulator') == CadSpectSimulator.SIMULATOR:
        kwargs.pop('simulator')
        return CadSpectSimulator(*args, **kwargs)
    return _original_factory(cls, *args, **kwargs)


if CadSpectSimulator.SIMULATOR not in Simulator.SIMULATORS:
    Simulator.factory = _factory # type:ignore
    Simulator.SIMULATORS += (CadSpectSimulator.SIMULATOR,) # type:ignore
