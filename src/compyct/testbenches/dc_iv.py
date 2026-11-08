
from typing import TYPE_CHECKING



from compyct.definitions.sweep import SingleUniformSweep, SweepPlan
from compyct.definitions.testbench import TestBench, DUT, Model

if TYPE_CHECKING:
    from InSpice.Spice.Netlist import Circuit
    from InSpice.Spice.Simulation import Simulation
    from typing import Mapping

class DCIdVg(TestBench):
    DEFAULT_MAPPING = {'g':'gate', 'd':'drain', 'b':'gnd', 's':'gnd'}

    def __init__(self, name:str, sweep_plan: SweepPlan, 
                    biases: Mapping[str, str|float]=DEFAULT_MAPPING):
        super().__init__(name)
        if (sweep_plan.inner and sweep_plan.inner!='VG'):
            raise ValueError("Inner sweep must be VG for DCIdVg testbench.")
        self._sweep_plan = sweep_plan
        if not ('gate' in biases.values() and 'drain' in biases.values()):
            raise ValueError("Biases must map at least one terminal to each of 'gate' and 'drain'.")
        self._biases = biases

    def get_simulation(self, dut: DUT, model: Model) -> Simulation:
        from InSpice.Spice.Netlist import Circuit
        from InSpice.Spice.Simulator import Simulator
        circuit = Circuit(self.name)
        netmap={}
        for term, voltage_or_net in self._biases.items():
            if isinstance(voltage_or_net, (int, float)):
                circuit.V('aux_'+term, 'auxnet_'+term, circuit.gnd, voltage_or_net) # type:ignore
                netmap[term] = 'auxnet_'+term
            elif voltage_or_net in ('gate', 'drain'):
                circuit.V(voltage_or_net, voltage_or_net, circuit.gnd, 0) # type:ignore
                netmap[term] = voltage_or_net
            elif voltage_or_net == 'gnd': 
                netmap[term] = circuit.gnd

        self.embedding.add_to_circuit(circuit, netmap, dut, model)

        simulation = Simulator.factory(simulator='vacask').simulation(circuit)
        for sweep in self._sweep_plan.iter_sweeps():
            if isinstance(sweep, SingleUniformSweep):
                simulation.dc(vgate=sweep.to_slice(), run=False)
            break   # JUST FOR TESTING REMOVE REMOVE
        return simulation


if __name__ == '__main__':
    from compyct.definitions.sweep import RectangularSweepPlan
    from compyct.definitions.testbench import PDUT
    class Short(Model):
        def add_to_circuit(self, circuit:Circuit, netmap:dict[str,str], iparams:InstanceParams) -> None:
            circuit.R('shortgd', netmap['g'], netmap['d'], iparams['Rgd'])
            circuit.R('shortds', netmap['d'], netmap['s'], 5)
    dciv=DCIdVg('DCIdVg',
           RectangularSweepPlan._('VG', VG_start=0, VG_stop=1, VG_step=0.1, VD_vals=[.05,1]),
           biases={'g':'gate','d':'drain','s':'gnd','b':-.5})                
    import time
    
    from InSpice.Spice.Simulator import Simulator
    Simulator.factory(simulator='vacask')
    print("Creating DCIdVg netlist...")
    start_time = time.time()
    simulation=dciv.get_simulation(PDUT({'Rgd':10}),Short())
    print(f"Netlist created in {time.time()-start_time:.6f} seconds.")    
    #print(simulation)
    print("Running DCIdVg simulation...")
    start_time = time.time()
    analysis = simulation.simulator.run(simulation)
    print(f"Simulation completed in {time.time()-start_time:.6f} seconds.")

    print("Creating DCIdVg netlist...")
    start_time = time.time()
    simulation=dciv.get_simulation(PDUT({'Rgd':20}),Short())
    print(f"Netlist created in {time.time()-start_time:.6f} seconds.")    
    #print(simulation)
    print("Running DCIdVg simulation...")
    start_time = time.time()
    analysis = simulation.simulator.run(simulation)
    print(f"Simulation completed in {time.time()-start_time:.6f} seconds.")


    import numpy as np
    idrain = -np.array(analysis.Vdrain)
    print(f"Drain current: {idrain}")
