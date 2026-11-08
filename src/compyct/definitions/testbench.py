from typing import TYPE_CHECKING, Optional, Self

if TYPE_CHECKING:
    from InSpice.Spice.Netlist import Circuit
    from InSpice.Spice.Simulation import Simulation

class BenchData: 
    dut: DUT

class Embedding:
    def add_to_circuit(self, circuit: Circuit, netmap: dict[str,str], dut: DUT, model: Model) -> None:
        raise NotImplementedError("Embedding subclasses must implement add_to_circuit()")
class NilEmbedding(Embedding):
    def add_to_circuit(self, circuit: Circuit, netmap: dict[str,str], dut: DUT, model: Model) -> None:
        dut.add_to_circuit(circuit, netmap, model)

class InstanceParams: pass 

class Model:
    def add_to_circuit(self, circuit: Circuit, netmap: dict[str,str], iparams: InstanceParams) -> None:
        raise NotImplementedError("Model subclasses must implement add_to_circuit()")

class DUT:
    def add_to_circuit(self, circuit: Circuit, netmap: dict[str, str], model: Model) -> None:
        raise NotImplementedError("DUT subclasses must implement add_to_circuit()")
class PDUT(DUT):
    def __init__(self, iparams: InstanceParams):
        self.iparams = iparams
    def add_to_circuit(self, circuit: Circuit, netmap: dict[str, str], model: Model) -> None:
        model.add_to_circuit(circuit, netmap, self.iparams)

class TestBench:
    def __init__(self, name:str, embedding: Embedding=NilEmbedding()):
        self.name:str = name
        self.embedding = embedding

    def with_DUT(self, dut: DUT) -> ConnectedTestBench:
        return ConnectedTestBench(self, dut)

    @classmethod
    def like_data(cls, data: BenchData, *args, **kwargs) -> Self:
        raise NotImplementedError("TestBench subclasses must implement like_data()")
    
    def get_simulation(self, dut: DUT, model: Model) -> Simulation:
        raise NotImplementedError("TestBench must implement get_simulation()")

class TestBenchTemplate[TB: TestBench]:
    def __init__(self, tbclass:type[TB], *args, **kwargs):
        self.tbclass = tbclass
        self.args = args
        self.kwargs = kwargs
    def bind(self, data: BenchData) -> ConnectedTestBench[TB]:
        return self.tbclass.like_data(data, *self.args, **self.kwargs).with_DUT(data.dut)

class ConnectedTestBench[TB: TestBench]:
    def __init__(self, testbench: TB, dut: DUT):
        self.testbench = testbench
        self.dut = dut

    def get_simulation(self, model: Model) -> Simulation:
        return self.testbench.get_simulation(self.dut, model)
