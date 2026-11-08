# Core concepts

This page will describe the main abstractions in Compyct and how they fit
together. It is a stub; the APIs described here are still changing.

## TestBenches and DUTs and Models

A `TestBench` describes a measurement setup, such as a DC Id-Vg sweep.

A `DUT` (Device-Under-Test) is the object to simulate subject to the `TestBench`.
It may be e.g. a transistor specified by some `InstanceParamPatch`,
or it may be a fixed subcircuit.

A `Model` is a description of how to simulate a DUT.
It can be backed by e.g. a Verilog implementation, or a InSpice `SubCircuit` function.
A `Model` has a `ParamSet` describing the values which can be tuned.

An `Embedding` represents additional elements that may be present around the DUT
for a given `TestBench` but are not considered part of the `DUT`/`Model` in general.
For instance, an IdVg sweep may be simulated with the additional probe-resistances.
The most common case is a `NilEmbedding`, which just wires the `DUT` up directly.

## How a netlist gets made

A `TestBench` and a `DUT` together are called a `ConnectedTestBench`.
This is the basic simulatable entity in Compyct.  When `ConnectedTestBench.get_simulation(model)` is called:
- A `TestBench` creates an InSpice `Circuit` with the relevent test components (e.g. voltage sources, ports etc) and Spice analyses (e.g. dc sweep), and tells its `Embedding` what nets to wire up to.
- The `Embedding`, may add its own subcircuit of parasitics to the simulation, and tells the `DUT` what nets to wire up to.
- The `DUT` typically asks the `Model` to netlist it with the given instance parameters.