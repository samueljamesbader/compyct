from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence, cast, overload

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

SWEEP_STEP_REL_TOL = 1e-3

class SingleSweep:
    def to_numpy(self) -> NDArray[np.float64]:
        raise NotImplementedError("Subclasses must implement to_numpy()")

    @classmethod
    def _(cls,**kwargs) -> SingleSweep:
        if 'vals' in kwargs:
            return SingleNonUniformSweep(kwargs['vals'])
        else:
            return SingleUniformSweep(**kwargs)

class SingleUniformSweep(SingleSweep):

    @overload
    def __init__(self, s: SingleUniformSweepLike) -> None: ...
    def _from_existing(self, s: SingleUniformSweepLike) -> None:
        if isinstance(s, slice):
            self.__init__(start=s.start, stop=s.stop, step=s.step)
        elif isinstance(s, dict):
            self.__init__(**s) # type: ignore
        elif isinstance(s, SingleUniformSweep):
            self.start=s.start; self.stop=s.stop; self.step=s.step; self.nump=s.nump

    @overload
    def __init__(self, start:float, stop:float, step:Optional[float]=None, nump:Optional[int]=None) -> None: ...
    def _from_explicit(self, start:float, stop:float, step:Optional[float]=None, nump:Optional[int]=None) -> None:
        self.start:float = start
        self.stop:float = stop
        if (step is None) == (nump is None):
            raise ValueError("One of either step or num_points must be specified.")
        if step is not None:
            self.step:float = step
        else:
            assert nump is not None
            self.step:float = (stop - start) / (nump - 1)
        if nump is not None:
            if not isinstance(nump, int): raise ValueError("num_points must be an integer.")
            self.nump:int = nump
        else:
            assert step is not None
            self.nump:int = int((stop - start) / step) + 1
            if abs((stop - start) - (self.nump - 1) * step) > SWEEP_STEP_REL_TOL * abs(stop - start):
                raise ValueError(f"Step size {step} does not evenly divide the range [{start}, {stop}].")

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs: self._from_existing(args[0])
        else: self._from_explicit(*args, **kwargs)

    def to_numpy(self) -> NDArray[np.float64]:
        import numpy as np
        return np.linspace(self.start, self.stop, self.nump)

    def to_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)
type SingleUniformSweepLike = SingleUniformSweep | slice | dict[str, float|int] 

class SingleNonUniformSweep(SingleSweep):
    def __init__(self, values:SingleNonUniformSweepLike):
        if isinstance(values, SingleNonUniformSweep):
            self._values = values.to_numpy()
        else:
            import numpy as np
            self._values: NDArray[np.float64] = np.array(values, dtype=np.float64)

    def to_numpy(self) -> NDArray[np.float64]:
        return self._values.copy()
type SingleNonUniformSweepLike = SingleNonUniformSweep | Sequence[float] | NDArray[np.float64]

class SweepPlan:
    inner:str|None
    outer:str|None
    outer_vals:Sequence[float]

    def iter_sweeps(self) -> Iterator[SingleSweep]:
        raise NotImplementedError("Subclasses must implement iter_sweeps()")

class RectangularSweepPlan(SweepPlan):
    def __init__(self, inner_sweep:SingleSweep, outer_vals:Sequence[float], inner:str|None, outer:str|None):
        self.inner_sweep = inner_sweep
        self.outer_vals = outer_vals
        self.inner = inner
        self.outer = outer

    def iter_sweeps(self) -> Iterator[SingleSweep]:
        for outer_val in self.outer_vals:
            yield self.inner_sweep

    @classmethod
    def _(cls, inner: str, **kwargs) -> RectangularSweepPlan:

        inner_sweep_kwargs = {k.rsplit('_', maxsplit=1)[1]: v for k, v in kwargs.items() if k.startswith(inner+"_")}
        outer = next(k.rsplit('_', maxsplit=1)[0] for k in kwargs if not k.startswith(inner+"_"))
        outer_sweep_kwargs = {k.rsplit('_', maxsplit=1)[1]: v for k, v in kwargs.items() if k.startswith(outer+"_")}

        inner_sweep = SingleSweep._(**inner_sweep_kwargs)
        outer_vals = list(SingleSweep._(**outer_sweep_kwargs).to_numpy())

        return cls(inner_sweep, outer_vals, inner=inner, outer=outer)

class RaggedSweepPlan(SweepPlan):
    def __init__(self, sweeps:Mapping[float, SingleSweep], inner:str|None, outer:str|None):
        self.inner = inner
        self.outer = outer
        self.outer_vals = list(sweeps.keys())
        self._sweeps = sweeps

    def iter_sweeps(self) -> Iterator[SingleSweep]:
        for inner_sweep in self._sweeps.values():
            yield inner_sweep
