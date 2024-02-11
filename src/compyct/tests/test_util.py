from compyct.util import s2y, y2s, s2z, z2s
import numpy as np

def test_s2y2s():
    s=[np.r_[.1,.11],np.r_[.2,.22],np.r_[.3,.33],np.r_[.4,.44]]
    assert np.allclose(y2s(*s2y(*s)),s)

def test_s2z2s():
    s=[np.r_[.1,.11],np.r_[.2,.22],np.r_[.3,.33],np.r_[.4,.44]]
    assert np.allclose(z2s(*s2z(*s)),s)

