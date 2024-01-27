from compyct.backends import backend
from compyct.backends import ngspice_backend


def test_get_with_backend():
    assert isinstance(
        backend.MultiSimSesh.get_with_backend(None,backend='ngspice'),
        ngspice_backend.NgspiceMultiSimSesh)

def test_get_osdi_path():
    ngspice_backend.get_confirmed_osdi_path('trivial_xtor.va')


if __name__=='__main__':
    test_get_with_backend()
    test_get_osdi_path()
