import numpy as np
from compyct.backends import backend
from compyct.backends import ngspice_backend
from compyct.backends.backend import MultiSimSesh
from compyct.examples.trivial_xtor_defs import TrivialXtorParamSet, TrivialXtor
from compyct.templates import TemplateGroup, DCIVTemplate


def test_get_with_backend():
    assert isinstance(
        backend.MultiSimSesh.get_with_backend(None,backend='ngspice'),
        ngspice_backend.NgspiceMultiSimSesh)

def test_get_osdi_path():
    ngspice_backend.get_confirmed_osdi_path('asmhemt.va')

def test_trivial_xtor_dciv():
    paramset=TrivialXtorParamSet()
    tg=TemplateGroup(thedciv=DCIVTemplate(model_paramset=paramset))
    meas_data={'thedciv':TrivialXtor(paramset=paramset).evaluate_template(tg['thedciv'])}
    with MultiSimSesh.get_with_backend(tg,backend='ngspice') as sim:
        res=sim.run_with_params()

    assert len(meas_data['thedciv']['IdVg'])>0
    assert len(meas_data['thedciv']['IdVd'])>0

    for vd in meas_data['thedciv']['IdVg']:
        np.allclose(meas_data['thedciv']['IdVg'][vd]["ID/W [uA/um]"],
                    res['thedciv']['IdVg'][vd]["ID/W [uA/um]"],
                    rtol=1e-3)
    for vg in meas_data['thedciv']['IdVd']:
        np.allclose(meas_data['thedciv']['IdVd'][vg]["ID/W [uA/um]"],
                res['thedciv']['IdVd'][vg]["ID/W [uA/um]"],
                rtol=1e-3)

if __name__=='__main__':
    #test_get_with_backend()
    test_get_osdi_path()