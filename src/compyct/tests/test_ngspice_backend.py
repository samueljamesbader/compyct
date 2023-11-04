import numpy as np
from compyct.backends import backend
from compyct.backends import ngspice_backend
from compyct.backends.backend import MultiSimSesh
from compyct.examples.trivial_xtor_defs import TrivialXtorParamSet, TrivialXtor
from compyct.templates import TemplateGroup, DCIVTemplate, CVTemplate, IdealPulsedIdVdTemplate, SParTemplate


def test_get_with_backend():
    assert isinstance(
        backend.MultiSimSesh.get_with_backend(None,backend='ngspice'),
        ngspice_backend.NgspiceMultiSimSesh)

def test_get_osdi_path():
    ngspice_backend.get_confirmed_osdi_path('trivial_xtor.va')

def test_trivial_xtor_dciv():
    paramset=TrivialXtorParamSet()
    tg=TemplateGroup(thedciv=DCIVTemplate(model_paramset=paramset))
    meas_data={'thedciv':TrivialXtor(paramset=paramset).evaluate_template(tg['thedciv'])}
    with MultiSimSesh.get_with_backend(tg,backend='ngspice') as sim:
        res=sim.run_with_params()

    assert len(meas_data['thedciv']['IdVg'])>0
    assert len(meas_data['thedciv']['IdVd'])>0

    for vd in meas_data['thedciv']['IdVg']:
        assert np.allclose(
            meas_data['thedciv']['IdVg'][vd]["ID/W [uA/um]"],
            res['thedciv']['IdVg'][vd]["ID/W [uA/um]"],
            rtol=1e-3)
    for vg in meas_data['thedciv']['IdVd']:
        assert np.allclose(
            meas_data['thedciv']['IdVd'][vg]["ID/W [uA/um]"],
            res['thedciv']['IdVd'][vg]["ID/W [uA/um]"],
            rtol=1e-3)

def test_trivial_xtor_cv():
    paramset=TrivialXtorParamSet()
    tg=TemplateGroup(thecv=CVTemplate(model_paramset=paramset))
    meas_data={'thecv':TrivialXtor(paramset=paramset).evaluate_template(tg['thecv'])}
    with MultiSimSesh.get_with_backend(tg,backend='ngspice') as sim:
        res=sim.run_with_params()

    assert np.allclose(
        meas_data['thecv'][0]["Cgg [fF/um]"],
        res['thecv'][0]["Cgg [fF/um]"],
        rtol=1e-3)

def test_trivial_xtor_psiv():
    paramset=TrivialXtorParamSet()
    tg=TemplateGroup(thepsiv=IdealPulsedIdVdTemplate(model_paramset=paramset))
    meas_data={'thepsiv':TrivialXtor(paramset=paramset).evaluate_template(tg['thepsiv'])}
    with MultiSimSesh.get_with_backend(tg,backend='ngspice') as sim:
        res=sim.run_with_params()

    assert len(meas_data['thepsiv'])>0

    for vg in meas_data['thepsiv']:
        assert np.allclose(
            meas_data['thepsiv'][vg]["ID/W [uA/um]"],
            res['thepsiv'][vg]["ID/W [uA/um]"],
            rtol=1e-3,atol=1e-6)

def test_trivial_xtor_spar():
    paramset=TrivialXtorParamSet().copy_with_changes({'gtrap':0})
    tg=TemplateGroup(thespar=SParTemplate(vg=.6,vd=1.8,pts_per_dec=4,fstart='10meg',fstop='10e9',model_paramset=paramset))
    meas_data={'thespar':TrivialXtor(paramset=paramset).evaluate_template(tg['thespar'])}
    with MultiSimSesh.get_with_backend(tg,backend='ngspice') as sim:
        res=sim.run_with_params()

    assert np.allclose(
        meas_data['thespar'][0]["freq"],
        res['thespar'][0]["freq"],
        rtol=1e-3,atol=1e-6)
    assert np.allclose(
        meas_data['thespar'][0]["Y11"],
        res['thespar'][0]["Y11"],
        rtol=1e-2,atol=1e-6)
    assert np.allclose(
        meas_data['thespar'][0]["Y12"],
        res['thespar'][0]["Y12"],
        rtol=1e-2,atol=1e-6)
    assert np.allclose(
        meas_data['thespar'][0]["Y21"],
        res['thespar'][0]["Y21"],
        rtol=1e-2,atol=1e-6)
    #assert np.allclose(
    #    meas_data['thespar'][0]["Y22"],
    #    res['thespar'][0]["Y11"],
    #    rtol=1e-2,atol=1e-6)
    #print(res['thespar'][0])
    # Actual comparison

if __name__=='__main__':
    #test_get_with_backend()
    #test_get_osdi_path()
    test_trivial_xtor_spar()