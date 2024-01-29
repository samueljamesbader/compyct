import numpy as np
from compyct.backends import ngspice_backend
from compyct.backends.backend import MultiSimSesh
from compyct.examples.trivial_xtor_defs import TrivialXtorParamSet, TrivialXtor
from compyct.templates import TemplateGroup, DCIVTemplate, CVTemplate, IdealPulsedIdVdTemplate, SParTemplate

from _pytest.python import Metafunc
import os

def pytest_generate_tests(metafunc: Metafunc):
    if os.name=='nt': # on Windows test only ngspice
        metafunc.parametrize("backend",['ngspice'])
    else: # otherwise, test all backends
        metafunc.parametrize("backend",['ngspice','spectre'])

def test_trivial_xtor_dciv(backend):
    patch=TrivialXtorParamSet().mcp_()
    tg=TemplateGroup(thedciv=DCIVTemplate(patch=patch))
    meas_data={'thedciv':TrivialXtor(patch=patch).evaluate_template(tg['thedciv'])}
    with MultiSimSesh.get_with_backend(tg,backend=backend) as sim:
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

def test_trivial_xtor_cv(backend):
    patch=TrivialXtorParamSet().mcp_()
    tg=TemplateGroup(thecv=CVTemplate(patch=patch))
    meas_data={'thecv':TrivialXtor(patch=patch).evaluate_template(tg['thecv'])}
    with MultiSimSesh.get_with_backend(tg,backend=backend) as sim:
        res=sim.run_with_params()

    assert np.allclose(
        list(meas_data['thecv'].values())[0]["Cgg [fF/um]"],
        list(res['thecv'].values())[0]["Cgg [fF/um]"],
        rtol=1e-3)

def test_trivial_xtor_psiv(backend):
    patch=TrivialXtorParamSet().mcp_()
    tg=TemplateGroup(thepsiv=IdealPulsedIdVdTemplate(patch=patch))
    meas_data={'thepsiv':TrivialXtor(patch=patch).evaluate_template(tg['thepsiv'])}
    with MultiSimSesh.get_with_backend(tg,backend=backend) as sim:
        res=sim.run_with_params()

    assert len(meas_data['thepsiv'])>0

    for vg in meas_data['thepsiv']:
        assert np.allclose(
            meas_data['thepsiv'][vg]["ID/W [uA/um]"],
            res['thepsiv'][vg]["ID/W [uA/um]"],
            rtol=1e-3,atol=1e-6)

def test_trivial_xtor_spar(backend):
    patch=TrivialXtorParamSet().mcp_(gtrap=0)
    vg=.6;vd=1.8
    tg=TemplateGroup(thespar=SParTemplate(vg=vg,vd=vd,pts_per_dec=4,fstart='10meg',fstop='10e9',patch=patch))
    meas_data={'thespar':TrivialXtor(patch=patch).evaluate_template(tg['thespar'])}
    with MultiSimSesh.get_with_backend(tg,backend=backend) as sim:
        sim.print_netlists()
        res=sim.run_with_params()

    print(meas_data['thespar'][(vg,vd)].columns)
    print(res['thespar'][(vg,vd)].columns)
    print(f"ideal:")
    print(meas_data['thespar'][(vg,vd)][['Y11','Y21']])
    print('res:')
    print(res['thespar'][(vg,vd)][['Y11','Y21']])

    assert np.allclose(
        meas_data['thespar'][(vg,vd)]["freq"],
        res['thespar'][(vg,vd)]["freq"],
        rtol=1e-4,atol=1e-6)
    assert np.allclose(
        meas_data['thespar'][(vg,vd)]["Y11"],
        res['thespar'][(vg,vd)]["Y11"],
        rtol=5e-3,atol=1e-6)
    assert np.allclose(
        meas_data['thespar'][(vg,vd)]["Y12"],
        res['thespar'][(vg,vd)]["Y12"],
        rtol=1e-4,atol=1e-6)
    assert np.allclose(
        meas_data['thespar'][(vg,vd)]["Y21"],
        res['thespar'][(vg,vd)]["Y21"],
        rtol=1e-4,atol=1e-6)
    #assert np.allclose(
    #    meas_data['thespar'][(vg,vd)]["Y22"],
    #    res['thespar'][(vg,vd)]["Y11"],
    #    rtol=1e-2,atol=1e-6)
    #print(res['thespar'][0])
    # Actual comparison

if __name__=='__main__':
    #test_get_with_backend()
    #test_get_osdi_path()
    #test_trivial_xtor_dciv(backend='spectre')
    #print('passed dciv')
    #test_trivial_xtor_psiv(backend='spectre')
    test_trivial_xtor_spar(backend='ngspice')
    print("passed ngspice")
    test_trivial_xtor_spar(backend='spectre')
    print("passed spectre")
