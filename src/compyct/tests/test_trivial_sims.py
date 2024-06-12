import numpy as np
from compyct.backends import ngspice_backend
from compyct.backends.backend import MultiSimSesh
from compyct.examples.trivial_xtor_defs import TrivialXtorParamSet, TrivialXtor
from compyct.templates import TemplateGroup, DCIVTemplate, CVTemplate, IdealPulsedIdVdTemplate, SParTemplate, \
    SParVFreqTemplate, SParVBiasTemplate, LFNoiseVBiasTemplate, LFNoiseVFreqTemplate, HFNoiseVFreqTemplate, \
    HFNoiseVBiasTemplate

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

def test_trivial_xtor_sparvfreq(backend):
    patch=TrivialXtorParamSet().mcp_(gtrap=0)
    vg=.6;vd=1.8
    tg=TemplateGroup(thespar=SParVFreqTemplate(vg=vg,vd=vd,pts_per_dec=4,fstart='10meg',fstop='10e9',patch=patch))
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
def test_trivial_xtor_sparvbias(backend):
    patch=TrivialXtorParamSet().mcp_(gtrap=0)
    vgvds=[(.6,1.8),(.7,1.8)]
    tg=TemplateGroup(thespar=SParVBiasTemplate(vgvds=vgvds,frequency='1e9',patch=patch))
    meas_data={'thespar':TrivialXtor(patch=patch).evaluate_template(tg['thespar'])}
    with MultiSimSesh.get_with_backend(tg,backend=backend) as sim:
        sim.print_netlists()
        res=sim.run_with_params()

    for vg,vd in vgvds:
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

def test_trivial_xtor_lfnoisevfreq(backend):
    patch=TrivialXtorParamSet().mcp_(gtrap=0)
    vg,vd=.8,1.8
    tg=TemplateGroup(theflick=LFNoiseVFreqTemplate(vg=vg, vd=vd, fstart=1e0, fstop=1e4, pts_per_dec=1, patch=patch))
    meas_data={'theflick':TrivialXtor(patch=patch).evaluate_template(tg['theflick'])}
    with MultiSimSesh.get_with_backend(tg,backend=backend) as sim:
        sim.print_netlists()
        res=sim.run_with_params()
    for x in ['freq','sid [A^2/Hz]','svg [V^2/Hz]','gain [A/V]']:
        print(x,res['theflick'][(vg,vd)][x].iloc[0])
        assert np.allclose(
            meas_data['theflick'][(vg,vd)][x],
            res['theflick'][(vg,vd)][x],
            rtol=1e-4,atol=0), f"failed {x}"

    print('gain should be',TrivialXtor(patch=patch).GM(vd,vg,0,0,tg['theflick'].temp+273.15,trap_state='DC',traps_move=False))
    #TrivialXtor(patch=patch).GM(self,VD,VG,VS,VB,T,trap_state='DC',traps_move=True):

def test_trivial_xtor_lfnoisevbias(backend):
    patch=TrivialXtorParamSet().mcp_(gtrap=0)
    vgvds=[(.1,1.8),(1.2,1.8)]
    tg=TemplateGroup(theflick=LFNoiseVBiasTemplate(vgvds=vgvds, frequency=1e0, patch=patch))
    meas_data={'theflick':TrivialXtor(patch=patch).evaluate_template(tg['theflick'])}
    with MultiSimSesh.get_with_backend(tg,backend=backend) as sim:
        sim.print_netlists()
        res=sim.run_with_params()
    for vg,vd in vgvds:
        print(f"VG,VD={vg},{vd}")
        for x in ['freq','sid [A^2/Hz]','svg [V^2/Hz]','gain [A/V]']:
            print(x,res['theflick'][(vg,vd)][x].iloc[0])
            assert np.allclose(
                meas_data['theflick'][(vg,vd)][x],
                res['theflick'][(vg,vd)][x],
                rtol=1e-3,atol=0), f"failed {x}"

        print('gain should be',TrivialXtor(patch=patch).GM(vd,vg,0,0,tg['theflick'].temp+273.15,trap_state='DC',traps_move=False))
    #TrivialXtor(patch=patch).GM(self,VD,VG,VS,VB,T,trap_state='DC',traps_move=True):


def test_trivial_xtor_hfnoisevb(backend):
    print("Not checking results")
    patch = TrivialXtorParamSet().mcp_(gtrap=0)
    vgvds = [(.1, 1.8), (1.2, 1.8)]
    tg = TemplateGroup(thetherm=HFNoiseVBiasTemplate(vgvds=vgvds, frequency=1e9, patch=patch))
    #meas_data = {'thetherm': TrivialXtor(patch=patch).evaluate_template(tg['theflick'])}
    with MultiSimSesh.get_with_backend(tg, backend=backend) as sim:
        sim.print_netlists()
        res = sim.run_with_params()
    print(res['thetherm'][vgvds[1]][['NFmin','Rn']])

if __name__=='__main__':
    #test_get_with_backend()
    #test_get_osdi_path()
    #test_trivial_xtor_dciv(backend='spectre')
    #print('passed dciv')
    #test_trivial_xtor_psiv(backend='spectre')
    #test_trivial_xtor_sparvfreq(backend='ngspice')
    #test_trivial_xtor_sparvbias(backend='ngspice')
    #test_trivial_xtor_lfnoisevfreq(backend='ngspice')
    #test_trivial_xtor_lfnoisevbias(backend='ngspice')
    test_trivial_xtor_hfnoisevb(backend='ngspice')
    #print("passed ngspice")
    #test_trivial_xtor_sparvfreq(backend='spectre')
    #test_trivial_xtor_sparvbias(backend='spectre')
    #test_trivial_xtor_lfnoisevbias(backend='spectre')
    test_trivial_xtor_hfnoisevb(backend='spectre')
    #print("passed spectre")
    pass
