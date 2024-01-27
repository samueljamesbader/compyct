from compyct.templates import TemplateGroup, DCIVTemplate, DCIdVgTemplate, DCIdVdTemplate, JointTemplate, CVTemplate, \
    IdealPulsedIdVdTemplate, SParTemplate
from compyct.paramsets import CMCParamSet, spicenum_to_float
from scipy.constants import elementary_charge as q, Boltzmann as kb
import pandas as pd
import numpy as np

class TrivialXtorParamSet(CMCParamSet):
    terminals=['d','g','s','b']
    def __init__(self,**kwargs):
        super().__init__(vaname='trivial_xtor.va',**kwargs)
    @staticmethod
    def get_total_device_width_for_patch(patch):
        return spicenum_to_float(patch['w'])


class TrivialXtor():
    def __init__(self,patch):
        self.patch=patch
    def qg_oa(self,VGS,VDS,T,trap_state='DC'):
        cg=spicenum_to_float(self.patch.get('cg'))
        n=spicenum_to_float(self.patch.get('n'))
        vt0=spicenum_to_float(self.patch.get('vt0'))
        gtrap=spicenum_to_float(self.patch.get('gtrap'))
        vth = kb*T/q
        if trap_state=='DC':
            vtshift=gtrap*(VDS-VGS)
        else:
            vtshift=gtrap*(trap_state['VD']-trap_state['VG'])
        return n*cg*vth*np.log(1+np.exp((VGS-vt0-vtshift)/(n*vth)))
    def v(self,VDS,T):
        u0=spicenum_to_float(self.patch.get('u0'))
        l=spicenum_to_float(self.patch.get('l'))
        vs=spicenum_to_float(self.patch.get('vs'))
        beta=2
        vu= u0 * VDS / l
        return vu/(1+(vu/vs)**beta)**(1/beta)
    def ID(self,VD,VG,VS,VB,T,trap_state='DC'):
        w=spicenum_to_float(self.patch.get('w'))
        qg_oa=self.qg_oa(VGS=VG-VS,VDS=VD-VS,T=T,trap_state=trap_state)
        v = self.v(VD-VS,T)
        return w * qg_oa * v
    def GM(self,VD,VG,VS,VB,T,trap_state='DC',traps_move=True):
        assert not traps_move
        l=spicenum_to_float(self.patch.get('l'))
        v = self.v(VD-VS,T)
        return self.Cgg(VD,VG,VS,VB,T,trap_state=trap_state)/l * v
    def G0(self,VD,VG,VS,VB,T,trap_state='DC',traps_move=True):
        assert not traps_move
        u0=spicenum_to_float(self.patch.get('u0'))
        l=spicenum_to_float(self.patch.get('l'))
        vs=spicenum_to_float(self.patch.get('vs'))
        beta=2
        #vu= u0 * VDS / l
        #return vu/(1+(vu/vs)**beta)**(1/beta)
    def Cgg(self,VD,VG,VS,VB,T,trap_state='DC',traps_move=False):
        VGS=VG-VS
        w=spicenum_to_float(self.patch.get('w'))
        l=spicenum_to_float(self.patch.get('l'))
        cg=spicenum_to_float(self.patch.get('cg'))
        n=spicenum_to_float(self.patch.get('n'))
        vt0=spicenum_to_float(self.patch.get('vt0'))
        gtrap=spicenum_to_float(self.patch.get('gtrap'))
        vth = kb*T/q
        if traps_move: assert np.allclose(VD,VS), "VD==VS for CV"
        if trap_state=='DC':
            vtshift=gtrap*(VD-VG)
        else:
            vtshift=gtrap*(trap_state['VD']-trap_state['VG'])
        return w * l * cg / (1+np.exp((vt0+vtshift-VGS)/(n*vth)))

    def evaluate_template(self,simtemp,T=273.15+27):
        w=spicenum_to_float(self.patch.get('w'))
        if isinstance(simtemp,DCIdVgTemplate):
            results={}
            start,step,stop=simtemp.vg_range
            VG=np.arange(start,stop+1e-6,step)
            for vd in simtemp.vd_values:
                for sdir in ('f','r'):
                    results[(vd,sdir)]=pd.DataFrame({'VG':VG,
                                              'ID/W [uA/um]':self.ID(vd,VG,0,0,T,trap_state='DC')/w,
                                              'GM/W [uS/um]':self.ID(vd,VG,0,0,T,trap_state='DC')/w*np.NaN,
                                              })
            return results

        elif isinstance(simtemp,DCIdVdTemplate):
            results={}
            start,step,stop=simtemp.vd_range
            VD=np.arange(start,stop+1e-6,step)
            for vg in simtemp.vg_values:
                for sdir in ('f','r'):
                    results[(vg,sdir)]=pd.DataFrame({'VD':VD,'ID/W [uA/um]':self.ID(VD,vg,0,0,T,trap_state='DC')/w})
            return results

        elif isinstance(simtemp,CVTemplate):
            results={}
            start,step,stop=simtemp.vg_range
            VG=np.arange(start,stop+1e-6,step)
            results[0]=pd.DataFrame({'VG':VG,'Cgg [fF/um]':self.Cgg(0,VG,0,0,T,trap_state='DC')/w * 1e9})
            return results

        elif isinstance(simtemp,IdealPulsedIdVdTemplate):
            results={}
            start,step,stop=simtemp.vd_range
            VD=np.arange(start,stop+1e-6,step)
            for vg in simtemp.vg_values:
                for sdir in ('f','r'):
                    results[(vg,sdir)]=pd.DataFrame({'VD':VD,'ID/W [uA/um]':self.ID(VD,vg,0,0,T,
                                             trap_state={'VD':simtemp.vdq,'VG':simtemp.vgq,'VS':0})/w})
            return results

        elif isinstance(simtemp,SParTemplate):
            results={}

            freq=np.power(10,np.arange(np.log10(simtemp.fstart),np.log10(simtemp.fstop)+1e-6,1/simtemp.pts_per_dec))
            w=2*np.pi*freq

            vg,vd=simtemp.outer_values[0]
            Cgs=self.Cgg(VD=vd,VG=vg,VS=0,VB=0,T=simtemp.temp)
            gm=self.GM(VD=vd,VG=vg,VS=0,VB=0,T=simtemp.temp,traps_move=False)

            results[(vg,vd)]=pd.DataFrame({
                'freq':freq,
                'Y11':1j*w*Cgs,
                'Y12':freq*0,
                'Y21':gm,
                'Y22': freq*np.NaN
            })
            return results

        elif isinstance(simtemp,JointTemplate):
            return {k:self.evaluate_template(subsimtemp) for k,subsimtemp in simtemp.subtemplates.items()}
