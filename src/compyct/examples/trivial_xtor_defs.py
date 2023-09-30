from compyct.templates import TemplateGroup, DCIVTemplate, DCIdVgTemplate, DCIdVdTemplate, JointTemplate
from compyct.paramsets import CMCParamSet, spicenum_to_float
from scipy.constants import elementary_charge as q, Boltzmann as kb
import pandas as pd
import numpy as np

class TrivialXtorParamSet(CMCParamSet):
    terminals=['d','g','s','b']
    def __init__(self,**kwargs):
        super().__init__(model='trivial_xtor',vaname='trivial_xtor.va',**kwargs)
    def get_total_device_width(self):
        return spicenum_to_float(self.get_value('w'))


class TrivialXtor():
    def __init__(self,paramset):
        self.paramset=paramset
    def qg_oa(self,VGS,T):
        cg=spicenum_to_float(self.paramset.get_value('cg'))
        n=spicenum_to_float(self.paramset.get_value('n'))
        vt0=spicenum_to_float(self.paramset.get_value('vt0'))
        vth = kb*T/q
        return cg*np.log(1+np.exp((VGS-vt0)/(n*vth)))
    def v(self,VDS,T):
        u0=spicenum_to_float(self.paramset.get_value('u0'))
        l=spicenum_to_float(self.paramset.get_value('l'))
        vs=spicenum_to_float(self.paramset.get_value('vs'))
        beta=2
        vu= u0 * VDS / l
        return vu/(1+(vu/vs)**beta)**(1/beta)
    def ID(self,VD,VG,VS,VB,T):
        w=spicenum_to_float(self.paramset.get_value('w'))
        qg_oa=self.qg_oa(VGS=VG-VS,T=T)
        v = self.v(VD-VS,T)
        return w * qg_oa * v

    def evaluate_template(self,simtemp,T=273.15+27):
        w=spicenum_to_float(self.paramset.get_value('w'))
        if isinstance(simtemp,DCIdVgTemplate):
            results={}
            start,step,stop=simtemp.vg_range
            VG=np.arange(start,stop+1e-6,step)
            for vd in simtemp.vd_values:
                results[vd]=pd.DataFrame({'VG':VG,'ID/W [uA/um]':self.ID(vd,VG,0,0,T)/w})
            return results

        elif isinstance(simtemp,DCIdVdTemplate):
            results={}
            start,step,stop=simtemp.vd_range
            VD=np.arange(start,stop+1e-6,step)
            for vg in simtemp.vg_values:
                results[vg]=pd.DataFrame({'VD':VD,'ID/W [uA/um]':self.ID(VD,vg,0,0,T)/w})
            return results
        elif isinstance(simtemp,JointTemplate):
            return {k:self.evaluate_template(subsimtemp) for k,subsimtemp in simtemp.subtemplates.items()}
