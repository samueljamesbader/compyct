import numpy as np
import bokeh
import bokeh.plotting
import pandas as pd
import bokeh.layouts
import panel as pn

from bokeh_smith import smith_chart
from compyct import logger
from compyct.backends.backend import Netlister
from compyct.gui import fig_legend_config, get_tools

from compyct.paramsets import ParamSet, spicenum_to_float, ParamPlace
from compyct.util import s2y



class SimTemplate():
    title='Unnamed plot'
    def __init__(self, model_paramset=None, internals_to_save=[]):
        if model_paramset is not None:
            self.set_paramset(model_paramset)
        self.internals_to_save=internals_to_save

    def set_paramset(self,model_paramset):
        self.model_paramset=model_paramset.copy() if model_paramset else None
        
    def parse_return(self, result):
        return result

    def postparse_return(self,parsed_result):
        return parsed_result

    def _parsed_result_to_cds_data(self,parsed_result):
        raise NotImplementedError
        
    def _update_cds_with_parsed_result(self,cds,parsed_result,flattened=False):
        data=self._parsed_result_to_cds_data(parsed_result)
        if flattened and len(list(data.values())[0]):
            flattened_data={k:np.concatenate(v) for k,v in data.items()
                                 if hasattr(v[0],'__len__')}
            npts=len(data[list(flattened_data.keys())[0]][0])
            flattened_data.update({k:np.repeat(v,npts) for k,v in data.items()
                                 if (not hasattr(v[0],'__len__'))})
            data=flattened_data
        if cds is None:
            cds=bokeh.models.ColumnDataSource(data)
        else:
            cds.data=data
        return cds

    def _validate_parsed_result(self,parsed_result):
        raise NotImplementedError


    # def _get_instance_param_part(self):
    #     return ' '.join(f'{k}=instparam_{k}' for k in self.model_paramset
    #                     if self.model_paramset.get_place(k)==ParamPlace.INSTANCE)
            

class MultiSweepSimTemplate(SimTemplate):
    def __init__(self, *args,
                 outer_variable=None, inner_variable=None,
                 outer_values=None, inner_range=None,
                 ynames=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_variable=outer_variable
        self.inner_variable=inner_variable
        self.outer_values=outer_values
        self.inner_range=inner_range
        self.ynames=ynames
        self._sources={}
        self._fig_is_clear=True
        
    def _parsed_result_to_cds_data(self,parsed_result):
        #assert len(self.ynames)==1
        #yname=self.ynames[0]
        
        data=dict(x=[],legend=[],**{yname:[] for yname in self.ynames})
        if parsed_result is None: parsed_result={}
        for key, df in parsed_result.items():
            data['x'].append(df[self.inner_variable].to_numpy())
            for i,yname in enumerate(self.ynames):
                data[yname].append(df[yname].to_numpy())
            data['legend'].append(key)
        return data
        
        
    def generate_figures(self, meas_data=None,
                         layout_params={}, y_axis_type='linear', x_axis_type='linear',
                         vizid=None):
    
        self._sources[vizid]\
               =[self._update_cds_with_parsed_result(cds=None,parsed_result=None)]

        if meas_data is not None:
            self._validate_parsed_result(meas_data)

        meas_cds_c=self._update_cds_with_parsed_result(cds=None,
                    parsed_result=meas_data,flattened=True)
        meas_cds_l=self._update_cds_with_parsed_result(cds=None,
                    parsed_result=meas_data,flattened=False)
        sim_cds=self._sources[vizid][0]

        return self._make_figures(meas_cds_c=meas_cds_c, meas_cds_l=meas_cds_l, sim_cds=sim_cds,
                                  layout_params=layout_params,
                                  y_axis_type=y_axis_type, x_axis_type=x_axis_type)

    def _make_figures(self, meas_cds_c, meas_cds_l, sim_cds, layout_params, y_axis_type='linear',x_axis_type='linear'):
        num_ys=len(self.ynames)
        if type(y_axis_type) is str: y_axis_type=[y_axis_type]*num_ys
        if type(x_axis_type) is str: x_axis_type=[x_axis_type]*num_ys
        figs=[]
        for i in range(num_ys):
            fig=bokeh.plotting.figure(tools=get_tools(),#x_range=self.vg_range,y_range=(1e-8,1),
                                      y_axis_type=y_axis_type[i],x_axis_type=x_axis_type[i],**layout_params)
            fig.circle(x='x',y=self.ynames[i],source=meas_cds_c,legend_field='legend')
            fig.multi_line(xs='x',ys=self.ynames[i],source=meas_cds_l)
            fig.multi_line(xs='x',ys=self.ynames[i],source=sim_cds,color='red')

            fig.yaxis.axis_label=self.ynames[i]#",".join(self.ynames)
            fig.xaxis.axis_label=self.inner_variable
            fig_legend_config(fig)
            figs.append(fig)
        return figs

    def update_figures(self, parsed_result, vizid=None):
        if parsed_result is None and self._fig_is_clear:
            return False
        logger.debug(f"Updating figure {self.__class__.__name__}")
        self._validate_parsed_result(parsed_result)
        self._update_cds_with_parsed_result(
            cds=self._sources[vizid][0],parsed_result=parsed_result)
        self._fig_is_clear=(parsed_result is None)
        return True

    def parsed_results_to_vector(self, parsed_results, rois, meas_parsed_results):
        if rois is None: return np.array([])
        if type(rois) is dict: rois=[rois]
        arr=[]
        for roi in rois:
            for k,sl in roi.items():
                inds,col=sl
                arr.append(self._rescale_vector(parsed_results[k].loc[sl],col,meas_parsed_results[k].loc[sl]))
        return np.concatenate(arr)
        
    def _rescale_vector(self, arr, col, meas_arr):
        raise NotImplementedError

    def _validate_parsed_result(self,parsed_result):
        if parsed_result is None: return
        if self.outer_variable is not None:
            assert parsed_result.keys()==set(self.outer_values), \
                f"{self.__class__.__name__} requests {self.outer_variable} {self.outer_values},"\
                f" but results are {list(parsed_result.keys())}"
        for val in parsed_result.values():
            assert np.isclose(val[self.inner_variable].to_numpy()[0],self.inner_range[0]),\
                f"{self.__class__.__name__} expects {self.inner_variable}[0]={self.inner_range[0]},"\
                f" but results have {val[self.inner_variable].to_numpy()[0]}"
            assert np.isclose(val[self.inner_variable].to_numpy()[-1],self.inner_range[2]),\
                f"{self.__class__.__name__} expects {self.inner_variable}[-1]={self.inner_range[-1]},"\
                f" but results have {val[self.inner_variable].to_numpy()[-1]}"
            #print(val[self.inner_variable])
            #import pdb; pdb.set_trace()
            assert np.allclose(np.diff(val[self.inner_variable]),self.inner_range[1],rtol=1e-3), \
                f"{self.__class__.__name__} expects Δ{self.inner_variable}={self.inner_range[1]},"\
                f" but results have {list(np.diff(val[self.inner_variable]))}"


class DCIdVdTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27, yscale='linear',
                 vg_values=[0,.6,1.2,1.8], vd_range=(0,.1,1.8), **kwargs):
        self.yscale=yscale
        super().__init__(outer_variable='VG', inner_variable='VD',
                         outer_values=vg_values, inner_range=vd_range,
                         ynames=['ID/W [uA/um]' if pol=='n' else '-ID/W [uA/um]'],
                         *args, **kwargs)
        self.temp=temp
        
        num_vd=(vd_range[2]-vd_range[0])/vd_range[1]
        assert abs(num_vd-round(num_vd))<1e-3, f"Make sure the IdVd range gives even steps {str(vd_range)}"

        self.pol=pol

    @property
    def vg_values(self): return self.outer_values
    @property
    def vd_range(self): return self.inner_range

    def get_schematic_listing(self,netlister:Netlister):
            #netlister.nstr_param(params={'vg':0,'vd':0})+\
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst",netd='netd',netg='netg',
                                        nets=netlister.GND,netb=netlister.GND,dt=None),
            netlister.nstr_VDC("D",netp='netd',netm=netlister.GND,dc=0),
            netlister.nstr_VDC("G",netp='netg',netm=netlister.GND,dc=0)]

    def get_analysis_listing(self,netlister:Netlister):
        analysis_listing=[]
        for i_vg,vg in enumerate(self.vg_values,start=1):
            analysis_listing.append(netlister.astr_altervdc('G',vg))
            analysis_listing.append(netlister.astr_sweepvdc('D',name=f'idvd_vgnum{i_vg}',
                start=self.vd_range[0],step=self.vd_range[1],stop=self.vd_range[2]))
        return analysis_listing

    def parse_return(self,result):
        parsed_result={}
        for i_vg,vg in enumerate(self.vg_values,start=1):
            for key in result:
                if f'idvd_vgnum{i_vg}' in key:
                    df=result[key].copy()
                    sgn=-1 if self.pol=='p' else 1
                    sgnstr="-" if self.pol=='p' else ''
                    df[f'{sgnstr}ID/W [uA/um]']=-sgn*df['vd#p']/\
                            self.model_paramset.get_total_device_width()
                    df[f'{sgnstr}IG/W [uA/um]']=-sgn*df['vg#p']/\
                            self.model_paramset.get_total_device_width()
                    parsed_result[vg]=df.rename(columns=\
                                {'netd':'VD','netg':'VG'})\
                            [['VD','VG',f'{sgnstr}ID/W [uA/um]',f'{sgnstr}IG/W [uA/um]']]
        return parsed_result
        
    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type',self.yscale)
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self, arr, col, meas_arr):
        if col[0]=='I':
            return 5*arr/np.max(meas_arr)

class DCIdVgTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27,
                 vd_values=[.05,1.8], vg_range=(0,.03,1.8), plot_gm=True, **kwargs):
        ynames=['ID/W [uA/um]' if pol=='n' else '-ID/W [uA/um]']
        if plot_gm: ynames+=['GM/W [uS/um]']
        super().__init__(outer_variable='VD', inner_variable='VG',
                         outer_values=vd_values, inner_range=vg_range,
                         ynames=ynames,
                         *args, **kwargs)
        self.temp=temp
        
        num_vg=(vg_range[2]-vg_range[0])/vg_range[1]+1
        assert abs(num_vg-round(num_vg))<1e-3, f"Make sure the IdVg range gives even steps {str(vg_range)}"
        
        self.pol=pol

    @property
    def vd_values(self): return self.outer_values
    @property
    def vg_range(self): return self.inner_range


    def get_schematic_listing(self,netlister:Netlister):
            #netlister.nstr_param(params={'vg':0,'vd':0})+\
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst",netd='netd',netg='netg',
                                        nets=netlister.GND,netb=netlister.GND,dt=None,
                                        internals_to_save=self.internals_to_save),
            netlister.nstr_VDC("D",netp='netd',netm=netlister.GND,dc=0),
            netlister.nstr_VDC("G",netp='netg',netm=netlister.GND,dc=0)]

    def get_analysis_listing(self,netlister:Netlister):
        analysis_listing=[]
        for i_vd,vd in enumerate(self.vd_values,start=1):
            analysis_listing.append(netlister.astr_altervdc('D',vd))
            analysis_listing.append(netlister.astr_sweepvdc('G',name=f'idvg_vdnum{i_vd}',
                start=self.vg_range[0],step=self.vg_range[1],stop=self.vg_range[2]))
        return analysis_listing
        
    def parse_return(self,result):
        parsed_result={}
        for i_vd,vd in enumerate(self.vd_values,start=1):
            for key in result:
                if f'idvg_vdnum{i_vd}' in key:
                    #import pdb; pdb.set_trace()
                    df=result[key].copy()
                    #print(vd,"gmi:",list(df['@ninst[gmi]']))
                    sgn=-1 if self.pol=='p' else 1
                    sgnstr="-" if self.pol=='p' else ''
                    df[f'{sgnstr}ID/W [uA/um]']=-sgn*df['vd#p']/\
                            self.model_paramset.get_total_device_width()
                    df[f'{sgnstr}IG/W [uA/um]']=-sgn*df['vg#p']/\
                            self.model_paramset.get_total_device_width()
                    # TODO: reinstate column restriction (removed for device internals testing)
                    parsed_result[vd]=df.rename(columns=\
                                {'netd':'VD','netg':'VG'})#\
                            #[['VD','VG',f'{sgnstr}ID/W [uA/um]',f'{sgnstr}IG/W [uA/um]']]
        return parsed_result

    def postparse_return(self,parsed_result):
        sgn=-1 if self.pol=='p' else 1
        sgnstr="-" if self.pol=='p' else ''
        for vd,df in parsed_result.items():
            df[f'GM/W [uS/um]']=np.gradient(sgn*df[f'{sgnstr}ID/W [uA/um]'],self.vg_range[1])
        return parsed_result
        
    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=[kwargs.get('y_axis_type','log'),'linear']
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self, arr, col, meas_arr):
        if col[0]=='I':
            return np.log10(np.abs(arr)+1e-14)
        else:
            return 5*arr/np.max(meas_arr)

class JointTemplate(SimTemplate):
    def __init__(self,subtemplates:dict, *args, model_paramset:ParamSet=None, **kwargs):
        self.subtemplates=subtemplates
        super().__init__(model_paramset=model_paramset,*args,**kwargs)

    def set_paramset(self,model_paramset):
        super().set_paramset(model_paramset)
        for st in self.subtemplates.values():
            st.model_paramset=self.model_paramset

    def postparse_return(self,parsed_result):
        return {k:t.postparse_return(parsed_result[k])for k,t in self.subtemplates.items()}

    def _validate_parsed_result(self,parsed_result):
        {k:t._validate_parsed_result(parsed_result[k])for k,t in self.subtemplates.items()}

    def update_figures(self, parsed_result, vizid=None):
        actually_did_update=False
        for k,t in self.subtemplates.items():
            if t.update_figures((parsed_result[k] if parsed_result else None),vizid=vizid):
                actually_did_update=True
        return actually_did_update


    def __getitem__(self,key):
        return self.subtemplates[key]

    #TODO: Move more of the DCIVTemplate code into JointTemplate

class DCIVTemplate(JointTemplate):
    def __init__(self, *args,pol='n',
                 idvd_vg_values=[0,.6,1.2,1.8], idvd_vd_range=(0,.1,1.8),
                 idvg_vd_values=[.05,1.8], idvg_vg_range=(0,.03,1.8), **kwargs):
        self._dcidvg=DCIdVgTemplate(*args, **kwargs,pol=pol,
                                    vd_values=idvg_vd_values, vg_range=idvg_vg_range)
        self._dcidvd=DCIdVdTemplate(*args, **kwargs,pol=pol,
                                    vg_values=idvd_vg_values, vd_range=idvd_vd_range)
        super().__init__(subtemplates={'IdVg':self._dcidvg,'IdVd':self._dcidvd}, *args, **kwargs)

    def get_schematic_listing(self,netlister:Netlister):
        lines=self._dcidvg.get_schematic_listing(netlister)
        assert lines==self._dcidvd.get_schematic_listing(netlister)
        return lines
    def get_analysis_listing(self,netlister:Netlister):
        return  self._dcidvg.get_analysis_listing(netlister)+\
                self._dcidvd.get_analysis_listing(netlister)
        
    def parse_return(self,result):
        result_idvg={k:v for k,v in result.items() if 'idvg' in k}
        result_idvd={k:v for k,v in result.items() if 'idvd' in k}
        return {'IdVg': self._dcidvg.parse_return(result_idvg),
                'IdVd': self._dcidvd.parse_return(result_idvd)}
    
    def generate_figures(self, meas_data=None, layout_params={}, vizid=None):
        # Typical templates receive None for no meas_data...
        meas_data={} if meas_data is None else meas_data

        figs1=self._dcidvg.generate_figures(
            meas_data=meas_data.get('IdVg',None), layout_params=layout_params, vizid=vizid)
        figs2=self._dcidvd.generate_figures(
            meas_data=meas_data.get('IdVd',None), layout_params=layout_params, vizid=vizid)
        return figs1+figs2
        

    def parsed_results_to_vector(self, parsed_results, roi, meas_parsed_results):
        return np.concatenate([
            self._dcidvg.parsed_results_to_vector(parsed_results['IdVg'],roi['IdVg'], meas_parsed_results['IdVg']),
            self._dcidvd.parsed_results_to_vector(parsed_results['IdVd'],roi['IdVd'], meas_parsed_results['IdVd'])])

class DCKelvinIDVDTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27, yscale='linear',
                 vg_values=[0,.6,1.2,1.8], idow_range=(.1,.1,1), **kwargs):
        self.yscale=yscale
        super().__init__(outer_variable='VG', inner_variable=('ID/W [uA/um]' if pol=='n' else '-ID/W [uA/um]'),
                         outer_values=vg_values, inner_range=idow_range,
                         ynames=['RW [kohm.um]'],
                         *args, **kwargs)
        self.temp=temp
        self.pol=pol

        num_id=(idow_range[2]-idow_range[0])/idow_range[1]
        assert abs(num_id-round(num_id))<1e-3, f"Make sure the KelvinIdVd range gives even steps {str(idow_range)}"

    @property
    def vg_values(self): return self.outer_values
    @property
    def idow_range(self): return self.inner_range

    def get_schematic_listing(self,netlister:Netlister):
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst",netd='netd',netg='netg',
                                        nets=netlister.GND,netb=netlister.GND,dt=None),
            netlister.nstr_IDC("D",netp='netd',netm=netlister.GND,dc=0),
            netlister.nstr_VDC("G",netp='netg',netm=netlister.GND,dc=0)]

    def get_analysis_listing(self,netlister:Netlister):
        w=self.model_paramset.get_total_device_width()
        analysis_listing=[]
        for i_vg,vg in enumerate(self.vg_values,start=1):
            analysis_listing.append(netlister.astr_altervdc('G',vg))
            analysis_listing.append(netlister.astr_sweepidc('D',name=f'idvd_vgnum{i_vg}',
                            start=-self.idow_range[0]*w,step=-self.idow_range[1]*w,stop=-self.idow_range[2]*w))
        return analysis_listing

    def parse_return(self,result):
        parsed_result={}
        for i_vg,vg in enumerate(self.vg_values,start=1):
            for key in result:
                if f'idvd_vgnum{i_vg}' in key:
                    df=result[key].copy()
                    sgn=-1 if self.pol=='p' else 1
                    sgnstr="-" if self.pol=='p' else ''
                    # You'd think this line would be 'id#p' instead of '#p'
                    # But for some reason Ngspice or PySpice is not attaching a name to this sweeping-current branch
                    # The '#p' only comes from my backend-code that smooths over the spectre-vs-spice deltas
                    df[f'{sgnstr}ID/W [uA/um]']=-sgn*df['#p']/ \
                                                self.model_paramset.get_total_device_width()
                    df[f'{sgnstr}IG/W [uA/um]']=-sgn*df['vg#p']/ \
                                                self.model_paramset.get_total_device_width()
                    df['RW [kohm.um]']=df['netd']/(sgn*df[f'{sgnstr}ID/W [uA/um]']) * 1e3
                    parsed_result[vg]=df.rename(columns= \
                                                    {'netd':'VD','netg':'VG'}) \
                        [['VD','VG',f'{sgnstr}ID/W [uA/um]',f'{sgnstr}IG/W [uA/um]','RW [kohm.um]']]
        return parsed_result

    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type',self.yscale)
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self, arr, col, meas_arr):
        return arr

class CVTemplate(MultiSweepSimTemplate):

    def __init__(self, *args,
                 vg_range=(0,.03,1.8), freq='1meg', **kwargs):
        super().__init__(outer_variable=None, inner_variable='VG', inner_range=vg_range,
                         ynames=['Cgg [fF/um]'],
                         *args, **kwargs)
        
        num_vg=(vg_range[2]-vg_range[0])/vg_range[1]+1
        assert abs(num_vg-round(num_vg))<1e-3, f"Make sure the CV VG range gives even steps {str(vg_range)}"
        
        self.freq=freq
        try:
            spicenum_to_float(freq)
        except:
            raise Exception(f"Invalid frequency: {freq}")

    @property
    def vg_range(self):
        return self.inner_range

    def get_schematic_listing(self,netlister:Netlister):
            #netlister.nstr_param(params={'vg':0})+\
        return [
            netlister.nstr_modeled_xtor("inst",netd=netlister.GND,netg='netg',
                                        nets=netlister.GND,netb=netlister.GND,dt=None),
            netlister.nstr_VAC("G",netp='netg',netm=netlister.GND,dc=0)]
    
    def get_analysis_listing(self,netlister:Netlister):
        return [netlister.astr_sweepvac('G',
                start=self.vg_range[0],step=self.vg_range[1],
                stop=self.vg_range[2], freq=self.freq, name='cv')]
        return analysis_listing
        
    def parse_return(self,result):
        parsed_result={}
        assert len(result)==1
        freq=spicenum_to_float(self.freq)
        df=list(result.values())[0]
        I=-df['vg#p']
        df['Cgg [fF/um]']=np.imag(I)/(2*np.pi*freq) /1e-15 /\
            (self.model_paramset.get_total_device_width()/1e-6)
        df['VG']=np.real(df['v-sweep'])
        # TODO: reinstate column restriction (removed for device internals testing)
        parsed_result={0:df}#[['VG','Cgg [fF/um]']]}
        return parsed_result

    def _rescale_vector(self,arr,col,meas_arr):
        return arr

    #def _validate_parsed_result(self,parsed_result):
    #    pass
    #    #assert parsed_result.keys()==set([0])
    #    # TODO: check spacings

class IdealPulsedIdVdTemplate(MultiSweepSimTemplate):

    def __init__(self, *args,
                 vg_values=[0,.6,1.2,1.8], vd_range=(0,.1,1.8),
                 pulse_width='1u',rise_time='100n',
                 vgq=0, vdq=0,
                 **kwargs):
        super().__init__(outer_variable='VG', inner_variable='VD',
                         outer_values=vg_values, inner_range=vd_range,
                         ynames=['ID/W [uA/um]'],
                         *args, **kwargs)
        
        num_vd=(vd_range[2]-vd_range[0])/vd_range[1]
        assert abs(num_vd-round(num_vd))<1e-3,\
            f"Make sure the IdVd range gives even steps {str(vd_range)}"
        
        self.pulse_width=pulse_width
        self.rise_time=rise_time
        self.vgq=vgq
        self.vdq=vdq

    @property
    def vg_values(self): return self.outer_values
    @property
    def vd_range(self): return self.inner_range

    def get_schematic_listing(self,netlister:Netlister):
        return [
            netlister.nstr_modeled_xtor("inst",netd='netd',netg='netg',
                                        nets=netlister.GND,netb=netlister.GND,dt=None),
            netlister.nstr_VStep("D",netp='netd',netm=netlister.GND,dc=self.vdq, rise_time=self.rise_time, final_v=0),
            netlister.nstr_VStep("G",netp='netg',netm=netlister.GND,dc=self.vgq,rise_time=self.rise_time,final_v=0)]

    def get_analysis_listing(self,netlister:Netlister):
        analysis_listing=[]
        vd_sweep=list(np.arange(self.vd_range[0],self.vd_range[2]+1e-9,self.vd_range[1]))
        vdpulses=vd_sweep*len(self.vg_values)
        vgpulses=list(np.repeat(self.vg_values,len(vd_sweep)))
        meas_delay=spicenum_to_float(self.pulse_width)/2
        analysis_listing.append(netlister.astr_idealpulsed(vdcs={'D':self.vdq,'G':self.vgq},vpulses={'D':vdpulses,'G':vgpulses},name='pulsed',
                                                           rise_time=self.rise_time,meas_delay=meas_delay))
        return analysis_listing

    def parse_return(self,result):
        result=result['pulsed'].reset_index()
        parsed_result={}
        vd_sweep=list(np.arange(self.vd_range[0],self.vd_range[2]+1e-9,self.vd_range[1]))
        for i_vg,vg in enumerate(self.vg_values):
            # Pandas slicing is endpoint-inclusive
            mask=slice((i_vg*len(vd_sweep)),((i_vg+1)*len(vd_sweep))-1)
            df={}
            df['ID/W [uA/um]']=-result.loc[mask,'vd#p']/\
                    self.model_paramset.get_total_device_width()
            df['IG/W [uA/um]']=-result.loc[mask,'vg#p']/\
                    self.model_paramset.get_total_device_width()
            df['VD']=result.loc[mask,'netd']
            df['VG']=result.loc[mask,'netg']
            parsed_result[vg]=pd.DataFrame(df)
        return parsed_result
        
    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type','linear')
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self,arr,col, meas_arr):
        return arr
        
    def _validate_parsed_result(self,parsed_result):
        pass
        #assert parsed_result.keys()==set([0])
        # TODO: check EVERYTHING

class SParTemplate(MultiSweepSimTemplate):

    def __init__(self, *args,
                 vg, vd, pts_per_dec, fstart, fstop, temp=27, **kwargs):
        try:
            fstart=spicenum_to_float(fstart)
            fstop=spicenum_to_float(fstop)
        except:
            raise Exception(f"Invalid frequency range: {fstart} - {fstop}")

        super().__init__(outer_variable=None, inner_variable='freq', inner_range=(fstart,pts_per_dec,fstop),
                         ynames=['|h21| [dB]','U [dB]','MAG-MSG [dB]','ReS11','ImS11','ReS22','ImS22','ReS12','ImS12'],
                         *args, **kwargs)
        self.vg=vg
        self.vd=vd
        self.temp=temp

        num_f=(np.log10(fstop)-np.log10(fstart))*pts_per_dec+1
        assert abs(num_f-round(num_f))<1e-3,\
            f"Make sure the SPar range gives log-even steps {pts_per_dec,str(fstart),str(fstop)}"

    @property
    def fstart(self):
        return self.inner_range[0]
    @property
    def fstop(self):
        return self.inner_range[2]
    @property
    def pts_per_dec(self):
        return self.inner_range[1]

    def get_schematic_listing(self,netlister:Netlister):
        #netlister.nstr_param(params={'vg':0,'vd':0})+\
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst",netd='netd',netg='netg',
                                        nets=netlister.GND,netb=netlister.GND,dt=None),
            netlister.nstr_port("D",netp='netd',netm=netlister.GND,dc=self.vd,portnum=2),
            netlister.nstr_port("G",netp='netg',netm=netlister.GND,dc=self.vg,portnum=1)]

    def get_analysis_listing(self,netlister:Netlister):
        return [netlister.astr_spar(pts_per_dec=self.pts_per_dec, fstart=self.fstart, fstop=self.fstop, name='spar')]
        return analysis_listing

    def parse_return(self,result):
        parsed_result={0: result['spar']}
        #assert len(result)==1
        #freq=spicenum_to_float(self.freq)
        #df=list(result.values())[0]
        #I=-df['vg#p']
        #df['Cgg [fF/um]']=np.imag(I)/(2*np.pi*freq) /1e-15 / \
        #                  (self.model_paramset.get_total_device_width()/1e-6)
        #df['VG']=np.real(df['v-sweep'])
        #parsed_result={0:df[['VG','Cgg [fF/um]']]}
        return parsed_result

    def postparse_return(self,parsed_result):
        df=parsed_result[0]

        df['freq']=np.real(df['freq'])
        
        if 'Y11' not in df.columns:
            df['Y11'],df['Y12'],df['Y21'],df['Y22']=s2y(df['S11'],df['S12'],df['S21'],df['S22'])

        # h21 is a current ratio, so 20x log
        df[f'|h21| [dB]']=20*np.log10(np.abs(df.Y21/df.Y11))

        # https://en.wikipedia.org/wiki/Mason%27s_invariant#Derivation_of_U
        # U is already a power ratio so just 10x log
        re=np.real; im=np.imag
        df[f'U [dB]']=10*np.log10(
            (np.abs(df.Y21-df.Y12)**2 /
                  (4*(re(df.Y11)*re(df.Y22)-re(df.Y12)*re(df.Y21)))))

        # https://www.microwaves101.com/encyclopedias/stability-factor
        Delta=df.S11*df.S22-df.S12*df.S21
        K = (1-np.abs(df.S11)**2-np.abs(df.S22)**2+np.abs(Delta)**2)/(2*np.abs(df.S21*df.S12))
        
        # this formula with 1/(K+sqrt(K^2-1)) is less common but more robust for large K
        # according to Microwaves 101 and easy to show it's equal.
        k2m1=np.clip(K**2-1,0,np.inf) # we only use the K>1 values of MAG anyway, so clip to avoid sqrt(-)
        MAG = (1/(K+np.sqrt(k2m1))) * np.abs(df.S21)/np.abs(df.S12)
        MSG = np.abs(df.S21)/np.abs(df.S12)
        df['MAG-MSG [dB]']=20*np.log10(np.choose(K>=1,[MSG,MAG]))
        
        # S-parameters
        for ii in ['11','12','21','22']:
            for comp,func in (('Re',np.real),('Im',np.imag)):
                df[f'{comp}S{ii}']=func(df[f'S{ii}'])
        return parsed_result

    def _rescale_vector(self,arr,col, meas_arr):
        return arr

    def _validate_parsed_result(self,parsed_result):
        # Overriding because this does freq num points instead of freq delta
        # TODO: implement this
        pass

    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type','log')
        kwargs['x_axis_type']=kwargs.get('x_axis_type','log')
        return super().generate_figures(*args,**kwargs)

    def _make_figures(self, meas_cds_c, meas_cds_l, sim_cds, layout_params, y_axis_type='log',x_axis_type='log'):
        assert y_axis_type=='log'
        assert x_axis_type=='log'
        figpow=bokeh.plotting.figure(tools=get_tools(),x_axis_type='log',**layout_params,tooltips=[('','$x Hz'),('','$name = $y')])
        figpow.circle(x='x',y='|h21| [dB]',source=meas_cds_c,color='blue',legend_label='h21',name='|h21| meas')
        figpow.multi_line(xs='x',ys='|h21| [dB]',source=meas_cds_l,color='blue',legend_label='h21',name='|h21| meas')
        figpow.multi_line(xs='x',ys='|h21| [dB]',source=sim_cds,color='red',legend_label='h21',name='|h21| sim')

        figpow.circle(x='x',y='U [dB]',source=meas_cds_c,color='green',legend_label='U',name='U meas')
        figpow.multi_line(xs='x',ys='U [dB]',source=meas_cds_l,color='green',legend_label='U',name='U meas')
        figpow.multi_line(xs='x',ys='U [dB]',source=sim_cds,color='orange',legend_label='U',name='U sim')

        figpow.circle(x='x',y='MAG-MSG [dB]',source=meas_cds_c,color='lightblue',legend_label='MAG/MSG',name='MAG/MSG meas')
        figpow.multi_line(xs='x',ys='MAG-MSG [dB]',source=meas_cds_l,color='lightblue',legend_label='MAG/MSG',name='MAG/MSG meas')
        figpow.multi_line(xs='x',ys='MAG-MSG [dB]',source=sim_cds,color='burlywood',legend_label='MAG/MSG',name='MAG/MSG sim')

        figpow.yaxis.axis_label='Power Gain [dB]'
        figpow.xaxis.axis_label='Frequency [Hz]'
        fig_legend_config(figpow)
        figpow.legend.location='top_right'

        figsmi=smith_chart(**layout_params)
        figsmi.circle(x='ReS11',y='ImS11',source=meas_cds_c,color='blue',legend_label='S11',line_width=2)
        figsmi.circle(x='ReS22',y='ImS22',source=meas_cds_c,color='green',legend_label='S22',line_width=2)
        figsmi.circle(x='ReS12',y='ImS12',source=meas_cds_c,color='lightblue',legend_label='S12',line_width=2)
        figsmi.multi_line(xs='ReS11',ys='ImS11',source=sim_cds,color='red',legend_label='S11',line_width=2)
        figsmi.multi_line(xs='ReS22',ys='ImS22',source=sim_cds,color='orange',legend_label='S22',line_width=2)
        figsmi.multi_line(xs='ReS12',ys='ImS12',source=sim_cds,color='burlywood',legend_label='S12',line_width=2)
        fig_legend_config(figsmi)
        figsmi.legend.location='top_right'

        return [figpow,figsmi]

class TemplateGroup:
    def __init__(self,**templates):
        self.temps=templates
        self._panes={}
    def set_paramset(self,params_by_template):
        if isinstance(params_by_template,ParamSet):
            for t in self.temps.values():
                t.set_paramset(params_by_template)
        else:
            assert set(params_by_template.keys())==set(self.temps.keys())
            for k,t in self.temps.items():
                t.set_paramset(params_by_template[k])
    def __getitem__(self,key):
        return self.temps[key]
    def items(self) -> list[tuple[str,SimTemplate]]:
        return self.temps.items()
    def __iter__(self):
        return iter(self.temps)
    def __len__(self):
        return len(self.temps)
    def only(self,*names):
        assert all(n in self.temps for n in names)
        return self.__class__(**{k:v for k,v in self.temps.items() if k in names})

    def get_figure_pane(self, meas_data=None, fig_layout_params={},vizid=None):
        figs=sum([t.generate_figures(
                            meas_data=meas_data.get(stname,None),
                            layout_params=fig_layout_params, vizid=vizid)
                       for stname,t in self.temps.items()],[])
        self._panes[vizid]=pn.pane.Bokeh(bokeh.layouts.gridplot([figs]))
        return self._panes[vizid]

    def update_figures(self, new_results, vizid=None):
        actually_did_update=False
        for stname in self.temps:
            if self.temps[stname].update_figures((new_results[stname] if new_results else None),vizid=vizid):
                actually_did_update=True
        if actually_did_update:
            logger.debug(f"(Not) pushing bokeh update to notebook for vizid {vizid}")
            # Apparently this isn't needed anymore?
            #pn.io.push_notebook(self._panes[vizid])

    def parsed_results_to_vector(self,parsed_results,roi, meas_parsed_results):
        return np.concatenate([simtemp.parsed_results_to_vector(parsed_results[k],roi[k],meas_parsed_results[k]) for k,simtemp in self.items()])
