import numpy as np
import bokeh
import bokeh.plotting
import pandas as pd
import bokeh.layouts
import panel as pn

from compyct.backends.backend import Netlister

from compyct.paramsets import ParamSet, spicenum_to_float, ParamPlace

class SimTemplate():
    def __init__(self, model_paramset=None):
        if model_paramset is not None:
            self.set_paramset(model_paramset)

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
        
    # def _get_instance_param_part(self):
    #     return ' '.join(f'{k}=instparam_{k}' for k in self.model_paramset
    #                     if self.model_paramset.get_place(k)==ParamPlace.INSTANCE)
            

class MultiSweepSimTemplate(SimTemplate):
    def __init__(self, *args,
                 outer_variable=None,inner_variable=None, ynames=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_variable=outer_variable
        self.inner_variable=inner_variable
        self.ynames=ynames
        self._sources={}
        
    def _parsed_result_to_cds_data(self,parsed_result):
        #assert len(self.ynames)==1
        #yname=self.ynames[0]
        
        data=dict(x=[],legend=[],**{f'y{i}':[] for i in range(len(self.ynames))})
        if parsed_result is None: parsed_result={}
        for key, df in parsed_result.items():
            data['x'].append(df[self.inner_variable].to_numpy())
            for i,yname in enumerate(self.ynames):
                data[f'y{i}'].append(df[yname].to_numpy())
            data['legend'].append(key)
        return data
        
        
    def _validate_parsed_result(self,parsed_result):
        raise NotImplementedError
    
    def generate_figures(self, meas_data=None,
                         layout_params={}, y_axis_type='linear', vizid=None):
    
        self._sources[vizid]\
               =[self._update_cds_with_parsed_result(cds=None,parsed_result=None)]

        if meas_data is not None:
            self._validate_parsed_result(meas_data)
        
        
        meas_cds=self._update_cds_with_parsed_result(cds=None,
                    parsed_result=meas_data,flattened=True)
        sim_cds=self._sources[vizid][0]

        num_ys=len([k for k in sim_cds.data if k.startswith('y')])
        if type(y_axis_type) is str: y_axis_type=[y_axis_type]*num_ys
        figs=[]
        for i in range(num_ys):
            fig=bokeh.plotting.figure(#x_range=self.vg_range,y_range=(1e-8,1),
                                      y_axis_type=y_axis_type[i],**layout_params)
            fig.circle(x='x',y=f'y{i}',source=meas_cds,legend_field='legend')
            fig.multi_line(xs='x',ys=f'y{i}',source=sim_cds,color='red')
    
            fig.yaxis.axis_label=self.ynames[i]#",".join(self.ynames)
            fig.legend.margin=0
            fig.legend.spacing=0
            fig.legend.padding=4
            fig.legend.label_text_font_size='8pt'
            fig.legend.label_height=10
            fig.legend.label_text_line_height=10
            fig.legend.glyph_height=10
            fig.legend.location='bottom_right'
            fig.xaxis.axis_label=self.inner_variable
            figs.append(fig)
        return figs
        
    def update_figures(self, parsed_result, vizid=None):
        self._validate_parsed_result(parsed_result)
        self._update_cds_with_parsed_result(
            cds=self._sources[vizid][0],parsed_result=parsed_result)

    def parsed_results_to_vector(self, parsed_results, roi):
        if roi is None: return np.array([])
        arr=[]
        for k, v in parsed_results.items():
            if k in roi:
                arr.append(parsed_results[roi[k]])
        return self._rescale_vector(np.array(arr))
        
    def _rescale_vector(self,arr):
        raise NotImplementedError

class DCIdVdTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27, yscale='linear',
                 vg_values=[0,.6,1.2,1.8], vd_range=(0,.1,1.8), **kwargs):
        self.yscale=yscale
        super().__init__(outer_variable='VG', inner_variable='VD',
                         ynames=['ID/W [uA/um]' if pol=='n' else '-ID/W [uA/um]'],
                         *args, **kwargs)
        self.temp=temp
        
        num_vd=(vd_range[2]-vd_range[0])/vd_range[1]
        assert abs(num_vd-int(num_vd))<1e-3, f"Make sure the IdVd range gives even steps {str(vd_range)}"

        self.pol=pol
        self.vg_values=vg_values
        self.vd_range=vd_range
    
    def get_schematic_listing(self,netlister:Netlister):
            #netlister.nstr_param(params={'vg':0,'vd':0})+\
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_modeled_xtor("inst",netd='netd',netg='netg',
                                        nets=netlister.GND,netb=netlister.GND,dt=None),
            netlister.nstr_VDC("D",netp='netd',netm=netlister.GND,dc=0),
            netlister.nstr_VDC("G",netp='netg',netm=netlister.GND,dc=0)]

    def get_analysis_listing(self,netlister:Netlister):
        analysis_listing=[]
        analysis_listing.append(netlister.nstr_temp(temp=self.temp))
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

    def _rescale_vector(self,arr):
        return arr
        
    def _validate_parsed_result(self,parsed_result):
        assert parsed_result.keys()==set(self.vg_values),\
            f"Template requests VG {self.vg_values}, but results are {list(parsed_result.keys())}"
        # TODO: check spacings 
        
class DCIdVgTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27,
                 vd_values=[.05,1.8], vg_range=(0,.03,1.8), plot_gm=True, **kwargs):
        ynames=['ID/W [uA/um]' if pol=='n' else '-ID/W [uA/um]']
        if plot_gm: ynames+=['GM/W [uS/um]']
        super().__init__(outer_variable='VD', inner_variable='VG',
                         ynames=ynames,
                         *args, **kwargs)
        self.temp=temp
        
        num_vg=(vg_range[2]-vg_range[0])/vg_range[1]
        assert abs(num_vg-int(num_vg))<1e-3, f"Make sure the IdVg range gives even steps {str(vg_range)}"
        
        self.pol=pol
        self.vg_range=vg_range
        self.vd_values=vd_values

    def get_schematic_listing(self,netlister:Netlister):
            #netlister.nstr_param(params={'vg':0,'vd':0})+\
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_modeled_xtor("inst",netd='netd',netg='netg',
                                        nets=netlister.GND,netb=netlister.GND,dt=None),
            netlister.nstr_VDC("D",netp='netd',netm=netlister.GND,dc=0),
            netlister.nstr_VDC("G",netp='netg',netm=netlister.GND,dc=0)]

    def get_analysis_listing(self,netlister:Netlister):
        analysis_listing=[]
        analysis_listing.append(netlister.nstr_temp(temp=self.temp))
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
                    sgn=-1 if self.pol=='p' else 1
                    sgnstr="-" if self.pol=='p' else ''
                    df[f'{sgnstr}ID/W [uA/um]']=-sgn*df['vd#p']/\
                            self.model_paramset.get_total_device_width()
                    df[f'{sgnstr}IG/W [uA/um]']=-sgn*df['vg#p']/\
                            self.model_paramset.get_total_device_width()
                    parsed_result[vd]=df.rename(columns=\
                                {'netd':'VD','netg':'VG'})\
                            [['VD','VG',f'{sgnstr}ID/W [uA/um]',f'{sgnstr}IG/W [uA/um]']]
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

    def _rescale_vector(self,arr):
        return np.log10(np.abs(arr))
        
    def _validate_parsed_result(self,parsed_result):
        assert parsed_result.keys()==set(self.vd_values),\
            f"Template requests VD {self.vd_values}, but results are {list(parsed_result.keys())}"
        # TODO: check spacings 

class JointTemplate(SimTemplate):
    def __init__(self,subtemplates:dict, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.subtemplates=subtemplates

    def postparse_return(self,parsed_result):
        return {k:t.postparse_return(parsed_result[k])for k,t in self.subtemplates.items()}

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

    def set_paramset(self,model_paramset):
        super().set_paramset(model_paramset)
        self._dcidvg.set_paramset(model_paramset)
        self._dcidvd.set_paramset(model_paramset)

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
        
    def update_figures(self, parsed_result, vizid=None):
        self._dcidvg.update_figures(parsed_result['IdVg'],vizid=vizid)
        self._dcidvd.update_figures(parsed_result['IdVd'],vizid=vizid)

    def parsed_results_to_vector(self, parsed_results, roi):
        return np.concatenate([
            self._dcidvg.parsed_results_to_vector(parsed_results['IdVg'],roi['IdVg']),
            self._dcidvd.parsed_results_to_vector(parsed_results['IdVd'],roi['IdVd'])])

class CVTemplate(MultiSweepSimTemplate):

    def __init__(self, *args,
                 vg_range=(0,.03,1.8), freq='1meg', **kwargs):
        super().__init__(outer_variable=None, inner_variable='VG',
                         ynames=['Cgg [fF/um]'],
                         *args, **kwargs)
        
        num_vg=(vg_range[2]-vg_range[0])/vg_range[1]
        assert abs(num_vg-int(num_vg))<1e-3, f"Make sure the CV VG range gives even steps {str(vg_range)}"
        
        self.vg_range=vg_range
        self.freq=freq
        try:
            spicenum_to_float(freq)
        except:
            raise Exception(f"Invalid frequency: {freq}")

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
        parsed_result={0:df[['VG','Cgg [fF/um]']]}
        return parsed_result
        
    def _validate_parsed_result(self,parsed_result):
        pass
        #assert parsed_result.keys()==set([0])
        # TODO: check spacings 

class IdealPulsedIdVdTemplate(MultiSweepSimTemplate):

    def __init__(self, *args,
                 vg_values=[0,.6,1.2,1.8], vd_range=(0,.1,1.8),
                 pulse_width='1u',rise_time='100n',
                 vgq=0, vdq=0,
                 **kwargs):
        super().__init__(outer_variable='VG', inner_variable='VD',
                         ynames=['ID/W [uA/um]'],
                         *args, **kwargs)
        
        num_vd=(vd_range[2]-vd_range[0])/vd_range[1]
        assert abs(num_vd-int(num_vd))<1e-3,\
            f"Make sure the IdVd range gives even steps {str(vd_range)}"
        
        self.vg_values=vg_values
        self.vd_range=vd_range
        self.pulse_width=pulse_width
        self.rise_time=rise_time
        self.vgq=vgq
        self.vdq=vdq

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

    def _rescale_vector(self,arr):
        return arr
        
    def _validate_parsed_result(self,parsed_result):
        pass
        #assert parsed_result.keys()==set([0])
        # TODO: check EVERYTHING

class TemplateGroup:
    def __init__(self,**templates):
        self.temps=templates
        self._panes={}
    def set_paramset(self,params_by_template):
        assert set(params_by_template.keys())==set(self.temps.keys())
        for k,t in self.temps.items():
            t.set_paramset(params_by_template[k])
    def __getitem__(self,key):
        return self.temps[key]
    def items(self):
        return self.temps.items()
    def __iter__(self):
        return iter(self.temps)
    def __len__(self):
        return len(self.temps)
    #def parsed_results_to_vector(self,parsed_results):

    def get_figure_pane(self, meas_data=None, fig_layout_params={},vizid=None):
        figs=sum([t.generate_figures(
                            meas_data=meas_data.get(stname,None),
                            layout_params=fig_layout_params)
                       for stname,t in self.temps.items()],[])
        self._panes[vizid]=pn.pane.Bokeh(bokeh.layouts.gridplot([figs]))
        return self._panes[vizid]

    def update_figures(self, new_results, vizid=None):
        for stname in self.temps:
            self.temps[stname].update_figures(new_results[stname])
        pn.io.push_notebook(self._panes[vizid])