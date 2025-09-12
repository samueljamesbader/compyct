from contextlib import contextmanager
from itertools import product
from collections import UserDict

import numpy as np
import bokeh
import bokeh.plotting
import pandas as pd
import bokeh.layouts
import panel as pn
from bokeh.models import HoverTool, CustomJSHover
from bokeh.palettes import TolRainbow, Category10_10
#from scipy.integrate import cumtrapz
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d

from bokeh_transform_utils.transforms import MultiAbsTransform, multi_abs_transform, abs_transform

from bokeh_smith import smith_chart
from compyct import logger
from compyct.backends.backend import Netlister
from compyct.gui import fig_legend_config, get_tools

from compyct.paramsets import ParamPatch, spicenum_to_float, ParamPlace
from compyct.util import s2y, s2z, form_multisweep

class SimTemplate():
    def __init__(self, patch=None, internals_to_save=[], title=None):
        if patch is not None:
            self.set_patch(patch)
        self.internals_to_save=internals_to_save
        self.title=title

    def set_patch(self,patch):
        assert isinstance(patch,ParamPatch)
        self._patch=patch.copy() if patch else None
        
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
                                 if k not in ['legend','color','outervariable','additionalinfo'] and hasattr(v[0],'__len__')}
            swpkey=list(flattened_data.keys())[0]
            flattened_data.update({k:[v[i] for i in range(len(data[swpkey])) for _ in data[swpkey][i]]
                                   for k,v in data.items()
                                       if k in ['legend','color','outervariable','additionalinfo'] or (not hasattr(v[0],'__len__'))})
            # Only works for rectangular arrays
            #npts=len(data[list(flattened_data.keys())[0]][0])
            #flattened_data.update({k:np.repeat(v,npts) for k,v in data.items()
                                 #if k=='legend' or (not hasattr(v[0],'__len__'))})
            data=flattened_data
        if cds is None:
            cds=bokeh.models.ColumnDataSource(data)
        else:
            cds.data=data
        return cds

    def _validated(self, parsed_result):
        raise NotImplementedError

    @contextmanager
    def tentative_base_deltas(self,params) -> ParamPatch:
        assert isinstance(params, ParamPatch)
        bk=self._patch.copy()
        try:
            yield self._patch.update_inplace_and_return_base_changes(params)
        except:
            self._patch.update_inplace_and_return_changes(bk)
            raise

    def apply_patch(self,params):
        self._patch.update_inplace_and_return_changes(params)


    # def _get_instance_param_part(self):
    #     return ' '.join(f'{k}=instparam_{k}' for k in self._patch
    #                     if self._patch.get_place(k)==ParamPlace.INSTANCE)
    def to_csv(self, param, filename):
        raise NotImplementedError


class MultiSweepSimTemplate(SimTemplate):
    def __init__(self, *args,
                 outer_variable=None, inner_variable=None,
                 outer_values=None, inner_range=None,
                 ynames=[], directions=['f','r'], **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_variable=outer_variable
        self.inner_variable=inner_variable
        self.outer_values=outer_values
        self.inner_range=inner_range
        self.ynames=ynames
        self.directions=directions
        self._sources={}
        self._fig_is_clear=True

        #fnpt=(inner_range[2]-inner_range[0])/inner_range[1]
        #if self.__class__ is CVTemplate:
        #    import pdb; pdb.set_trace()
        #if np.abs(np.round(fnpt)-fnpt)<1e-3:
        #    if inner_range[1]*np.round(fnpt)>(inner_range[2]-inner_range[0]):
        #        logger.debug(f"Tuning step for {self.__class__.__name__}")
        #        inner_range[1]=(inner_range[2]-inner_range[0])/np.round(fnpt)*.99999

    #def __copy__(self):
    #    return copy(super()

    @property
    def xname(self):
        if hasattr(self,'_xname'):
            return self._xname
        else:
            return self.inner_variable
        
    def _parsed_result_to_cds_data(self,parsed_result):
        #assert len(self.ynames)==1
        #yname=self.ynames[0]
        
        data=dict(x=[],legend=[],color=[],additionalinfo=[],outervariable=[],**{yname:[] for yname in self.ynames})
        if parsed_result is None: parsed_result={}
        #for key, df in parsed_result.items():
        k0s=list(sorted(set([k[0] for k in self._required_keys()])))
        colors=dict(zip(k0s,(TolRainbow[max(len(k0s),3)] if len(k0s)>3 else Category10_10)))
        if len(parsed_result):
            for key in self._required_keys():
                try:
                    df=parsed_result[key]
                except KeyError as e:
                    import pdb; pdb.set_trace()
                    raise e
                data['x'].append(df[self.xname].to_numpy())
                for yname in set(self.ynames):
                    data[yname].append(df[yname].to_numpy())
                data['legend'].append(' '.join(str(ki) for ki in key))
                data['outervariable'].append(key[0])
                data['additionalinfo'].append(','.join(str(ki) for ki in key[1:]))
                data['color'].append(colors[key[0]])
        return data
        
        
    def generate_figures(self, meas_data=None,
                         layout_params={}, y_axis_type='linear', x_axis_type='linear', override_line_color=None,
                         vizid=None):
    
        self._sources[vizid]\
               =[self._update_cds_with_parsed_result(cds=None,parsed_result=None)]

        if meas_data is not None:
            self._validated(meas_data)

        meas_cds_c=self._update_cds_with_parsed_result(cds=None,
                    parsed_result=meas_data,flattened=True)
        meas_cds_l=self._update_cds_with_parsed_result(cds=None,
                    parsed_result=meas_data,flattened=False)
        sim_cds=self._sources[vizid][0]

        return self._make_figures(meas_cds_c=meas_cds_c, meas_cds_l=meas_cds_l, sim_cds=sim_cds,
                                  layout_params=layout_params, override_line_color=override_line_color,
                                  y_axis_type=y_axis_type, x_axis_type=x_axis_type)

    def _make_figures(self, meas_cds_c, meas_cds_l, sim_cds, layout_params, y_axis_type='linear',x_axis_type='linear',override_line_color=None):
        num_ys=len(self.ynames)
        if type(y_axis_type) is str: y_axis_type=[y_axis_type]*num_ys
        if type(x_axis_type) is str: x_axis_type=[x_axis_type]*num_ys
        figs=[]
        for i in range(num_ys):
            TOOLTIPS=[
                (f"{self.xname}",f"@x"),
                #(f"{self.outer_variable}",f"@{self.outer_variable}"),
                (f"{self.ynames[i]}",f"@{{{self.ynames[i]}}}"),
            ]
            fig=bokeh.plotting.figure(tools=get_tools(),#x_range=self.vg_range,y_range=(1e-8,1),
                                      #tooltips=TOOLTIPS,
                                      y_axis_type=y_axis_type[i],x_axis_type=x_axis_type[i],**layout_params)
            if self.title is not None: fig.title=self.title
            mtrans=multi_abs_transform if y_axis_type[i]=='log' else (lambda x: x)
            strans=abs_transform if y_axis_type[i]=='log' else (lambda x: x)

            # blue-red scheme
            #fig.scatter(x='x',y=self.ynames[i],source=meas_cds_c,legend_field='legend')
            #fig.multi_line(xs='x',ys=self.ynames[i],source=meas_cds_l)
            #fig.multi_line(xs='x',ys=self.ynames[i],source=sim_cds,color='red')
            # technicolor
            circ_rend=fig.scatter(x='x',y=strans(self.ynames[i]),source=meas_cds_c,legend_field='legend',name='scatter',color='color')
            mult_rend=fig.multi_line(xs='x',ys=mtrans(self.ynames[i]),source=sim_cds,
                                 color=(override_line_color or 'color'),**({'line_width':2} if override_line_color else {}))
            # https://stackoverflow.com/a/68536069
            num=1
            t = f"""
                <div @x{{custom}}>
                    <b>{self.xname}: </b> @x <br/>
                    <b>{self.outer_variable}: </b> @outervariable <br/>
                    <b>{self.ynames[i]}: </b> @{{{self.ynames[i]}}} <br/>
                    <b>Details: </b> @additionalinfo
                </div>
                """
            f = CustomJSHover(code=f"""
                special_vars.indices = special_vars.indices.slice(0,{num})
                return special_vars.indices.includes(special_vars.index) ? " " : " hidden "
                """)
            #fig.select(type=HoverTool).renderers=[circ_rend]
            #fig.select(type=HoverTool).formatters={'@x': f}
            # Should be able to include visible=False when I upgrade to bokeh=3.4.0
            fig.add_tools(HoverTool(tooltips=t,renderers=[circ_rend],formatters={'@x':f}))#,visible=False))


            fig.yaxis.axis_label=self.ynames[i]#",".join(self.ynames)
            fig.xaxis.axis_label=self.xname
            #fig_legend_config(fig)
            fig.legend.visible=False
            figs.append(fig)
        return figs

    def update_figures(self, parsed_result, vizid=None):
        if parsed_result is None and self._fig_is_clear:
            return False
        logger.debug(f"Updating figure {self.__class__.__name__}")
        self._validated(parsed_result)
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
                if k is None:
                    ks=product(self.outer_values,self.directions)
                else:
                    ks=[k]
                for k in ks:
                    inds,col=sl
                    arr.append(self._rescale_vector(
                        parsed_results[k].reset_index().loc[sl],
                        col,
                        meas_parsed_results[k].reset_index().loc[sl]))
        return np.concatenate(arr)
        
    def _rescale_vector(self, arr, col, meas_arr):
        raise NotImplementedError

    def _required_keys(self):
        return set(product(self.outer_values,self.directions))

    def _validated(self, parsed_result):
        if parsed_result is None: return parsed_result
        required_keys=self._required_keys()
        if self.outer_variable is not None:
            assert all(k in parsed_result.keys() for k in required_keys), \
                f"{self.__class__.__name__} requests {self.outer_variable}"\
                f" {list(required_keys)},"\
                f" but results are {list(parsed_result.keys())}"
        if self.inner_range is not None:
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
        return parsed_result

    def to_merged_table(self,result):
        dfs=[]
        for (ovv,d),df in result.items():
            #if d!='f': continue
            if self.outer_variable in df:
                df=df.drop(columns=[self.outer_variable])
            df=df.set_index([self.inner_variable])
            dfs.append(df.rename(columns=(lambda c: f'{c} @{self.outer_variable}={ovv} {d}')))

        # Make sure the indices are the same before merging (fillna trick to equate nan's)
        if not all(np.all(dfs[0].index.fillna(np.inf)==df.index.fillna(np.inf)) for df in dfs):
            import pdb; pdb.set_trace()
            raise Exception("Unequal indices, how to merge?")
        return pd.concat(dfs,axis=1)

    def to_csv(self,result,filename=None):
        return self.to_merged_table(result).to_csv(filename)

class DCIdVdTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27, probe_r=0,
                 vg_values=[0,.6,1.2,1.8], vd_range=(0,.1,1.8), plot_RD=True, **kwargs):
        super().__init__(outer_variable='VG', inner_variable='VD',
                         outer_values=vg_values, inner_range=(vd_range if type(vd_range) is not dict else None),
                         ynames=[('ID/W [uA/um]' if pol=='n' else '-ID/W [uA/um]'),
                                 ('ID/W [uA/um]' if pol=='n' else '-ID/W [uA/um]'),
                                 ('IG/W [uA/um]' if pol=='n' else '-IG/W [uA/um]'),
                                 *(['RD*W [Ohm*um]'] if plot_RD else [])
                                 ],
                         *args, **kwargs)
        self.temp=temp
        self.probe_r=probe_r
        self._plot_RD=plot_RD

        if type(vd_range) is not dict:
            vd_range={(vg,d):vd_range for vg in vg_values for d in self.directions}
        for vg,vdr in vd_range.items():
            num_vd=(vdr[2]-vdr[0])/vdr[1]
            assert abs(num_vd-round(num_vd))<1e-3, f"Make sure the IdVd range gives even steps {str(vdr)} @ VG={vg}"
        self.vd_range=vd_range

        self.pol=pol

    @property
    def vg_values(self): return self.outer_values

    def get_schematic_listing(self,netlister:Netlister):
        return DCIdVgTemplate.get_schematic_listing(self,netlister=netlister)

    def get_analysis_listing(self,netlister:Netlister):
        analysis_listing=[]
        d=self.directions[0]
        for i_vg,vg in enumerate(self.vg_values,start=1):
            analysis_listing.append(netlister.astr_altervdc('G',vg))
            analysis_listing.append(netlister.astr_sweepvdc('D',name=f'idvd_vgnum{i_vg}',
                start=self.vd_range[(vg,d)][0],step=self.vd_range[(vg,d)][1],stop=self.vd_range[(vg,d)][2]))
        return analysis_listing

    def parse_return(self,result):
        parsed_result={}
        for i_vg,vg in enumerate(self.vg_values,start=1):
            for key in result:
                if key==f'idvd_vgnum{i_vg}':
                    df=result[key].copy()
                    sgn=-1 if self.pol=='p' else 1
                    sgnstr="-" if self.pol=='p' else ''
                    df[f'{sgnstr}ID/W [uA/um]']=-sgn*df['vd#p']/\
                            self._patch.get_total_device_width()
                    df[f'{sgnstr}IG/W [uA/um]']=-sgn*df['vg#p']/\
                            self._patch.get_total_device_width()
                    parsed_result[(vg,'f')]=df.rename(columns=\
                                {'netd':'VD','netg':'VG'})\
                            [['VD','VG',f'{sgnstr}ID/W [uA/um]',f'{sgnstr}IG/W [uA/um]']]
                    # DC sim doesn't distinguish f/r
                    if 'r' in self.directions:
                        parsed_result[(vg,'r')]=parsed_result[(vg,'f')]
        return parsed_result

    def postparse_return(self, parsed_result):
        sgn = -1 if self.pol == 'p' else 1
        sgnstr = "-" if self.pol == 'p' else ''
        for (vg,d), df in parsed_result.items():
            with np.errstate(divide='ignore'):
                df[f'RD*W [Ohm*um]'] = 1/np.gradient(1e-6 * sgn * df[f'{sgnstr}ID/W [uA/um]'], self.vd_range[(vg,d)][1])
        return parsed_result

    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type',['linear','log','log']+(['linear'] if self._plot_RD else []))
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self, arr, col, meas_arr):
        if col[0]=='I':
            return 5*arr/np.max(meas_arr)

class DCIdVgTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27,
                 vd_values=[.05,1.8], vg_range=(0,.03,1.8), plot_gm=True, probe_r=0, **kwargs):
        ynames=['ID/W [uA/um]' if pol=='n' else '-ID/W [uA/um]']*2
        if plot_gm: ynames+=['GM/W [uS/um]']
        super().__init__(outer_variable='VD', inner_variable='VG',
                         outer_values=vd_values, inner_range=(vg_range if type(vg_range) is not dict else None),
                         ynames=ynames,
                         *args, **kwargs)
        self.temp=temp
        self.probe_r=probe_r

        if type(vg_range) is not dict:
            vg_range={(vd,d):vg_range for vd in vd_values for d in self.directions}
        for vd,vgr in vg_range.items():
            num_vg=(vgr[2]-vgr[0])/vgr[1]
            assert abs(num_vg-round(num_vg))<1e-3, f"Make sure the IdVg range gives even steps {str(vgr)} @ VD={vd}"
        self.vg_range=vg_range

        self.pol=pol

    @property
    def vd_values(self): return self.outer_values


    def get_schematic_listing(self,netlister:Netlister, dcvg=0, dcvd=0):
            #netlister.nstr_param(params={'vg':0,'vd':0})+\
        gnded=[t for t in self._patch.terminals if t not in ['d','g','t','dt']]
        if self.probe_r==0:
            netmap=dict(**{'d':'netd','g':'netg'},**{k:netlister.GND for k in gnded})
        else:
            netmap=dict(**{'d':'netdw','g':'netg'},**{k:'netsw' for k in gnded})
        nl= [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst",netmap=netmap,
                                        internals_to_save=self.internals_to_save),
            netlister.nstr_VDC("D",netp='netd',netm=netlister.GND,dc=dcvd),
            netlister.nstr_VDC("G",netp='netg',netm=netlister.GND,dc=dcvg)]
        if self.probe_r!=0:
            nl.append(netlister.nstr_res('probes','netsw',netlister.GND,self.probe_r))
            nl.append(netlister.nstr_res('probed','netdw',       'netd',self.probe_r))
        return nl

    def get_analysis_listing(self,netlister:Netlister):
        analysis_listing=[]
        d=self.directions[0]
        for i_vd,vd in enumerate(self.vd_values,start=1):
            analysis_listing.append(netlister.astr_altervdc('D',vd))
            analysis_listing.append(netlister.astr_sweepvdc('G',name=f'idvg_vdnum{i_vd}',
                start=self.vg_range[(vd,d)][0],step=self.vg_range[(vd,d)][1],stop=self.vg_range[(vd,d)][2]))
        return analysis_listing
        
    def parse_return(self,result):
        parsed_result={}
        #print(f"Result keys: {list(result.keys())}")
        for i_vd,vd in enumerate(self.vd_values,start=1):
            for key in result:
                if key==f'idvg_vdnum{i_vd}':
                    #import pdb; pdb.set_trace()
                    df=result[key].copy()
                    #print(vd,"gmi:",list(df['@ninst[gmi]']))
                    sgn=-1 if self.pol=='p' else 1
                    sgnstr="-" if self.pol=='p' else ''
                    df[f'{sgnstr}ID/W [uA/um]']=-sgn*df['vd#p']/\
                            self._patch.get_total_device_width()
                    df[f'{sgnstr}IG/W [uA/um]']=-sgn*df['vg#p']/\
                            self._patch.get_total_device_width()
                    # TODO: reinstate column restriction (removed for device internals testing)
                    parsed_result[(vd,'f')]=df.rename(columns=\
                                {'netd':'VD','netg':'VG'})\
                            [['VD','VG',f'{sgnstr}ID/W [uA/um]',f'{sgnstr}IG/W [uA/um]']]
                    # DC sim doesn't distinguish f/r
                    if 'r' in self.directions:
                        parsed_result[(vd,'r')]=parsed_result[(vd,'f')]
        return parsed_result

    def postparse_return(self,parsed_result):
        sgn=-1 if self.pol=='p' else 1
        sgnstr="-" if self.pol=='p' else ''
        for (vd,d),df in parsed_result.items():
            df[f'GM/W [uS/um]']=np.gradient(sgn*df[f'{sgnstr}ID/W [uA/um]'],self.vg_range[(vd,d)][1])
        return parsed_result
        
    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=[*kwargs.get('y_axis_type',['log','linear']),'linear']
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self, arr, col, meas_arr):
        if col[0]=='I':
            return np.log10(np.abs(arr)+1e-14)
        else:
            return 5*arr/np.max(meas_arr)


class DCIdVgVbTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27,
                 vd_value=1.8,vb_values=[0, 1.8], vg_range=(0, .03, 1.8), plot_gm=True, probe_r=0, **kwargs):
        ynames = ['ID/W [uA/um]' if pol == 'n' else '-ID/W [uA/um]'] * 2
        if plot_gm: ynames += ['GM/W [uS/um]']
        super().__init__(outer_variable='VB', inner_variable='VG',
                         outer_values=vb_values, inner_range=(vg_range if type(vg_range) is not dict else None),
                         ynames=ynames,
                         *args, **kwargs)
        self.temp = temp
        self.probe_r = probe_r

        if type(vg_range) is not dict:
            vg_range = {(vb, d): vg_range for vb in vb_values for d in self.directions}
        for vb, vgr in vg_range.items():
            num_vg = (vgr[2] - vgr[0]) / vgr[1]
            assert abs(num_vg - round(num_vg)) < 1e-3, f"Make sure the IdVg range gives even steps {str(vgr)} @ VB={vb}"
        self.vg_range = vg_range

        self.vd_value=vd_value
        self.pol = pol

    @property
    def vb_values(self):
        return self.outer_values

    def get_schematic_listing(self, netlister: Netlister, dcvg=0, dcvd=0):
        # netlister.nstr_param(params={'vg':0,'vd':0})+\
        gnded = [t for t in self._patch.terminals if t not in ['b','d', 'g', 't', 'dt']]
        if self.probe_r == 0:
            netmap = dict(**{'d': 'netd', 'g': 'netg', 'b':'netb'}, **{k: netlister.GND for k in gnded})
        else:
            netmap = dict(**{'d': 'netdw', 'g': 'netg', 'b':'netb'}, **{k: 'netsw' for k in gnded})
        nl = [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst", netmap=netmap,
                                        internals_to_save=self.internals_to_save),
            netlister.nstr_VDC("D", netp='netd', netm=netlister.GND, dc=self.vd_value),
            netlister.nstr_VDC("G", netp='netg', netm=netlister.GND, dc=dcvg),
            netlister.nstr_VDC("B", netp='netb', netm=netlister.GND, dc=self.vb_values[0])
        ]
        if self.probe_r != 0:
            nl.append(netlister.nstr_res('probes', 'netsw', netlister.GND, self.probe_r))
            nl.append(netlister.nstr_res('probed', 'netdw', 'netd', self.probe_r))
        return nl

    def get_analysis_listing(self, netlister: Netlister):
        analysis_listing = []
        d = self.directions[0]
        for i_vb, vb in enumerate(self.vb_values, start=1):
            analysis_listing.append(netlister.astr_altervdc('B', vb))
            analysis_listing.append(netlister.astr_sweepvdc('G', name=f'idvg_vbnum{i_vb}',
                                                            start=self.vg_range[(vb, d)][0],
                                                            step=self.vg_range[(vb, d)][1],
                                                            stop=self.vg_range[(vb, d)][2]))
        return analysis_listing

    def parse_return(self, result):
        parsed_result = {}
        # print(f"Result keys: {list(result.keys())}")
        for i_vb, vb in enumerate(self.vb_values, start=1):
            for key in result:
                if key == f'idvg_vbnum{i_vb}':
                    # import pdb; pdb.set_trace()
                    df = result[key].copy()
                    # print(vd,"gmi:",list(df['@ninst[gmi]']))
                    sgn = -1 if self.pol == 'p' else 1
                    sgnstr = "-" if self.pol == 'p' else ''
                    df[f'{sgnstr}ID/W [uA/um]'] = -sgn * df['vd#p'] / \
                                                  self._patch.get_total_device_width()
                    df[f'{sgnstr}IG/W [uA/um]'] = -sgn * df['vg#p'] / \
                                                  self._patch.get_total_device_width()
                    # TODO: reinstate column restriction (removed for device internals testing)
                    parsed_result[(vb, 'f')] = df.rename(columns= \
                                                             {'netd': 'VD', 'netg': 'VG', 'netb': 'VB'}) \
                        [['VD', 'VG', 'VB', f'{sgnstr}ID/W [uA/um]', f'{sgnstr}IG/W [uA/um]']]
                    # DC sim doesn't distinguish f/r
                    if 'r' in self.directions:
                        parsed_result[(vb, 'r')] = parsed_result[(vb, 'f')]
        return parsed_result

    def postparse_return(self, parsed_result):
        sgn = -1 if self.pol == 'p' else 1
        sgnstr = "-" if self.pol == 'p' else ''
        for (vb, d), df in parsed_result.items():
            df[f'GM/W [uS/um]'] = np.gradient(sgn * df[f'{sgnstr}ID/W [uA/um]'], self.vg_range[(vb, d)][1])
        return parsed_result

    def generate_figures(self, *args, **kwargs):
        kwargs['y_axis_type'] = [*kwargs.get('y_axis_type', ['log', 'linear']), 'linear']
        return super().generate_figures(*args, **kwargs)

    def _rescale_vector(self, arr, col, meas_arr):
        if col[0] == 'I':
            return np.log10(np.abs(arr) + 1e-14)
        else:
            return 5 * arr / np.max(meas_arr)

class DCIbVbTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, pol='n', temp=27,
                 vgvdvs_values=0, vb_range=(-2.6, .03, 2.6),probe_r=0, **kwargs):
        ynames = ['|IB/W| [uA/um]','|ID/W| [uA/um]','|IS/W| [uA/um]']
        super().__init__(outer_variable='VGVDVS', inner_variable='VB',
                         outer_values=vgvdvs_values, inner_range=(vb_range if type(vb_range) is not dict else None),
                         ynames=ynames,
                         *args, **kwargs)
        self.temp = temp
        self.probe_r = probe_r

        if type(vb_range) is not dict:
            vb_range = {(vgvdvs, d): vb_range for vgvdvs in vgvdvs_values for d in self.directions}
        for vgvdvs, vbr in vb_range.items():
            num_vg = (vbr[2] - vbr[0]) / vbr[1]
            assert abs(num_vg - round(num_vg)) < 1e-3, f"Make sure the IdVg range gives even steps {str(vbr)} @ VGVDVS={vgvdvs}"
        self.vb_range = vb_range

        self.pol = pol

    @property
    def vgvdvs_values(self):
        return self.outer_values

    def get_schematic_listing(self, netlister: Netlister):
        gnded = [t for t in self._patch.terminals if t not in ['b','d', 'g', 's', 't', 'dt']]
        if self.probe_r == 0:
            netmap = dict(**{'d': 'netd', 'g': 'netg', 'b':'netb', 's':'nets'}, **{k: netlister.GND for k in gnded})
        else:
            raise Exception("Probably wrong handling of source here")
            netmap = dict(**{'d': 'netdw', 'g': 'netg', 'b':'netb'}, **{k: 'netsw' for k in gnded})
        nl = [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst", netmap=netmap,
                                        internals_to_save=self.internals_to_save),
            netlister.nstr_VDC("D", netp='netd', netm=netlister.GND, dc=self.vgvdvs_values[0][1]),
            netlister.nstr_VDC("G", netp='netg', netm=netlister.GND, dc=self.vgvdvs_values[0][0]),
            netlister.nstr_VDC("S", netp='nets', netm=netlister.GND, dc=self.vgvdvs_values[0][2]),
            netlister.nstr_VDC("B", netp='netb', netm=netlister.GND, dc=0)
        ]
        if self.probe_r != 0:
            nl.append(netlister.nstr_res('probes', 'netsw', netlister.GND, self.probe_r))
            nl.append(netlister.nstr_res('probed', 'netdw', 'netd', self.probe_r))
        return nl

    def get_analysis_listing(self, netlister: Netlister):
        analysis_listing = []
        d = self.directions[0]
        for i_vgvdvs, (vg,vd,vs) in enumerate(self.vgvdvs_values, start=1):
            analysis_listing.append(netlister.astr_altervdc('G',vg ))
            analysis_listing.append(netlister.astr_altervdc('D',vd ))
            analysis_listing.append(netlister.astr_altervdc('S',vs ))
            analysis_listing.append(netlister.astr_sweepvdc('B', name=f'idvb_vnum{i_vgvdvs}',
                                                            start=self.vb_range[((vg,vd,vs), d)][0],
                                                            step=self.vb_range[((vg,vd,vs), d)][1],
                                                            stop=self.vb_range[((vg,vd,vs), d)][2]))
        return analysis_listing

    def parse_return(self, result):
        parsed_result = {}
        # print(f"Result keys: {list(result.keys())}")
        for i_vgvdvs, vgvdvs in enumerate(self.vgvdvs_values, start=1):
            for key in result:
                if key == f'idvb_vnum{i_vgvdvs}':
                    df = result[key].copy()
                    df[f'|ID/W| [uA/um]'] = np.abs(df['vd#p']) / \
                                                  self._patch.get_total_device_width()
                    df[f'|IB/W| [uA/um]'] = np.abs(df['vb#p']) / \
                                                  self._patch.get_total_device_width()
                    df[f'|IS/W| [uA/um]'] = np.abs(df['vs#p']) / \
                                                  self._patch.get_total_device_width()
                    parsed_result[(vgvdvs, 'f')] = df.rename(columns= \
                                                    {'netd': 'VD', 'netg': 'VG', 'netb': 'VB', 'nets':'VS'}) \
                        [['VD', 'VG', 'VB', f'|ID/W| [uA/um]', f'|IB/W| [uA/um]', f'|IS/W| [uA/um]']]
                    # DC sim doesn't distinguish f/r
                    if 'r' in self.directions:
                        parsed_result[(vgvdvs, 'r')] = parsed_result[(vgvdvs, 'f')]
        return parsed_result

    def generate_figures(self, *args, **kwargs):
        kwargs['y_axis_type'] = 'log'#[*kwargs.get('y_axis_type', ['log', 'linear']), 'linear']
        lp=kwargs['layout_params']=kwargs.get('layout_params',{}).copy()
        lp['y_range']=(1e-12,1e4)
        return super().generate_figures(*args, **kwargs)


class JointTemplate(SimTemplate):
    def __init__(self,subtemplates:dict, *args, patch:ParamPatch=None, **kwargs):
        self.subtemplates=subtemplates
        super().__init__(patch=patch,*args,**kwargs)

    def set_patch(self,patch):
        super().set_patch(patch)
        for st in self.subtemplates.values():
            st._patch=self._patch

    def postparse_return(self,parsed_result):
        return {k:t.postparse_return(parsed_result[k])for k,t in self.subtemplates.items()}

    def _validated(self, parsed_result):
        return {k:t._validated(parsed_result[k]) for k,t in self.subtemplates.items()}

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
    def __init__(self, *args,pol='n', temp=27,
                 idvd_vg_values=[0,.6,1.2,1.8], idvd_vd_range=(0,.1,1.8),
                 idvg_vd_values=[.05,1.8], idvg_vg_range=(0,.03,1.8), **kwargs):
        self.temp=temp
        self._dcidvg=DCIdVgTemplate(*args, **kwargs,pol=pol,temp=temp,
                                    vd_values=idvg_vd_values, vg_range=idvg_vg_range)
        self._dcidvd=DCIdVdTemplate(*args, **kwargs,pol=pol,temp=temp,
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

    def __init__(self, *args, pol='n', temp=27, yscale='linear', shunt=None,
                 vg_values=[0,.6,1.2,1.8], idow_range=(.1e3,.1e3,1e3), **kwargs):
        self.yscale=yscale
        super().__init__(outer_variable='VG', inner_variable=('ID/W [mA/um]' if pol=='n' else '-ID/W [mA/um]'),
                         outer_values=vg_values, inner_range=np.array(idow_range)/1e3,
                         ynames=['RW [kohm.um]'],
                         *args, **kwargs)
        self.temp=temp
        self.pol=pol
        self.shunt=shunt

        num_id=(idow_range[2]-idow_range[0])/idow_range[1]
        assert abs(num_id-round(num_id))<1e-3, f"Make sure the KelvinIdVd range gives even steps {str(idow_range)}"

    @property
    def vg_values(self): return self.outer_values
    @property
    def idow_range(self): return self.inner_range*1e3

    def get_schematic_listing(self,netlister:Netlister):
        gnded=[t for t in self._patch.terminals if t not in ['d','g','t','dt']]
        netmap=dict(**{'d':'netd','g':'netg'},**{k:netlister.GND for k in gnded})
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst",netmap=netmap),
            *([netlister.nstr_R("shunt",netp='netd',netm=netlister.GND,r=self.shunt)] if self.shunt else[]),
            netlister.nstr_IDC("D",netp='netd',netm=netlister.GND,dc=0),
            netlister.nstr_VDC("G",netp='netg',netm=netlister.GND,dc=0)]

    def get_analysis_listing(self,netlister:Netlister):
        w=self._patch.get_total_device_width()
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
                    df[f'{sgnstr}ID/W [mA/um]']=-sgn*df['#p']/1e3/ \
                                                self._patch.get_total_device_width()
                    df[f'{sgnstr}IG/W [mA/um]']=-sgn*df['vg#p']/1e3/ \
                                                self._patch.get_total_device_width()
                    df['RW [kohm.um]']=df['netd']/(sgn*df[f'{sgnstr}ID/W [mA/um]'])
                    parsed_result[(vg,'f')]=df.rename(columns= \
                                                    {'netd':'VD','netg':'VG'}) \
                        [['VD','VG',f'{sgnstr}ID/W [mA/um]',f'{sgnstr}IG/W [mA/um]','RW [kohm.um]']]
                    parsed_result[(vg,'r')]=parsed_result[(vg,'f')]
        return parsed_result

    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type',self.yscale)
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self, arr, col, meas_arr):
        return arr

class CVTemplate(MultiSweepSimTemplate):

    def __init__(self, *args, temp=27, hi='g', dcs={'d':0,'s':0,'b':0}, sw='g',
                 vg_range=(0,.03,1.8), freq='1meg', **kwargs):
        super().__init__(outer_variable=None, outer_values=[freq],
                         inner_variable=f'V{sw.upper()}', inner_range=vg_range,
                         ynames=[f'C{hi}{hi} [fF/um]'],
                         *args, **kwargs)
        self.temp=temp
        num_vg=(vg_range[2]-vg_range[0])/vg_range[1]+1
        assert abs(num_vg-round(num_vg))<1e-3, f"Make sure the CV Vsweep range gives even steps {str(vg_range)}"
        
        self.freq=freq
        self.hi=hi
        self.dcs=dcs
        self.sw=sw
        if hi not in dcs: dcs[hi]=0
        if sw not in dcs: dcs[hi]=0
        assert 'float_body' not in kwargs
        try:
            spicenum_to_float(freq)
        except:
            raise Exception(f"Invalid frequency: {freq}")

    @property
    def vg_range(self):
        return self.inner_range

    def get_schematic_listing(self,netlister:Netlister):
        netmap={k:f'net{k}' for k in self.dcs}
        return [
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("CVinst",netmap=netmap)]+\
            [netlister.nstr_VAC(k.upper(),netp=netmap[k],netm=netlister.GND,
                                dc=self.dcs[k],ac=(1 if k==self.hi else 0))
                for k in self.dcs]
    
    def get_analysis_listing(self,netlister:Netlister):
        return [netlister.astr_sweepvac(self.hi.upper(),
                start=self.vg_range[0],step=self.vg_range[1],
                stop=self.vg_range[2], freq=self.freq, name='cv')]
        return analysis_listing
        
    def parse_return(self,result):
        parsed_result={}
        assert len(result)==1
        freq=spicenum_to_float(self.freq)
        df=list(result.values())[0]
        I=-df[f'v{self.hi}#p']
        df[f'C{self.hi}{self.hi} [fF/um]']=np.imag(I)/(2*np.pi*freq) /1e-15 /\
            (self._patch.get_total_device_width()/1e-6)
        df[f'V{self.sw.upper()}']=np.real(df['v-sweep'])
        parsed_result={(self.freq,'f'):df,(self.freq,'r'):df}
        return parsed_result

    def _rescale_vector(self,arr,col,meas_arr):
        return arr

    #def _validate_parsed_result(self,parsed_result):
    #    pass
    #    #assert parsed_result.keys()==set([0])
    #    # TODO: check spacings

class IdealPulsedIdVdTemplate(MultiSweepSimTemplate):

    def __init__(self, *args,temp=27,
                 vg_values=[0,.6,1.2,1.8], vd_range=(0,.1,1.8),
                 pulse_width='1u',rise_time='100n',
                 vgq=0, vdq=0,
                 **kwargs):
        super().__init__(outer_variable='VG', inner_variable='VD',
                         outer_values=vg_values, inner_range=vd_range,
                         ynames=['ID/W [uA/um]'],
                         *args, **kwargs)
        self.temp=temp
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
        gnded=[t for t in self._patch.terminals if t not in ['d','g','t','dt']]
        netmap=dict(**{'d':'netd','g':'netg'},**{k:netlister.GND for k in gnded})
        return [
            netlister.nstr_modeled_xtor("inst",netmap=netmap),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_VStep("D",netp='netd',netm=netlister.GND,dc=self.vdq,rise_time=self.rise_time, final_v=0),
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
                    self._patch.get_total_device_width()
            df['IG/W [uA/um]']=-result.loc[mask,'vg#p']/\
                    self._patch.get_total_device_width()
            df['VD']=result.loc[mask,'netd']
            df['VG']=result.loc[mask,'netg']
            parsed_result[(vg,'f')]=parsed_result[(vg,'r')]=pd.DataFrame(df)
        return parsed_result
        
    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type','linear')
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self,arr,col, meas_arr):
        return arr
        
    def _validated(self, parsed_result):
        return parsed_result
        #assert parsed_result.keys()==set([0])
        # TODO: check EVERYTHING

class VsFreqAtIrregularBias():
    def init_helper(self, fstart, fstop, pts_per_dec=None, fstep=None):
        try:
            fstart=spicenum_to_float(fstart)
            fstop=spicenum_to_float(fstop)
        except:
            raise Exception(f"Invalid frequency range: {fstart} - {fstop}")

        assert ((pts_per_dec is not None) != (fstep is not None)), \
            "Supply EITHER pts_per_dec for a log sweep, OR fstep for a linear sweep"

        if pts_per_dec is not None:
            num_f=(np.log10(fstop)-np.log10(fstart))*pts_per_dec+1
            assert abs(num_f-round(num_f))<1e-3, \
                f"Make sure the SPar range gives log-even steps {pts_per_dec,str(fstart),str(fstop)}"
            self.frequency_sweep_option='log'
        if fstep is not None:
            num_f=(fstop-fstart)/fstep+1
            assert abs(num_f-round(num_f))<1e-3, f"Make sure the frequency range gives even steps {(fstart,fstep,fstop)}"
            self.frequency_sweep_option='lin'
        self.num_f=int(round(num_f))

    @property
    def fstart(self):
        return self.inner_range[0]
    @property
    def fstop(self):
        return self.inner_range[2]
    @property
    def pts_per_dec(self):
        assert self.frequency_sweep_option=='log', "Can't give pts_per_dec if it's not a log-sweep"
        return self.inner_range[1]
    @property
    def fstep(self):
        assert self.frequency_sweep_option=='lin', "Can't give fstep if it's not a linear-sweep"
        return self.inner_range[1]

    def get_analysis_listing_helper(self, netlister_func, name):
        assert len(self.outer_values)==1, "if more than one bias point, gotta alter DC values between sweeps"
        match self.frequency_sweep_option:
            case 'log':
                return [netlister_func(pts_per_dec=self.pts_per_dec,sweep_option='dec',
                                       fstart=self.fstart, fstop=self.fstop, name=name)]
            case 'lin':
                return [netlister_func(points=self.num_f,sweep_option='lin',
                                       fstart=self.fstart, fstop=self.fstop, name=name)]
            case _:
                raise NotImplementedError(self.frequency_sweep_option)

    def parse_return_helper(self,result,name):
        assert len(self.outer_values)==1
        vg,vd=self.outer_values[0]
        if f'dc_{name}' in result:
            # TODO: 'VD#p' is a magic string
            #print("Options included:",result[f'dc_{name}'].keys())
            for x in ['vd#p','vport2#p']:
                if x in result[f'dc_{name}']:
                    result[name]['I [A]']=-result[f'dc_{name}'][x].iloc[0]
        parsed_result={(vg,vd): result[name]}
        return parsed_result

    def to_merged_table(self,result):
        return VsIrregularBiasAtFreq.to_merged_table(self,result)

def _identity(x): return x

class VsIrregularBiasAtFreq():
    def init_helper(self, vgvds, vs_vg, vs_vd, vs_vo, vs_id, frequency):
        if hasattr(frequency,'__len__'):
            fstart,pts_per_dec,fstop=frequency
        else:
            fstart,pts_per_dec,fstop=frequency,1,frequency
            self._sweeper_kwargs={'queryvar':'freq','querytarget':frequency}
        VsFreqAtIrregularBias.init_helper(self,fstart=fstart,fstop=fstop,pts_per_dec=pts_per_dec)
        vgs=list(sorted(set([k[0] for k in vgvds])))
        vds=list(sorted(set([k[1] for k in vgvds])))
        if len(vs_vg):
            self._vsvg=MultiSweepSimTemplate(outer_variable='VD',inner_variable='VG',
                                             ynames=vs_vg,outer_values=vds,directions=['f'])
        if len(vs_vd):
            self._vsvd=MultiSweepSimTemplate(outer_variable='VG',inner_variable='VD',
                                             ynames=vs_vd,outer_values=vgs,directions=['f'])
        if len(vs_vo):
            self._vsvo=MultiSweepSimTemplate(outer_variable='VD',inner_variable='VoV',
                                             ynames=vs_vo,outer_values=vds,directions=['f'])
        if len(vs_id):
            self._vsid=MultiSweepSimTemplate(outer_variable='VD',inner_variable='VG',
                                             ynames=vs_id,outer_values=vds,directions=['f'])
            # Override validation for vs ID
            self._vsid._validated=_identity
            sgn = -1 if self.pol == 'p' else 1
            sgnstr = "-" if self.pol == 'p' else ''
            self._vsid._xname=f'{sgnstr}I/W [uA/um]'
        return fstart,pts_per_dec,fstop
    def get_analysis_listing_helper(self, netlister_alter, netlister_func, namepre, inc_portnum=True):
        assert self.frequency_sweep_option=='log'
        lst=[]
        for i,(vg,vd) in enumerate(self.outer_values):
            lst.append(netlister_alter('G',vg,**(dict(portnum=1) if inc_portnum else {})))
            lst.append(netlister_alter('D',vd,**(dict(portnum=2) if inc_portnum else {})))
            lst.append(netlister_func(pts_per_dec=self.pts_per_dec,sweep_option='dec',
                                           fstart=self.fstart, fstop=self.fstop, name=f'{namepre}{i}'))
        return lst

    def parse_return_helper(self, result, namepre):
        result={k:v for k,v in result.items() if k.startswith(namepre) or k.startswith(f'dc_{namepre}')}
        assert len([k for k in result if k.startswith(namepre)])==len(self.outer_values)
        parsed_result={}
        for i,(vg,vd) in enumerate(self.outer_values):

            if f'dc_{namepre}{i}' in result:
                # TODO: 'VD#p' is a magic string
                #print("Options included:", result[f'dc_{namepre}{i}'].keys())
                for x in ['vd#p', 'vport2#p']:
                    if x in result[f'dc_{namepre}{i}']:
                        result[f'{namepre}{i}']['I [A]'] = -result[f'dc_{namepre}{i}'][x].iloc[0]
            parsed_result[(vg,vd)]=result[f'{namepre}{i}']
        return parsed_result

    def generate_figures_helper(self, meas_data=None,
                         layout_params={}, y_axis_type='linear', x_axis_type='linear',
                         vizid=None):
        #assert (x_axis_type is None)
        vg_sweeps=form_multisweep(meas_data,1,0,'VG',**self._sweeper_kwargs)
        vd_sweeps=form_multisweep(meas_data,0,1,'VD',**self._sweeper_kwargs)
        return [
            *(self._vsvg.generate_figures(meas_data=vg_sweeps,layout_params=layout_params,vizid=vizid,
                                          y_axis_type=y_axis_type) if hasattr(self,'_vsvg') else []),
            *(self._vsvo.generate_figures(meas_data=vg_sweeps,layout_params=layout_params,vizid=vizid,
                                          y_axis_type=y_axis_type) if hasattr(self,'_vsvo') else []),
            *(self._vsvd.generate_figures(meas_data=vd_sweeps,layout_params=layout_params,vizid=vizid,
                                          y_axis_type=y_axis_type) if hasattr(self,'_vsvd') else []),
            *(self._vsid.generate_figures(meas_data=vg_sweeps,layout_params=layout_params,vizid=vizid,
                                          y_axis_type=y_axis_type, x_axis_type=x_axis_type) if hasattr(self,'_vsid') else []),]

    def update_figures_helper(self, parsed_result, vizid=None):
        vg_sweeps=form_multisweep(parsed_result,1,0,'VG',**self._sweeper_kwargs)
        vd_sweeps=form_multisweep(parsed_result,0,1,'VD',**self._sweeper_kwargs)
        if hasattr(self,'_vsvg'): self._vsvg.update_figures(vg_sweeps, vizid=vizid)
        if hasattr(self,'_vsvo'): self._vsvo.update_figures(vg_sweeps, vizid=vizid)
        if hasattr(self,'_vsvd'): self._vsvd.update_figures(vd_sweeps, vizid=vizid)
        if hasattr(self,'_vsid'): self._vsid.update_figures(vg_sweeps, vizid=vizid)

    @property
    def fstart(self):
        return self.inner_range[0]
    @property
    def fstop(self):
        return self.inner_range[2]
    @property
    def pts_per_dec(self):
        assert self.frequency_sweep_option=='log', "Can't give pts_per_dec if it's not a log-sweep"
        return self.inner_range[1]
    @property
    def fstep(self):
        assert self.frequency_sweep_option=='lin', "Can't give fstep if it's not a linear-sweep"
        return self.inner_range[1]


    def to_merged_table(self,result):
        dfs=[]
        for (vg,vd),df in result.items():
            df=df.set_index([self.inner_variable])
            df['VG']=vg
            df['VD']=vd
            dfs.append(df)
        return pd.concat(dfs)

class SParTemplate(MultiSweepSimTemplate):

    def __init__(self, temp=27, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.temp=temp

    def get_schematic_listing(self,netlister:Netlister):
        #netlister.nstr_param(params={'vg':0,'vd':0})+\
        vg,vd=self.outer_values[0]
        gnded=[t for t in self._patch.terminals if t not in ['d','g','t','dt']]
        netmap=dict(**{'d':'netd','g':'netg'},**{k:netlister.GND for k in gnded})
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst",netmap=netmap),
            netlister.nstr_port("D",netp='netd',netm=netlister.GND,dc=vd,portnum=2,ac=0),
            netlister.nstr_port("G",netp='netg',netm=netlister.GND,dc=vg,portnum=1,ac=1)]

    @staticmethod
    def _sparam_dataframe_helper(df):
        df['freq'] = np.real(df['freq'])
        if 's11' in df.columns and 'S11' not in df.columns:
            df.rename(columns={'s11':'S11','s12':'S12','s21':'S21','s22':'S22'},inplace=True)

        # print(df[['S11', 'S12', 'S21', 'S22']])
        if 'Y11' not in df.columns:
            df['Y11'], df['Y12'], df['Y21'], df['Y22'] = s2y(df['S11'], df['S12'], df['S21'], df['S22'])
        if 'Z11' not in df.columns:
            df['Z11'], df['Z12'], df['Z21'], df['Z22'] = s2z(df['S11'], df['S12'], df['S21'], df['S22'])

    @staticmethod
    def rf_param_helper(df,width):
        SParTemplate._sparam_dataframe_helper(df)

        # h21 is a current ratio, so 20x log
        df[f'|h21| [dB]']=20*np.log10(np.abs(df.Y21/df.Y11))

        # https://en.wikipedia.org/wiki/Mason%27s_invariant#Derivation_of_U
        # U is already a power ratio so just 10x log
        re=np.real; im=np.imag
        with np.errstate(invalid='ignore'):
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
        df['K']=K
        df['MAG [dB]']=10*np.log10(np.choose(MAG>0,[np.NaN,MAG]))
        df['MSG [dB]']=10*np.log10(MSG)
        df['MAG-MSG [dB]']=10*np.log10(np.choose(K>=1,[MSG,MAG]))

        # RF small-signal circuit parameters
        Wum=width*1e6

        fF=1e-15; uS=1e-6
        w=2*np.pi*df['freq']
        df['Cgd/W [fF/um]']=-im(df.Y12) / w / Wum /fF
        df['Cgs/W [fF/um]']=im(df.Y11 + df.Y12) / w / Wum /fF
        df['Cds/W [fF/um]']=im(df.Y22+df.Y12) / w / Wum /fF
        df['Rds*W [Ohm.um]']=1/re(df.Y22+df.Y12) * Wum
        df['GM/W [uS/um]']=np.abs(df.Y21-df.Y12) / Wum / uS
        Rs=df['Rs [Ohm.um]']=re(df.Z12) * Wum
        df['Rd*W [Ohm.um]']=(re(df.Z22)-Rs) * Wum
        df['Rg*W [Ohm.um]']=(re(df.Z11)-Rs) * Wum
        df['GM/2πCgs [GHz]']=df['GM/W [uS/um]']/(2*np.pi*df['Cgs/W [fF/um]']) #uS/fF=GHz

        # S-parameters
        for ii in ['11','12','21','22']:
            for comp,func in (('Re',np.real),('Im',np.imag)):
                df[f'{comp}S{ii}']=func(df[f'S{ii}'])

        return df

    def postparse_return(self, parsed_result):
        W = self._patch.get_total_device_width()
        for df in parsed_result.values():
            self.rf_param_helper(df,width=W)
        return parsed_result

    def _rescale_vector(self,arr,col, meas_arr):
        return arr
    def _required_keys(self):
        return self.outer_values


    def _validated(self, parsed_result):
        # Overriding because this does freq num points instead of freq delta
        # TODO: implement this
        return parsed_result

class SParVFreqTemplate(SParTemplate,VsFreqAtIrregularBias):
    def __init__(self, *args,
                 vg, vd, fstart, fstop, temp=27, pts_per_dec=None, fstep=None, **kwargs):
        VsFreqAtIrregularBias.init_helper(self,fstart=fstart,fstop=fstop,pts_per_dec=pts_per_dec,fstep=fstep)
        SParTemplate.__init__(self,outer_variable=None, outer_values=[(vg,vd)], inner_variable='freq',
                         inner_range=(fstart,pts_per_dec,fstop), temp=temp,
                         ynames=['|h21| [dB]','U [dB]','MAG-MSG [dB]','ReS11','ImS11','ReS22','ImS22','ReS12','ImS12'],
                         *args, **kwargs)

    def get_analysis_listing(self,netlister:Netlister):
        return VsFreqAtIrregularBias.get_analysis_listing_helper(self,netlister_func=netlister.astr_spar,name='spar')

    def parse_return(self,result):
        return VsFreqAtIrregularBias.parse_return_helper(self,result,name='spar')

    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type','log')
        kwargs['x_axis_type']=kwargs.get('x_axis_type','log')
        return super().generate_figures(*args,**kwargs)

    def _make_figures(self, meas_cds_c, meas_cds_l, sim_cds, layout_params, y_axis_type='log',x_axis_type='log', override_line_color=None):
        assert y_axis_type=='log'
        assert x_axis_type=='log'
        assert override_line_color is None
        figpow=bokeh.plotting.figure(tools=get_tools(),x_axis_type='log',**layout_params,tooltips=[('','$snap_x Hz'),('','$name = $snap_y')])
        figpow.scatter(x='x',y='|h21| [dB]',source=meas_cds_c,color='blue',legend_label='h21',name='|h21| meas')
        figpow.multi_line(xs='x',ys='|h21| [dB]',source=meas_cds_l,color='blue',legend_label='h21',name='|h21| meas')
        figpow.multi_line(xs='x',ys='|h21| [dB]',source=sim_cds,color='red',legend_label='h21',name='|h21| sim')

        figpow.scatter(x='x',y='U [dB]',source=meas_cds_c,color='green',legend_label='U',name='U meas')
        figpow.multi_line(xs='x',ys='U [dB]',source=meas_cds_l,color='green',legend_label='U',name='U meas')
        figpow.multi_line(xs='x',ys='U [dB]',source=sim_cds,color='orange',legend_label='U',name='U sim')

        figpow.scatter(x='x',y='MAG-MSG [dB]',source=meas_cds_c,color='lightblue',legend_label='MAG/MSG',name='MAG/MSG meas')
        figpow.multi_line(xs='x',ys='MAG-MSG [dB]',source=meas_cds_l,color='lightblue',legend_label='MAG/MSG',name='MAG/MSG meas')
        figpow.multi_line(xs='x',ys='MAG-MSG [dB]',source=sim_cds,color='burlywood',legend_label='MAG/MSG',name='MAG/MSG sim')

        figpow.yaxis.axis_label='Power Gain [dB]'
        figpow.xaxis.axis_label='Frequency [Hz]'
        fig_legend_config(figpow)
        figpow.legend.location='top_right'
        figpow.title=str(self.outer_values)

        figsmi=smith_chart(**layout_params)
        figsmi.scatter(x='ReS11',y='ImS11',source=meas_cds_c,color='blue',legend_label='S11',line_width=2)
        figsmi.scatter(x='ReS22',y='ImS22',source=meas_cds_c,color='green',legend_label='S22',line_width=2)
        figsmi.scatter(x='ReS12',y='ImS12',source=meas_cds_c,color='lightblue',legend_label='S12',line_width=2)
        figsmi.multi_line(xs='ReS11',ys='ImS11',source=sim_cds,color='red',legend_label='S11',line_width=2)
        figsmi.multi_line(xs='ReS22',ys='ImS22',source=sim_cds,color='orange',legend_label='S22',line_width=2)
        figsmi.multi_line(xs='ReS12',ys='ImS12',source=sim_cds,color='burlywood',legend_label='S12',line_width=2)
        fig_legend_config(figsmi)
        figsmi.legend.location='top_right'

        return [figpow,figsmi]

    def to_merged_table(self,result):
        return VsFreqAtIrregularBias.to_merged_table(self,result)

class SParVBiasTemplate(SParTemplate,VsIrregularBiasAtFreq):
    def __init__(self, *args, vgvds, frequency, temp=27, **kwargs):
        SParTemplate.__init__(self,*args, outer_variable=None, outer_values=vgvds, inner_variable='freq',
                              inner_range=(frequency,1,frequency), temp=temp, **kwargs)
        VsIrregularBiasAtFreq.init_helper(self,vgvds=vgvds,frequency=frequency,
              vs_vg=['GM/W [uS/um]','Cgs/W [fF/um]','Cgd/W [fF/um]','GM/2πCgs [GHz]'],
              vs_vd=['Rds*W [Ohm.um]','Cds/W [fF/um]'],
              vs_vo=[], vs_id=[])

    def get_analysis_listing(self,netlister:Netlister):
        return VsIrregularBiasAtFreq.get_analysis_listing_helper(self,
            netlister_alter=netlister.astr_altervportdc,netlister_func=netlister.astr_spar,namepre='spar')

    def parse_return(self,result):
        return VsIrregularBiasAtFreq.parse_return_helper(self,result,namepre='spar')

    def generate_figures(self, *args, **kwargs):
        return VsIrregularBiasAtFreq.generate_figures_helper(self,*args,**kwargs)

    def update_figures(self, *args, **kwargs):
        return VsIrregularBiasAtFreq.update_figures_helper(self,*args, **kwargs)

    def to_merged_table(self,result):
        return VsIrregularBiasAtFreq.to_merged_table(self,result)

class HFNoiseTemplate(SParTemplate):

    def postparse_return(self,parsed_result):

        from scipy.constants import k as kb
        Ts=290 # to match IEEE definition, source temperature is 290K
        #import pdb; pdb.set_trace()
        for df in parsed_result.values():
            #print(f"Calculating Fmin from {list(df.columns)}")
            self._sparam_dataframe_helper(df)
            Rn=np.real(1/(4*kb*Ts)*df.cy22/np.abs(df.Y21)**2)

            Yc=-df.Y21*df.cy12/df.cy22+df.Y11
            Gc=np.real(Yc)
            Bc=np.imag(Yc)

            tmp=(Yc-df.Y11)/df.Y21
            Gu=np.real(df.cy11-np.abs(df.cy12)**2/df.cy22)/(4*kb*Ts)

            Gopt=np.sqrt(Gc**2+Gu/Rn)
            Bopt=-Bc

            Fmin=1+2*Rn*(Gopt+Gc)

            df['Fmin']=Fmin
            df['NFmin'] = 10*np.log10(Fmin)
            df['Gopt']=Gopt
            df['Bopt']=Bopt
            df['Rn']=Rn
        return parsed_result

class HFNoiseVFreqTemplate(HFNoiseTemplate,VsFreqAtIrregularBias):
    def __init__(self, *args,
                 vg, vd, fstart, fstop, temp=27, pts_per_dec=None, fstep=None, **kwargs):
        VsFreqAtIrregularBias.init_helper(self,fstart=fstart,fstop=fstop,pts_per_dec=pts_per_dec,fstep=fstep)
        HFNoiseTemplate.__init__(self,outer_variable=None, outer_values=[(vg,vd)], inner_variable='freq',
                         inner_range=(fstart,pts_per_dec,fstop), temp=temp,
                         ynames=['NFmin','Rn'],
                         *args, **kwargs)

    def get_analysis_listing(self,netlister:Netlister):
        return VsFreqAtIrregularBias.get_analysis_listing_helper(self,netlister_func=netlister.astr_sparnoise,name='sparnoise')

    def parse_return(self,result):
        return VsFreqAtIrregularBias.parse_return_helper(self,result,name='sparnoise')

    def generate_figures(self,*args,**kwargs):
        #kwargs['y_axis_type']=kwargs.get('y_axis_type','log')
        kwargs['x_axis_type']=kwargs.get('x_axis_type','log')
        return super().generate_figures(*args,**kwargs)

    def to_merged_table(self,result):
        return VsFreqAtIrregularBias.to_merged_table(self,result)

class HFNoiseVBiasTemplate(HFNoiseTemplate,VsIrregularBiasAtFreq):
    def __init__(self, *args, vgvds, frequency, temp=27, pol='n',**kwargs):
        self.pol=pol
        SParTemplate.__init__(self,*args, outer_variable=None, outer_values=vgvds, inner_variable='freq',
                              inner_range=(frequency,1,frequency), temp=temp, **kwargs)
        VsIrregularBiasAtFreq.init_helper(self,vgvds=vgvds,frequency=frequency,
              vs_vg=['NFmin','Rn'],
              vs_vd=[],
              vs_vo=[],
              vs_id=['NFmin','Rn'],)

    def get_analysis_listing(self,netlister:Netlister):
        return VsIrregularBiasAtFreq.get_analysis_listing_helper(self,
            netlister_alter=netlister.astr_altervportdc,netlister_func=netlister.astr_sparnoise,namepre='sparnoise')

    def parse_return(self,result):
        sgn = -1 if self.pol == 'p' else 1
        sgnstr = "-" if self.pol == 'p' else ''
        result=VsIrregularBiasAtFreq.parse_return_helper(self,result,namepre='sparnoise')
        for subresult in result.values():
            subresult[f'{sgnstr}I/W [uA/um]'] = sgn*subresult['I [A]'] / self._patch.get_total_device_width()
        return result

    def generate_figures(self, *args, **kwargs):
        return VsIrregularBiasAtFreq.generate_figures_helper(self,*args,**kwargs)

    def update_figures(self, *args, **kwargs):
        return VsIrregularBiasAtFreq.update_figures_helper(self,*args, **kwargs)

    def to_merged_table(self,result):
        return VsIrregularBiasAtFreq.to_merged_table(self,result)

class LFNoiseTemplate(MultiSweepSimTemplate):

    def __init__(self, temp=27, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.temp=temp

    def get_schematic_listing(self,netlister:Netlister):
        vg,vd=self.outer_values[0]
        gnded=[t for t in self._patch.terminals if t not in ['d','g','t','dt']]
        netmap=dict(**{'d':'netd','g':'netg'},**{k:netlister.GND for k in gnded})
        return [
            netlister.nstr_iabstol('1e-15'),
            netlister.nstr_temp(temp=self.temp),
            netlister.nstr_modeled_xtor("inst",netmap=netmap,
                                        internals_to_save=self.internals_to_save),
            netlister.nstr_iprobe("IPRB",netp='netdap',netm='netd'),
            netlister.nstr_VDC("D",netp='netdap',netm=netlister.GND,dc=vd),
            netlister.nstr_VAC("G",netp='netg',netm=netlister.GND,dc=vg,ac=1)]

    def _rescale_vector(self,arr,col, meas_arr):
        return arr
    def _required_keys(self):
        return self.outer_values

    def postparse_return(self,parsed_result):
        for k,v in parsed_result.items():
            v['sid [A^2/Hz]']=v['onoise [A/sqrt(Hz)]']**2
            v['sid/W^2 [A^2/Hz/um^2]']=v['sid [A^2/Hz]']/(self._patch.get_total_device_width()*1e6)**2
            v['svg [V^2/Hz]']=v['inoise [V/sqrt(Hz)]']**2
            v['Gm [uS/um]']=v['gain [A/V]']/self._patch.get_total_device_width()
            v['sid/ID^2 [1/Hz]']=v['sid [A^2/Hz]']/(v['I [A]'])**2
        return parsed_result

    def _validated(self, parsed_result):
        # Overriding because this does freq num points instead of freq delta
        # TODO: implement this
        return parsed_result

class LFNoiseVFreqTemplate(LFNoiseTemplate, VsFreqAtIrregularBias):
    def __init__(self,
                 vg, vd, fstart, fstop, temp=27, pts_per_dec=None, fstep=None, *args, **kwargs):
        VsFreqAtIrregularBias.init_helper(self,fstart=fstart,fstop=fstop,pts_per_dec=pts_per_dec,fstep=fstep)
        LFNoiseTemplate.__init__(self, outer_variable=None, outer_values=[(vg, vd)], inner_variable='freq',
                                 inner_range=(fstart,pts_per_dec,fstop), temp=temp,
                                 ynames=['sid [A^2/Hz]','sid/ID^2 [1/Hz]','svg [V^2/Hz]'],
                                 *args, **kwargs)
    def get_analysis_listing(self,netlister:Netlister):
        netlister_func=lambda *args,**kwargs: netlister.astr_noise(outprobe='IPRB',vsrc='VG',*args,**kwargs)
        return VsFreqAtIrregularBias.get_analysis_listing_helper(self,netlister_func=netlister_func,name='noise')

    def parse_return(self,result):
        result=VsFreqAtIrregularBias.parse_return_helper(self,result,name='noise')
        # TODO: REMOVE
        # Only necessary because don't support getting I in spectre yet
        for k,sw in result.items():
            if 'I [A]' not in sw:
                sw['I [A]'] = sw['freq'] * np.NaN
        return result

    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type','log')
        kwargs['x_axis_type']=kwargs.get('x_axis_type','log')
        return super().generate_figures(*args,**kwargs,override_line_color='red')

    def to_merged_table(self,result):
        return VsFreqAtIrregularBias.to_merged_table(self,result)

class LFNoiseVBiasTemplate(LFNoiseTemplate, VsIrregularBiasAtFreq):

    @staticmethod
    def _integrate_1of(df, lower, upper):

        row={}
        y_int_list=[('sid/ID^2 [1/Hz]','int. sid/ID^2'),('svg [V^2/Hz]','int. svg [V^2]'),('Gm [uS/um]','avg. Gm [uS/um]')]
        for yvar,intyvar in y_int_list:
            x=np.array(df['freq'])
            y=np.array(df[yvar])
            assert x[0]<lower*1.1
            assert x[-1]>upper*.9

            # Conventional integration
            if ('Gm' in yvar):
                cinty=cumtrapz(y,x,initial=0)
                inty=np.diff(interp1d(x,cinty,fill_value='extrapolate')([lower,upper]))[0]
            elif ('sid' in yvar) or ('svg' in yvar):
                # 1/f tuned integration
                # This is equivalent to integral g(f)df in the dense points limit
                # but numerically should be better for sparse frequency points assuming g(f) goes as ~1/f
                # This is just the same integral with u-substituion u=ln(f), du=df/f -> integral g(f)f dln(f)
                # And the edge interpolation is done with ln(f) rather than f
                # since the result should be a smoother function of ln(f)
                cinty = cumtrapz(y*x, np.log(x), initial=0)
                inty=np.diff(interp1d(np.log(x),cinty,fill_value='extrapolate')(np.log([lower,upper])))[0]
            else:
                raise Exception(f"What's the best way to integrate {yvar}")


            assert ('int.' in intyvar) or ('avg.' in intyvar)
            if 'avg.' in intyvar: inty=inty/(upper-lower)
            row[intyvar]=[inty]

        for c in df.columns:
            if c not in [yvar for yvar,intyvar in y_int_list]+['freq']:

                if np.allclose(df[c],df[c].iloc[0],atol=0):
                    row[c]=[df[c].iloc[0]]
                #else:
                #    row[c]=[np.NaN]

        return pd.concat([df,pd.DataFrame(row)]).reset_index(drop=True)

    def __init__(self, *args,
                 vgvds, frequency, temp=27, pol='n', integration_minf=100, integration_maxf=1e6, **kwargs):
        self.pol=pol
        LFNoiseTemplate.__init__(self, *args, outer_variable=None, outer_values=vgvds, inner_variable='freq',
                                 temp=temp, **kwargs)
        fstart,pts_per_dec,fstop=VsIrregularBiasAtFreq.init_helper(self,vgvds=vgvds,frequency=frequency,
                                          vs_vg=[],#['sid/W^2 [A^2/Hz/um^2]','sid/ID^2 [1/Hz]','svg [V^2/Hz]','Gm [uS/um]'],
                                          #vs_vo=['sid/ID^2 [1/Hz]','svg [V^2/Hz]','Gm [uS/um]'],
                                          vs_vo=[],
                                          #vs_vd=['sid/W^2 [A^2/Hz/um^2]','svg [V^2/Hz]','Gm [uS/um]']
                                          vs_vd=[],
                                          vs_id=['int. sid/ID^2','int. svg [V^2]','avg. Gm [uS/um]'])
        self.inner_range=(fstart,pts_per_dec,fstop)
        #self._sweeper_kwargs={'collapser':self._integrate_1of}
        self._sweeper_kwargs={'queryvar':'freq'}
        self._integration_minf=integration_minf
        self._integration_maxf=integration_maxf


    def get_analysis_listing(self,netlister:Netlister):
        netlister_func=lambda *args,**kwargs: netlister.astr_noise(outprobe='IPRB',vsrc='VG',*args,**kwargs)
        return VsIrregularBiasAtFreq.get_analysis_listing_helper(self,
            netlister_alter=netlister.astr_altervdc,netlister_func=netlister_func,namepre='noise',inc_portnum=False)

    def parse_return(self,result):
        result=VsIrregularBiasAtFreq.parse_return_helper(self,result,namepre='noise')

        sgn = -1 if self.pol == 'p' else 1
        sgnstr = "-" if self.pol == 'p' else ''
        vg_sweeps=form_multisweep(result,1,0,'VG',queryvar='freq', querytarget=self.fstart)
        for (vd,dir_),sw in vg_sweeps.items():
            sw[f'{sgnstr}I/W [uA/um]']=sgn*sw['I [A]']/self._patch.get_total_device_width()
            from datavac.util.maths import VTCC
            try:
                Iarr,VGarr=np.array([sw[f'{sgnstr}I/W [uA/um]']]),np.array([sw['VG']])
                icc=1
                if Iarr[0][0]<icc:
                    vt=VTCC(Iarr,VGarr,icc)[0]
                else:
                    #import pdb; pdb.set_trace()
                    Iarr, VGarr = Iarr[0],VGarr[0]
                    assert (Iarr[0]/icc)<1.3, "Not gonna extrapolate more than 30%"
                    vt=VGarr[0]-((VGarr[1]-VGarr[0])/np.log(Iarr[1]/Iarr[0])*np.log(Iarr[0]/icc))
            except:
                vt=np.NaN
            for (vg_,vd_),subresult in result.items():
                if vd_==vd:
                    subresult['VoV']=vg_-vt
                    subresult[f'{sgnstr}I/W [uA/um]']=sgn*subresult['I [A]']/self._patch.get_total_device_width()
        return result

    def postparse_return(self,parsed_result):
        parsed_result=super().postparse_return(parsed_result)
        res={}
        for k,df in parsed_result.items():
            res[k]=self._integrate_1of(df, self._integration_minf, self._integration_maxf)
        return res

    def generate_figures(self, *args, **kwargs):
        #import pdb; pdb.set_trace()
        return VsIrregularBiasAtFreq.generate_figures_helper(self,*args,**kwargs,y_axis_type='log',x_axis_type='log')

    def update_figures(self, *args, **kwargs):
        return VsIrregularBiasAtFreq.update_figures_helper(self,*args, **kwargs)

    def to_merged_table(self,result):
        return VsIrregularBiasAtFreq.to_merged_table(self,result)

class TemplateGroup(UserDict[str,SimTemplate]):
    def __init__(self,**templates):
        super().__init__(**templates)
        self._panes={}
    def set_patch(self,params_by_template):
        if isinstance(params_by_template,ParamPatch):
            for t in self.values():
                t.set_patch(params_by_template)
        else:
            assert set(params_by_template.keys())==set(self.keys())
            for k,t in self.items():
                t.set_patch(params_by_template[k])
    def only(self,*names):
        assert all(n in self for n in names)
        return self.__class__(**{k:v for k,v in self.items() if k in names})

    def get_figure_pane(self, meas_data=None, fig_layout_params={},vizid=None):
        figs=sum([t.generate_figures(
                            meas_data=meas_data.get(stname,None),
                            layout_params=fig_layout_params, vizid=vizid)
                       for stname,t in self.items()],[])
        self._panes[vizid]=pn.pane.Bokeh(bokeh.layouts.gridplot([figs]))
        return self._panes[vizid]

    def update_figures(self, new_results, vizid=None):
        actually_did_update=False
        for stname in self:
            if self[stname].update_figures((new_results[stname] if new_results else None),vizid=vizid):
                actually_did_update=True
        if actually_did_update:
            logger.debug(f"(Not) pushing bokeh update to notebook for vizid {vizid}")
            # Apparently this isn't needed anymore?
            #pn.io.push_notebook(self._panes[vizid])

    def parsed_results_to_vector(self,parsed_results,roi, meas_parsed_results):
        return np.concatenate([simtemp.parsed_results_to_vector(parsed_results[k],roi[k],meas_parsed_results[k]) for k,simtemp in self.items()])

    def to_csv(self,parsed_results,prefix=''):
        for stname in self:
            if stname in parsed_results and parsed_results[stname] is not None:
                self[stname].to_csv(parsed_results[stname],filename=f'{prefix}{stname}.csv')
            else:
                print(f"Skipping {stname} because not present")