import numpy as np
import pyspectre as psp
from tempfile import NamedTemporaryFile
import bokeh
import bokeh.plotting

from compyct.paramsets import ParamSet, spicenum_to_float, ParamPlace
from compyct.python_models import python_compact_models

class SimTemplate():
    def __init__(self, model_paramset=None):
        self._tf=None
        if model_paramset is not None:
            self.set_paramset(model_paramset)

    def set_paramset(self,model_paramset):
        assert self._tf is None, "Can't change paramset after creating netlist"
        self.model_name=model_paramset.model+"_standin"
        self.model_paramset=model_paramset.copy()
        
    def get_netlist_str(self):
        raise NotImplementedError
    def get_analyses_str(self):
        raise NotImplementedError
    def parse_return(self, result):
        return result

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
        
    def get_netlist_file(self):
        if self._tf is None:
            self._tf=NamedTemporaryFile(prefix=self.__class__.__name__,mode='w')
            self._tf.write(f"// {self.__class__.__name__}\n")
            self._tf.write(f"simulator lang = spectre\n")
            if self.model_paramset is not None:
                for i in self.model_paramset.includes:
                    if type(i)==str:
                        self._tf.write(
                            f"{'ahdl_' if i.endswith('.va') else ''}include"\
                                f" \"{i}\"\n")
                    else:
                        self._tf.write(
                            f"{'ahdl_' if i[0].endswith('.va') else ''}include"\
                                f" \"{i[0]}\" {' '.join(i[1:])}\n")
                self._tf.write(f"model {self.model_name} {self.model_paramset.model}\n")
                paramlinedict={("modparam_"+k):self.model_paramset.get_value(k)
                                    for k in self.model_paramset}
                self._tf.write(
                    f"parameters "+\
                    ' '.join([f'{k}={v}' for k,v in paramlinedict.items()])\
                    +"\n")
            instance_params={k:self.model_paramset.get_value(k)
                                 for k in self.model_paramset
                                     if self.model_paramset.get_place(k)\
                                         ==ParamPlace.INSTANCE}
            if len(instance_params):
                self._tf.write(f"parameters "+\
                                   ' '.join(f'instparam_{k}={v}'
                                        for k,v in instance_params.items())+\
                               "\n")
            self._tf.write(self.get_netlist_str()+"\n")
            if self.model_paramset is not None:                
                self._tf.write(
                    f"set_modparams altergroup {{\nmodel {self.model_name}"\
                    f" {self.model_paramset.model} "+\
                    " ".join((f"{k}=modparam_{k}" for k in self.model_paramset))\
                    +"\n}\n")
            self._tf.write(self.get_analyses_str()+"\n")
            self._tf.flush()
        return self._tf.name

    def _get_instance_param_part(self):
        return ' '.join(f'{k}=instparam_{k}' for k in self.model_paramset
                        if self.model_paramset.get_place(k)==ParamPlace.INSTANCE)
        
    def get_spectre_names_for_param(self,param):
        prefix={ParamPlace.MODEL.value:'mod',ParamPlace.INSTANCE.value:'inst'}\
                    [self.model_paramset.get_place(param).value]+'param_'
        return prefix+param
        
    def update_paramset_and_return_spectre_changes(self,new_values):
        nv=self.model_paramset.update_and_return_changes(new_values)
        return {self.get_spectre_names_for_param(k):v for k,v in nv.items()}
        

class MultiSweepSimTemplate(SimTemplate):
    def __init__(self, *args,
                 outer_variable=None,inner_variable=None, ynames=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.outer_variable=outer_variable
        self.inner_variable=inner_variable
        self.ynames=ynames
        self._sources={}
        
    def _parsed_result_to_cds_data(self,parsed_result):
        assert len(self.ynames)==1
        yname=self.ynames[0]
        
        data={'x':[],'y':[],'legend':[]}
        if parsed_result is None: parsed_result={}
        for key, df in parsed_result.items():
            data['x'].append(df[self.inner_variable].to_numpy())
            data['y'].append(df[yname].to_numpy())
            data['legend'].append(key)
        return data
    
    def generate_figures(self, meas_data=None,
                         layout_params={}, y_axis_type='linear',vizid=None):
        fig=bokeh.plotting.figure(#x_range=self.vg_range,y_range=(1e-8,1),
                                  y_axis_type=y_axis_type,**layout_params)
    
        self._sources[vizid]\
               =[self._update_cds_with_parsed_result(cds=None,parsed_result=None)]
        
        #if meas_data is not None:
        meas_cds=self._update_cds_with_parsed_result(cds=None,
                    parsed_result=meas_data,flattened=True)
        fig.circle(x='x',y='y',source=meas_cds,legend_field='legend')
            
        sim_cds=self._sources[vizid][0]
        fig.multi_line(xs='x',ys='y',source=sim_cds,color='red')

        fig.yaxis.axis_label=",".join(self.ynames)
        fig.legend.margin=0
        fig.legend.spacing=0
        fig.legend.padding=4
        fig.legend.label_text_font_size='8pt'
        fig.legend.label_height=10
        fig.legend.label_text_line_height=10
        fig.legend.glyph_height=10
        fig.legend.location='bottom_right'
        return [fig]
        
    def update_figures(self, parsed_result, vizid=None):
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

    def __init__(self, *args,
                 vg_values=[0,.6,1.2,1.8], vd_range=(0,.1,1.8), **kwargs):
        super().__init__(outer_variable='VG', inner_variable='VD',
                         ynames=['ID/W [uA/um]'],
                         *args, **kwargs)
        
        num_vd=(vd_range[2]-vd_range[0])/vd_range[1]
        assert abs(num_vd-int(num_vd))<1e-3, f"Make sure the IdVd range gives even steps {str(vd_range)}"
        
        self.vg_values=vg_values
        self.vd_range=vd_range
    
    def get_netlist_str(self):
        setup_part:str=\
            f"parameters vd=0 vg=0\n"+\
            f"X0 (netd netg 0 0) {self.model_name} "+\
                self._get_instance_param_part() + "\n"\
            f"VD (netd 0) vsource dc=vd type=dc\n"\
            f"VG (netg 0) vsource dc=vg type=dc\n"
        return setup_part
        
    def get_analyses_str(self):
        analysis_part:str=""
        for i_vg,vg in enumerate(self.vg_values,start=1):
            analysis_part+=f"tovgnum{i_vg} alter param=vg value={vg}\n"
            analysis_part+=f"dcvgnum{i_vg} dc dev=VD param=dc "\
                                f"start={self.vd_range[0]} step={self.vd_range[1]} stop={self.vd_range[2]}\n"
        return analysis_part

    def parse_return(self,result):
        parsed_result={}
        for i_vg,vg in enumerate(self.vg_values,start=1):
            for key in result:
                if f'dcvgnum{i_vg}' in key:
                    df=result[key].copy()
                    df['ID/W [uA/um]']=-df['VD:p']/\
                            self.model_paramset.get_total_device_width()
                    df['IG/W [uA/um]']=-df['VG:p']/\
                            self.model_paramset.get_total_device_width()
                    parsed_result[vg]=df.rename(columns=\
                                {'netd':'VD','netg':'VG'})\
                            [['VD','VG','ID/W [uA/um]','IG/W [uA/um]']]
        return parsed_result
        
    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type','linear')
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self,arr):
        return arr
        
class DCIdVgTemplate(MultiSweepSimTemplate):

    def __init__(self, *args,
                 vd_values=[.05,1.8], vg_range=(0,.03,1.8), **kwargs):
        super().__init__(outer_variable='VD', inner_variable='VG',
                         ynames=['ID/W [uA/um]'],
                         *args, **kwargs)
        num_vg=(vg_range[2]-vg_range[0])/vg_range[1]
        assert abs(num_vg-int(num_vg))<1e-3, f"Make sure the IdVg range gives even steps {str(vg_range)}"
        
        self.vg_range=vg_range
        self.vd_values=vd_values
    
    def get_netlist_str(self):
        setup_part:str=\
            f"parameters vd=0 vg=0\n"\
            f"X0 (netd netg 0 0) {self.model_name} "+\
                self._get_instance_param_part() + "\n"\
            f"VD (netd 0) vsource dc=vd type=dc\n"\
            f"VG (netg 0) vsource dc=vg type=dc\n"
        return setup_part
        
    def get_analyses_str(self):
        analysis_part:str=""
        for i_vd,vd in enumerate(self.vd_values,start=1):
            analysis_part+=f"tovdnum{i_vd} alter param=vd value={vd}\n"
            analysis_part+=f"dcvdnum{i_vd} dc dev=VG param=dc"\
                           f" start={self.vg_range[0]} step={self.vg_range[1]} stop={self.vg_range[2]}\n"
            
        return analysis_part

    def parse_return(self,result):
        parsed_result={}
        for i_vd,vd in enumerate(self.vd_values,start=1):
            for key in result:
                if f'dcvdnum{i_vd}' in key:
                    df=result[key].copy()
                    df['ID/W [uA/um]']=-df['VD:p']/\
                            self.model_paramset.get_total_device_width()
                    df['IG/W [uA/um]']=-df['VG:p']/\
                            self.model_paramset.get_total_device_width()
                    parsed_result[vd]=df.rename(columns=\
                                {'netd':'VD','netg':'VG'})\
                            [['VD','VG','ID/W [uA/um]','IG/W [uA/um]']]
        return parsed_result
        
    def generate_figures(self,*args,**kwargs):
        kwargs['y_axis_type']=kwargs.get('y_axis_type','log')
        return super().generate_figures(*args,**kwargs)

    def _rescale_vector(self,arr):
        return np.log10(np.abs(arr))

class DCIVTemplate(SimTemplate):
    def __init__(self, *args,
                 idvd_vg_values=[0,.6,1.2,1.8], idvd_vd_range=(0,.1,1.8),
                 idvg_vd_values=[.05,1.8], idvg_vg_range=(0,.03,1.8), **kwargs):
        super().__init__(*args, **kwargs)
        self._dcidvg=DCIdVgTemplate(*args, **kwargs,
                                    vd_values=idvg_vd_values, vg_range=idvg_vg_range)
        self._dcidvd=DCIdVdTemplate(*args, **kwargs,
                                    vg_values=idvd_vg_values, vd_range=idvd_vd_range)
        
    def set_paramset(self,model_paramset):
        super().set_paramset(model_paramset)
        self._dcidvg.set_paramset(model_paramset)
        self._dcidvd.set_paramset(model_paramset)
        
    def get_netlist_str(self):
        lines=self._dcidvg.get_netlist_str()
        assert lines==self._dcidvd.get_netlist_str()
        return lines
    def get_analyses_str(self):
        return self._dcidvg.get_analyses_str()+self._dcidvd.get_analyses_str()
        
    def parse_return(self,result):
        result_idvg={k:v for k,v in result.items() if 'VG:dc' in k}
        result_idvd={k:v for k,v in result.items() if 'VD:dc' in k}
        return {'IdVg': self._dcidvg.parse_return(result_idvg),
                'IdVd': self._dcidvd.parse_return(result_idvd)}
    
    def generate_figures(self, meas_data=None, layout_params={}, vizid=None):
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
                 vg_range=(0,.03,1.8), freq='1M', **kwargs):
        super().__init__(outer_variable=None, inner_variable='VG',
                         ynames=['Cgg [fF/um]'],
                         *args, **kwargs)
        
        num_vg=(vg_range[2]-vg_range[0])/vg_range[1]
        assert abs(num_vg-int(num_vg))<1e-3, f"Make sure the IdVg range gives even steps {str(vg_range)}"
        
        self.vg_range=vg_range
        self.freq=freq

    def get_netlist_str(self):
        setup_part:str=\
            f"parameters vg=0\n"\
            f"X0 (0 netg 0 0) {self.model_name} "+\
                self._get_instance_param_part() + "\n"+\
            f"VG (netg 0) vsource dc=vg mag=1 type=dc\n"
        return setup_part
        
    def get_analyses_str(self):
        analysis_part=f"cv ac dev=VG param=dc start={self.vg_range[0]}"\
            f" step={self.vg_range[1]} stop={self.vg_range[2]} freq={self.freq}\n"
        return analysis_part
        
    def parse_return(self,result):
        parsed_result={}
        assert len(result)==1
        freq=spicenum_to_float(self.freq)
        df=list(result.values())[0]
        I=-df['VG:p']
        df['Cgg [fF/um]']=np.imag(I)/(2*np.pi*freq) /1e-15 /\
            (self.model_paramset.get_total_device_width()/1e-6)
        df['VG']=np.real(df['dc'])
        parsed_result={0:df[['VG','Cgg [fF/um]']]}
        return parsed_result


class MultiSimSesh():
    def __init__(self, simtemps: dict[str,SimTemplate]):
        self.simtemps: dict[str,SimTemplate]=simtemps
        self._sessions: dict[str,psp.Session]={}
        
    def __enter__(self):
        print("Opening simulation session(s)")
        assert len(self._sessions)==0, "Previous sessions exist somehow!!"
        
    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing sessions")
        
    def __del__(self):
        if len(self._sessions):
            print("Somehow deleted MultiSimSesh without closing sessions."\
                  "  That's bad but I can try to handle it.")
            self.__exit__(None,None,None)

    def run_with_params(self, params={}):
        raise NotImplementedError

class SpectreMultiSimSesh(MultiSimSesh):

    def __enter__(self):
        super().__enter__()
        for simname,simtemp in self.simtemps.items():
            try:
                sesh=psp.start_session(net_path=simtemp.get_netlist_file())
            except Exception as myexc:
                args=next((k for k in myexc.value.split("\n") if k.startswith("args:")))
                print(" ".join(eval(args.split(":")[1])))
                #import pdb; pdb.set_trace()
                raise
            self._sessions[simname]=sesh
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            for simname in list(self._sessions.keys()):
                psp.stop_session(self._sessions.pop(simname))

    def run_with_params(self, params={}):
        results={}
        #import time
        for simname,sesh in self._sessions.items():
            #print(f"Running {simname}")
            simtemp=self.simtemps[simname]
            re_p_changed=simtemp.update_paramset_and_return_spectre_changes(params)
            #print('setting params',re_p_changed,time.time())
            psp.set_parameters(sesh,re_p_changed)
            #print('running', time.time())
            results[simname]=simtemp.parse_return(psp.run_all(sesh))
            #print('done', time.time())
        return results
        
class PythonMultiSimSesh(MultiSimSesh):
    def run_with_params(self,params={}):
        results={}
        for simname,simtemp in self.simtemps.items():
            re_p_changed=simtemp.update_paramset_and_return_spectre_changes(params)
            results[simname]=python_compact_models[simtemp.model_paramset.model].run_all()
        return results
            