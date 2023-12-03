import time
from datetime import datetime
from pathlib import Path
import os
from operator import itemgetter
import pickle

import bokeh.io
import numpy as np

from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError
from compyct import logger, set_logging_callback, unset_logging_callback
from compyct.paramsets import spicenum_to_float, ParamPatch
from compyct.backends.backend import MultiSimSesh, SimulatorCommandException
from scipy.optimize import curve_fit
from typing import Any
from copy import deepcopy
from dataclasses import dataclass, field
from compyct.templates import TemplateGroup
from contextlib import contextmanager


OUTPUT_DIR=Path(os.environ['MODELFIT_HOME'])/'output'

class OptimizationException(Exception):
    def __init__(self, bounds, latest, sim_err):
        self.bounds=bounds
        self.latest=latest
        self.sim_err=sim_err
        super().__init__("Optimization Failed")

@dataclass
class SemiAutoOptimizer():

    global_template_group: TemplateGroup
    global_patch: ParamPatch
    global_meas_data: dict[str,Any]

    tabbed_templates: dict[str,str]
    tabbed_actives: dict[str,list[str]]
    tabbed_rois: dict[str,Any]
    backend: str = 'ngspice'
    backend_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self._tabbed_template_groups: dict[str,TemplateGroup] =\
            {tabname: self.global_template_group.only(*tempnames)\
                 for tabname,tempnames in self.tabbed_templates.items()}
        self._mss=MultiSimSesh.get_with_backend(self.global_template_group, backend=self.backend,**self.backend_kwargs)

    def load(self, save_name, rerun=True):
        with open(OUTPUT_DIR/(save_name+".pkl"),'rb') as f:
            saved=pickle.load(f)
        self.global_patch.update(saved['global_values'])
        if rerun:
            self.rerun(None)
        return saved['additional']

    def save(self, save_name, additional=None):
        with open(OUTPUT_DIR/(save_name+".pkl"),'wb') as f:
            pickle.dump({'global_values':dict(self.global_patch),'additional':additional},f)

    def start_sesh(self):
        self._mss.__enter__()

    def end_sesh(self):
        self._mss.__exit__(None,None,None)

    @contextmanager
    def ensure_within_sesh(self):
        if self._mss.is_entered:
            yield self._mss
        else:
            with self._mss as mss:
                yield mss

    def optimize_tab(self, tabname, only_actives=None):
        backup_values=self.global_patch.copy()
        actives=only_actives if only_actives else self.tabbed_actives[tabname]
        #print(f"optimizing with {actives}")
        def _f(mss,*args):
            pvs={a:v*self.global_patch._ps_example.get_scale(a) for a,v in zip(actives,args)}
            self.global_patch.update(pvs)
            return self._tabbed_template_groups[tabname].parsed_results_to_vector(
                mss.run_with_params(self.global_patch,
                                    only_temps=self._tabbed_template_groups[tabname],),
                    self.tabbed_rois[tabname], meas_parsed_results=self.global_meas_data)

        bounds=list(zip(*[np.array(self.global_patch._ps_example.get_bounds(a,null=np.inf))[[0,2]]/self.global_patch._ps_example.get_scale(a) for a in actives]))
        #bounds=[[(b if b is not None else np.inf) for b in bi] for bi in bounds]
        try:
            with self.ensure_within_sesh() as mss:
                meas_vector=self._tabbed_template_groups[tabname].parsed_results_to_vector(
                    self.global_meas_data,self.tabbed_rois[tabname],meas_parsed_results=self.global_meas_data)

                p0=[spicenum_to_float(self.global_patch[a])/self.global_patch._ps_example.get_scale(a) for a in actives]
                #print(p0)
                #print(f"Initial args {dict(zip(actives,p0))}")
                #print(f"Meas vec {meas_vector}")
                #print(f"Initial vec {_f(mss,*p0)}")
                try:
                    res=curve_fit(_f,mss,ydata=meas_vector,p0=p0,bounds=bounds,full_output=True)
                except SimulatorCommandException as e:
                    bounds_dict=dict(zip(actives,zip(*bounds)))
                    latest={a:self.global_patch[a] for a in actives}
                    raise OptimizationException(bounds=bounds_dict,latest=latest, sim_err=e)
                except ValueError as e:
                    if 'Each lower bound must be strictly less than each upper bound' in str(e):
                        bounds_dict=dict(zip(actives,zip(*bounds)))
                        e.bounds=bounds_dict
                        #print(bounds_dict)
                        raise
                    elif 'array must not contain infs or NaNs' in str(e):
                        bounds_dict=dict(zip(actives,zip(*bounds)))
                        latest={a:self.global_patch[a] for a in actives}
                        raise OptimizationException(bounds=bounds_dict,latest=latest, sim_err=e)
                    elif '`x0` is infeasible.' in str(e):
                        bounds_dict=dict(zip(actives,zip(*bounds)))
                        latest={a:self.global_patch[a] for a in actives}
                        raise OptimizationException(bounds=bounds_dict,latest=latest, sim_err=e)
                    else:
                        raise

        except OptimizationException:
            self.global_patch.update(backup_values)
            raise

    def rerun(self,tabname):
        # Ignoring tabname and rerunning-updating all
        #assert tabname is None
        with self.ensure_within_sesh() as mss:
            new_results=mss.run_with_params(self.global_patch,
                    only_temps=(self._tabbed_template_groups[tabname] if tabname else None))
        return new_results


import panel as pn
from panel.widgets import CompositeWidget
from compyct.gui import make_widget

class SemiAutoOptimizerGui(CompositeWidget):

    def __init__(self, sao: SemiAutoOptimizer, save_name: str = 'sim_save', fig_layout_params={},**kwargs):
        super().__init__(height=600,sizing_mode='stretch_width',**kwargs)
        self._sao=sao
        self._default_save_name=save_name
        self._fig_layout_params=fig_layout_params
        self._should_respond_to_param_widgets=True
        self._composite[:]=[self.make_gui()]

        self.start_sesh()
        #self._load()
        self._needs_rerun={tn:True for tn in self._tabbed_template_groups}
        self.rerun_and_update(self._active_tab)


    def start_sesh(self):
        self._sao.start_sesh()
        set_logging_callback(self._logging_callback)
        if hasattr(self,'_sesh_button'):
            self._sesh_button.name='End Sesh'

    def end_sesh(self):
        self._sao.end_sesh()
        unset_logging_callback()
        if hasattr(self,'_sesh_button'):
            self._sesh_button.name='Start Sesh'

    def _logging_callback(self, record, formatter):
        self._log_view.value+=formatter.format(record)+"\n"

    def __getattr__(self,attr):
        if attr in ['global_template_group','global_patch','global_meas_data',
                    'tabbed_templates','tabbed_actives','tabbed_rois',
                    '_tabbed_template_groups','ensure_within_sesh']:
            return getattr(self._sao,attr)
        else:
            raise AttributeError(attr)

    def make_gui(self):
        tabs=[]
        self._widgets={}
        self._checkboxes={}
        self._wlines={}
        self._wcols={}
        self._tabname_to_vizid={}

        vizid_offset=round(datetime.now().timestamp()*1e3)
        for vizid_, (tabname, tg) in enumerate(self._tabbed_template_groups.items()):
            vizid=vizid_offset+vizid_
            self._tabname_to_vizid[tabname]=vizid
            fp=tg.get_figure_pane(meas_data=self.global_meas_data,fig_layout_params=self._fig_layout_params,vizid=vizid)

            self._widgets[tabname]={param:make_widget(self.global_patch._ps_example, param, self.global_patch[param])
                                    for param in self.tabbed_actives[tabname]}
            self._checkboxes[tabname]={
                param:pn.widgets.Checkbox(value=(self.global_patch._ps_example.get_dtype(param)==float),
                                          width=5,sizing_mode='stretch_height',
                                          disabled=(self.global_patch._ps_example.get_dtype(param)!=float))\
                    for param in self.tabbed_actives[tabname]}
            self._wlines[tabname]={p:pn.Row(pn.Column(
                                             pn.VSpacer(),pn.VSpacer(),c,pn.VSpacer(),width=15,height=60),w)
                                   for (p,c),w in zip(self._checkboxes[tabname].items(),
                                                      [w for w,_ in self._widgets[tabname].values()])}
            for param,(w,_) in self._widgets[tabname].items():
                w.param.watch((lambda event, tabname=tabname, param=param: self._updated_widget(tabname,param)),'value')
            self._wcols[tabname]=pn.Column(*self._wlines[tabname].values(),
                                     width=175,sizing_mode='stretch_height',scroll=True)
            content=pn.Row(self._wcols[tabname],fp)

            tabs.append((tabname,content))

        self._tabs=pn.Tabs(*tabs,height=350)
        sesh_button=self._sesh_button=pn.widgets.Button(name="Start Sesh")
        opt_button=pn.widgets.Button(name="Optimize tab")
        running_ind=self.running_ind=pn.widgets.LoadingSpinner(value=False,size=25)
        save_name_input=self._save_name_input=pn.widgets.TextInput(value=self._default_save_name,width=100)
        save_button=pn.widgets.Button(name="Save")
        load_button=pn.widgets.Button(name="Load")

        sesh_button.on_click(self._sesh_button_pressed)
        opt_button.on_click(self._opt_button_pressed)
        save_button.on_click(self._save_button_pressed)
        load_button.on_click(self._load_button_pressed)
        self._tabs.param.watch(self._tab_changed,['active'])
        self._log_view=pn.widgets.TextAreaInput(sizing_mode='stretch_both')

        self.redo_widget_visibility()
        return pn.Column(pn.Row(sesh_button,opt_button,running_ind,pn.HSpacer(),save_name_input,save_button,load_button,sizing_mode='stretch_width'),
                         self._tabs,self._log_view,height=550)

    @contextmanager
    def dont_trigger_param_widgets(self):
        should_respond=self._should_respond_to_param_widgets
        self._should_respond_to_param_widgets=False
        try:
            yield
        finally:
            self._should_respond_to_param_widgets=should_respond

    @property
    def _active_tab(self):
        return list(self._tabbed_template_groups)[self._tabs.active]

    def _sesh_button_pressed(self, event):
        if self._sao._mss.is_entered:
            self.end_sesh()
        else:
            self.start_sesh()

    def _opt_button_pressed(self,event):
        self.reset_all_figures()
        active=self._active_tab
        try:
            self.running_ind.value=True
            #print('only_actives',[a for a in self.tabbed_actives[active] if self._checkboxes[active][a].value])
            self._sao.optimize_tab(active, only_actives=[a for a in self.tabbed_actives[active] if self._checkboxes[active][a].value])
        except OptimizationException as e:
            logger.debug(f"Bounds: {e.bounds}")
            logger.debug(f"Latest: {e.latest}")
            logger.debug(f"Sim Err: {e.sim_err}")
            self._latest_error=e
            raise e
        except Exception as e:
            self._latest_error=e
            raise e
        finally:
            self.running_ind.value=False
        self.rerun_and_update(active)
        self.update_widgets_from_global_patch()

    def _load_button_pressed(self, event):
        self.reset_all_figures()
        save_name=self._save_name_input.value if hasattr(self,'_save_name_input') else self.default_save_name
        try:
            additional=self._sao.load(save_name,rerun=False)
            #print(additional)
            self.update_widgets_from_global_patch()
            for tn, checks in self._checkboxes.items():
                for a, check in checks.items():
                    if (saved_activation:=additional['activated'].get(tn,{}).get(a,None)) is not None:
                        check.value=saved_activation
        except Exception as e:
            logger.debug("Error loading from previous save:")
            logger.debug(str(e))
        self.rerun_and_update(self._active_tab)

    def _save_button_pressed(self, event):
        save_name=self._save_name_input.value if hasattr(self,'_save_name_input') else self.default_save_name
        self._sao.save(save_name,additional={'activated':{tn:{a:check.value for a,check in checks.items()}
                                                          for tn, checks in self._checkboxes.items()}})

    def reset_all_figures(self, except_for=None):
        for tn,vizid in self._tabname_to_vizid.items():
            if except_for==tn: continue
            self._needs_rerun[tn]=True
            self._tabbed_template_groups[tn].update_figures(None,vizid=vizid)


    def _tab_changed(self, event):
        if self._needs_rerun[self._active_tab]:
            self.rerun_and_update(self._active_tab)

    def _updated_widget(self,tabname,param):
        if self._should_respond_to_param_widgets:
            logger.debug(f'Responding to updated widget {param}')
            #print('updated widget')
            if self._active_tab==tabname:
                #import pdb; pdb.set_trace()
                values={n:float(w.value)/dscale for n,(w,dscale) in self._widgets[tabname].items()}
                self.global_patch.update(values)
                self.update_widgets_from_global_patch(only_param=param)
                logger.debug(f'about to reset figures')
                self.reset_all_figures(except_for=tabname)
                logger.debug(f'about to rerun')
                self.rerun_and_update(tabname)
                logger.debug(f'Done')

    def update_widgets_from_global_patch(self,only_param=None):
        #print("Updating widgets from global patch")
        with self.dont_trigger_param_widgets():
            for t,widgets in self._widgets.items():
                for param in widgets:
                    if (only_param is None) or (param==only_param):
                        w,dscale=widgets[param]
                        if isinstance(w,pn.widgets.TextInput):
                            w.value=f"{spicenum_to_float(self.global_patch[param])*dscale:.5g}"
                        else:
                            w.value=spicenum_to_float(self.global_patch[param])*dscale
        self.redo_widget_visibility()

    def redo_widget_visibility(self):
        logger.info(f"starting redo vis")
        #bokeh.io.curdoc().hold()
        #try:
        for t,wlines in self._wlines.items():
            for param, wline in wlines.items():
                wline.styles={'display':{None:'none',False:'none',True:'flex'}[
                                            not self._sao.global_patch.is_param_irrelevant(param)]}
                #wline.visible=(not self._sao.global_patch.is_param_irrelevant(param))
        #finally:
        #    bokeh.io.curdoc().unhold()
        logger.info(f"done redo vis")

    def rerun_and_update(self,tabname):
        try:
            self.running_ind.value=True
            new_results=self._sao.rerun(tabname)
            run_error=False
        except NgSpiceCommandError as e:
            new_results=None
            run_error=True
        finally:
            self.running_ind.value=False
        for tn,vizid in self._tabname_to_vizid.items():
            if ((tabname is None) or (tn==tabname)):
                self._needs_rerun[tn]=False
                self._tabbed_template_groups[tn].update_figures(new_results,vizid=vizid)
