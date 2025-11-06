import time
from datetime import datetime
from pathlib import Path
import os
from operator import itemgetter
import pickle

import bokeh.io
import numpy as np
import yaml

from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError
from compyct import logger, set_logging_callback, unset_logging_callback
from compyct.paramsets import spicenum_to_float, ParamPatch
from compyct.backends.backend import MultiSimSesh, SimulatorCommandException
from scipy.optimize import curve_fit
from typing import Any, Optional, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from compyct.templates import SimTemplate, TemplateGroup
from contextlib import contextmanager


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

    output_dir: Optional[str|Path] = None

    def __post_init__(self):
        assert self.output_dir is not None
        if any('|||' in k for k in self.global_template_group):
            self.major_tabnames=list(dict.fromkeys(k.split("|||")[0] for k in self.global_template_group))
        else: self.major_tabnames=['']
        print(self.major_tabnames)
        self._tabbed_template_groups: dict[str,TemplateGroup] = {k:v for k,v in \
                {f"{mt}|||{tabname}": self.global_template_group\
                        .only(*[f"{mt}|||{tn}" for tn in tempnames],error_if_missing=False)\
                    for mt in self.major_tabnames for tabname,tempnames in self.tabbed_templates.items()}\
            .items() if len(v)>0}
        print(self._tabbed_template_groups.keys())
            
        self._mss=MultiSimSesh.get_with_backend(
                    {k:v for k,v in self.global_template_group.items() if isinstance(v,SimTemplate)},
                    backend=self.backend,**self.backend_kwargs)

    def load(self, save_name, rerun=True):
        with open(self.output_dir/(save_name+".yaml"),'r') as f:
            saved=yaml.safe_load(f)
        # with open(self.output_dir/(save_name+".pkl"),'rb') as f:
        #     saved=pickle.load(f)
        self.global_patch.update({k:v for k,v in saved['global_values'].items() if k in self.global_patch})
        if rerun:
            self.rerun(None)
        return saved['additional']

    def save(self, save_name, additional=None):
        with open(self.output_dir/(save_name+".yaml"),'w') as f:
            yaml.dump({'global_values':dict(self.global_patch),'additional':additional},f)
        # with open(self.output_dir/(save_name+".pkl"),'wb') as f:
        #     pickle.dump({'global_values':dict(self.global_patch),'additional':additional},f)

    def start_sesh(self):
        for _,t in self.global_template_group.items():
            t.apply_patch(self.global_patch)
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
            pvs={a:v*self.global_patch.get_scale(a) for a,v in zip(actives,args)}
            self.global_patch.update(pvs)
            return self._tabbed_template_groups[tabname].parsed_results_to_vector(
                mss.run_with_params(self.global_patch,
                                    only_temps=self._tabbed_template_groups[tabname],),
                    self.tabbed_rois[tabname], meas_parsed_results=self.global_meas_data)

        bounds=list(zip(*[np.array(self.global_patch.get_bounds(a,null=np.inf))[[0,2]]/self.global_patch.get_scale(a) for a in actives]))
        #bounds=[[(b if b is not None else np.inf) for b in bi] for bi in bounds]
        try:
            with self.ensure_within_sesh() as mss:
                meas_vector=self._tabbed_template_groups[tabname].parsed_results_to_vector(
                    self.global_meas_data,self.tabbed_rois[tabname],meas_parsed_results=self.global_meas_data)

                p0=[spicenum_to_float(self.global_patch[a])/self.global_patch.get_scale(a) for a in actives]
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

    def rerun(self,temps:list[str]|None):
        with self.ensure_within_sesh() as mss:
            return rerun_with_params(patch=self.global_patch,temps=temps,global_template_group=self.global_template_group, mss=mss)

def rerun_with_params(patch:ParamPatch|None, temps:list[str]|None, global_template_group: TemplateGroup, mss: MultiSimSesh):
    new_results={}
    remaining_temps=set(temps) if temps is not None else set(global_template_group.keys())
    while len(remaining_temps):
        ready_to_run=[tn for tn in remaining_temps if all(global_template_group.name_of_template(dt) not in remaining_temps
                                                          for dt in global_template_group[tn].dependencies)]
        logger.debug(f"Rerun remaining temps: {remaining_temps}, ready to run: {ready_to_run}")
        simtemps=[tn for tn in ready_to_run if isinstance(global_template_group[tn],SimTemplate)]
        nonsimtemps=[tn for tn in ready_to_run if not isinstance(global_template_group[tn],SimTemplate)]
        new_results|=mss.run_with_params(patch, only_temps=simtemps)
        new_results|={tn:global_template_group[tn].extract() for tn in nonsimtemps}
        for tn in ready_to_run:
            global_template_group[tn].update_sim_results(new_results.get(tn,None))
        remaining_temps.difference_update(ready_to_run)
    return new_results


import panel as pn
from panel.widgets import CompositeWidget
from compyct.gui import make_widget

class SemiAutoOptimizerGui(CompositeWidget):

    _sao: SemiAutoOptimizer
    def __init__(self, sao: SemiAutoOptimizer, save_name: str = 'sim_save', fig_layout_params={},load_before_run=False,**kwargs):
        super().__init__(height=600,sizing_mode='stretch_width',**kwargs)
        self._sao=sao
        self._default_save_name=save_name
        self._fig_layout_params=fig_layout_params
        self._should_respond_to_param_widgets=True
        self._composite[:]=[self.make_gui()]

        self.start_sesh()
        #self._load()
        self._needs_rerun={stn:True for stn in self.global_template_group}
        if load_before_run:
            self._load_button_pressed(None)
        self.rerun_and_update_tab(self._active_tab)


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
        self._widgets={}
        self._checkboxes={}
        self._wlines={}
        self._hovers={}
        self._wcols={}
        self._tabname_to_vizid={}
        self._param_to_widgets={}
        major_tabs=[]
        self._mtnotno_to_tabn={}

        vizid_offset=round(datetime.now().timestamp()*1e3)
        for imt,mt in enumerate(self._sao.major_tabnames):
            tabs=[]
            for vizid_, (tabname, tg) in enumerate([(tabname,tg) for tabname,tg in self._tabbed_template_groups.items() if tabname.startswith(f"{mt}|||")]):
                vizid=vizid_offset+vizid_
                self._tabname_to_vizid[tabname]=vizid
                fp=tg.get_figure_pane(fig_layout_params=self._fig_layout_params,vizid=vizid)
                params=self.tabbed_actives.get(tabname.split("|||")[-1],[])
                self._widgets[tabname]={param:make_widget(self.global_patch.param_set, param, self.global_patch[param])
                                        for param in params}
                self._hovers[tabname]={param:pn.widgets.Button(name='?') for param in params}
                for p,h in self._hovers[tabname].items():
                    h.on_click(lambda e,p=p: logger.info(f"{p}: {self.global_patch.get_description(p)}"))
                self._checkboxes[tabname]={
                    param:pn.widgets.Checkbox(value=(self.global_patch.get_dtype(param)==float),
                                              width=5,sizing_mode='stretch_height',
                                              disabled=(self.global_patch.get_dtype(param)!=float))\
                        for param in params}
                self._wlines[tabname]={p:pn.Row(pn.Column(
                                                 pn.VSpacer(),pn.VSpacer(),c,pn.VSpacer(),width=15,height=60),w,h)
                                       for (p,c),w,h in zip(self._checkboxes[tabname].items(),
                                                          self._widgets[tabname].values(),
                                                          self._hovers[tabname].values())}
                for param,w in self._widgets[tabname].items():
                    w.param.watch((lambda event, tabname=tabname, param=param: self._updated_widget(tabname,param)),'value')

                p_by_cat=self._sao.global_patch.param_set.get_categorized(params)
                for p, w in self._widgets[tabname].items():
                    self._param_to_widgets[p]=self._param_to_widgets.get(p,[])+[w]
                #pse=self._sao.global_patch._ps_example
                #for p in self._wlines[tabname]:
                #    cat=pse._shared_paramdict[p].get("category","Misc")
                #    if cat not in p_by_cat: p_by_cat[cat]=[]
                #    p_by_cat[cat].append(p)
                #for cat in p_by_cat:
                #    p_by_cat[cat]=list(sorted(p_by_cat[cat],key=lambda p: ((pse.get_dtype(p) is not int), p)))
                self._wcols[tabname]=pn.Column(pn.layout.Accordion(
                        *[(cat,pn.Column(*(self._wlines[tabname][p] for p in p_by_cat[cat])))
                            for cat in sorted(p_by_cat)],
                        active=(list(range(len(p_by_cat))) if len(p_by_cat)<=2 else []),
                        sizing_mode='fixed',
                        width=175,),
                        width=200,sizing_mode='stretch_height',scroll=True)

                #self._wcols[tabname]=pn.Column(acc,
                                         #width=175,sizing_mode='stretch_height',scroll=True)
                content=pn.Row(self._wcols[tabname],fp)

                tabs.append((tabname.split("|||")[-1],content))
                self._mtnotno_to_tabn[(imt,vizid_)]=tabname
            tabs=pn.Tabs(*tabs,height=350)
            tabs.param.watch(self._tab_changed,['active'])
            major_tabs.append((mt,tabs))

        self._major_tabs=pn.Tabs(*major_tabs,height=400)
        sesh_button=self._sesh_button=pn.widgets.Button(name="Start Sesh")
        opt_button=pn.widgets.Button(name="Optimize tab")
        running_ind=self.running_ind=pn.widgets.LoadingSpinner(value=False,size=25)
        save_name_input=self._save_name_input=pn.widgets.TextInput(value=self._default_save_name,width=200)
        save_button=pn.widgets.Button(name="Save")
        load_button=pn.widgets.Button(name="Load")

        sesh_button.on_click(self._sesh_button_pressed)
        opt_button.on_click(self._opt_button_pressed)
        save_button.on_click(self._save_button_pressed)
        load_button.on_click(self._load_button_pressed)
        self._major_tabs.param.watch(self._tab_changed,['active'])
        self._log_view=pn.widgets.TextAreaInput(sizing_mode='stretch_both')

        self.redo_widget_visibility()
        return pn.Column(pn.Row(sesh_button,opt_button,running_ind,pn.HSpacer(),save_name_input,save_button,load_button,sizing_mode='stretch_width'),
                         self._major_tabs,self._log_view,height=550)

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
        return self._mtnotno_to_tabn[(self._major_tabs.active,self._major_tabs[self._major_tabs.active].active)]

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
        self.rerun_and_update_tab(active)
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
        self.rerun_and_update_tab(self._active_tab)

    def _save_button_pressed(self, event):
        save_name=self._save_name_input.value if hasattr(self,'_save_name_input') else self.default_save_name
        self._sao.save(save_name,additional={'activated':{tn:{a:check.value for a,check in checks.items()}
                                                          for tn, checks in self._checkboxes.items()}})

    def reset_all_figures(self, except_for_tabname=None):
        for stname,st in self.global_template_group.items():
            st.update_sim_results(None)
            self._needs_rerun[stname]=True
        for tabn,vizid in self._tabname_to_vizid.items():
            if except_for_tabname==tabn: continue
            for stname,st in self._tabbed_template_groups[tabn].items():
                st.update_figures(vizid=vizid)

    def _tab_changed(self, event):
        self.rerun_and_update_tab(self._active_tab)
        #tg=self._tabbed_template_groups[self._active_tab]
        #simtemps=[tn for tn,t in tg.items() if isinstance(t,SimTemplate) and self._needs_rerun[tn]]
        #self.rerun(simtemps)
        #for tname,t in tg.items():
        #    t.update_figures(vizid=self._tabname_to_vizid[self._active_tab])

    def _updated_widget(self,tabname,param):
        if self._should_respond_to_param_widgets:
            logger.debug(f'Responding to updated widget {param}')
            #print('updated widget')
            if self._active_tab==tabname:
                #import pdb; pdb.set_trace()
                values={n:float(w.value)/self.global_patch.get_display_scale(n) for n,w in self._widgets[tabname].items()}
                self.global_patch.update(values)
                logger.debug(f'Re-updating widgets from patch')
                #self.update_widgets_from_global_patch(only_param="YYYYYYYYYYYY")
                self.update_widgets_from_global_patch(only_param=param) # Can hopefully skip this, except would need to update visibility manually
                logger.debug(f'about to reset figures')
                self.reset_all_figures(except_for_tabname=tabname)
                logger.debug(f'about to rerun')
                self.rerun_and_update_tab(tabname)
                logger.debug(f'Done')

    def update_widgets_from_global_patch(self,only_param=None):
        #print("Updating widgets from global patch")
        with self.dont_trigger_param_widgets():
            params=[only_param] if only_param else self._param_to_widgets.keys()
            for param in params:
                w0=self._param_to_widgets[param][0]
                dscale=self.global_patch.get_display_scale(param)
                if isinstance(w0,pn.widgets.TextInput):
                    value=f"{spicenum_to_float(self.global_patch[param])*dscale:.5g}"
                else:
                    value=spicenum_to_float(self.global_patch[param])*dscale
                for widget in self._param_to_widgets[param]:
                    widget.value=value
        self.redo_widget_visibility()

    def redo_widget_visibility(self):
        logger.info(f"starting redo vis")
        #bokeh.io.curdoc().hold()
        #try:
        for t,wlines in self._wlines.items():
            for param, wline in wlines.items():
                # standard invisible method
                #wline.visible=(not self._sao.global_patch.is_param_irrelevant(param))
                # hack invisible method
                #wline.styles={'display':{None:'none',False:'none',True:'flex'}[
                #                            not self._sao.global_patch.is_param_irrelevant(param)]}
                # Disable method
                wline.objects[1].disabled=(self._sao.global_patch.is_param_irrelevant(param) is True)
                wline.objects[0].objects[2].disabled=(self._sao.global_patch.is_param_irrelevant(param) is True)
        #finally:
        #    bokeh.io.curdoc().unhold()
        logger.info(f"done redo vis")

    def _rerun(self,simtemps:list[str]):
        if not len(simtemps): return
        try:
            self.running_ind.value=True
            new_results=self._sao.rerun(simtemps)
        except NgSpiceCommandError as e:
            new_results=None
            logger.warning(f"Ngspice run error: {e}")
        else:
            logger.info(f"Simulation run complete {list(new_results.keys())}")
        finally:
            self.running_ind.value=False
        if new_results is None: return
        for stname in simtemps:
            self._needs_rerun[stname]=False
    
    def rerun_and_update_tab(self, tabname):
        to_rerun={tn:t for tn,t in self._tabbed_template_groups[tabname].items() if self._needs_rerun[tn]}
        while True:
            adds={self.global_template_group.name_of_template(dt):dt for t in to_rerun.values() for dt in t.dependencies}
            new_tos={tn:t for tn,t in adds.items() if self._needs_rerun[tn] and (tn not in to_rerun)}
            if not len(new_tos): break
            to_rerun.update(new_tos)
        self._rerun(list(to_rerun))
        #self.rerun([tn for tn,t in self._tabbed_template_groups[tabname].items() if isinstance(t,SimTemplate)])
        vizid=self._tabname_to_vizid[tabname]
        for tname,t in self._tabbed_template_groups[tabname].items():
            t.update_figures(vizid=vizid)
