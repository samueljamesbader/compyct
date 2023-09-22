import panel as pn
from compyct import sim
import bokeh.layouts

def make_widget(model_paramset, param_name, center):
    units=model_paramset.get_units(param_name)
    name_with_units=param_name+(f" [{units}]"
                                if (units is not None and units!="") else "")
    bounds=model_paramset.get_bounds(param_name)
    return pn.widgets.FloatInput(name=name_with_units,
                                 start=bounds[0],
                                 step=bounds[1],
                                 value=(center if center is not None\
                                     else spicenum_to_float( model_paramset.get_default(param_name))),
                                 sizing_mode='stretch_width')

class ManualOptimizer(pn.widgets.base.CompositeWidget):
    def __init__(self, mss:sim.MultiSimSesh=None, model_paramset=None,
                 active_paramset={},meas_data=None, fig_layout_params={}):
        super().__init__(height=300,sizing_mode='stretch_width')
        self.mss=mss
        self.paramset=active_paramset
        self._figs=sum([t.generate_figures(
                            meas_data=meas_data.get(stname,None),
                            layout_params=fig_layout_params)
                       for stname,t in self.mss.simtemps.items()],[])
        self._widgets={param:make_widget(model_paramset,param,center)
                        for param,center in active_paramset.items()}
        for w in self._widgets.values():
            w.param.watch(self._widget_updated,'value')
        self._composite[:]=[
            pn.Column(*self._widgets.values(),width=100,
                      sizing_mode='stretch_height',scroll=True),
            bokeh.layouts.gridplot([self._figs])]
        self._widget_updated()

    def _widget_updated(self,*args,**kwargs):
        model_params={n:w.value for n,w in self._widgets.items()}
        self.rerun_and_update(params=model_params)
        
    def rerun_and_update(self,params={}):
        new_results=self.mss.run_with_params(params=params)
        #import pdb; pdb.set_trace()
        for stname in self.mss.simtemps:
            self.mss.simtemps[stname].update_figures(new_results[stname])
        pn.io.push_notebook(self)