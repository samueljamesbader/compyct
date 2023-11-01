import panel as pn
from compyct import templates
from compyct.backends.backend import MultiSimSesh

from compyct.paramsets import spicenum_to_float, ParamPatch


def make_widget(model_paramset, param_name, center):
    units=model_paramset.get_units(param_name)
    name_with_units=param_name+(f" [{units}]"
                                if (units is not None and units!="") else "")
    bounds=model_paramset.get_bounds(param_name)
    center=spicenum_to_float(center)
    try:
        return pn.widgets.FloatInput(name=name_with_units,
                                     start=bounds[0],
                                     step=bounds[1],
                                     value=(center if center is not None\
                                         else spicenum_to_float( model_paramset.get_default(param_name))),
                                     sizing_mode='stretch_width')
    except:
        print(name_with_units,bounds,center)
        raise


class ManualOptimizer(pn.widgets.base.CompositeWidget):
    def __init__(self, mss:MultiSimSesh=None, user_paramset=None,
                 active_params={}, meas_data={}, fig_layout_params={}):
        super().__init__(height=300,sizing_mode='stretch_width')
        self.mss=mss
        self.user_paramset=user_paramset
        self._figpane=mss.simtemps.get_figure_pane(meas_data=meas_data,fig_layout_params=fig_layout_params)
        self._widgets={param:make_widget(user_paramset,param,center)
                       for param,center in active_params.items()}
        for w in self._widgets.values():
            w.param.watch(self._widget_updated,'value')
        self._composite[:]=[
            pn.Column(*self._widgets.values(),width=100,
                      sizing_mode='stretch_height',scroll=True),
            self._figpane]
        self._widget_updated()

    def get_widget_params(self):
        return {n:float(w.value) for n,w in self._widgets.items()}

    def _widget_updated(self,*args,**kwargs):
        self.rerun_and_update(params=ParamPatch(self.user_paramset,**self.get_widget_params()))
        
    def rerun_and_update(self,params={}):
        new_results=self.mss.run_with_params(params=params)
        #import pdb; pdb.set_trace()
        self.mss.simtemps.update_figures(new_results)