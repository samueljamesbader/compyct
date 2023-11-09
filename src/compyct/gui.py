import panel as pn
from pint import DimensionalityError

from compyct import ureg
from compyct import templates
from compyct.backends.backend import MultiSimSesh
from bokeh.models.formatters import BasicTickFormatter

from compyct.paramsets import spicenum_to_float, ParamPatch

from compyct.util import is_notebook
from bokeh.models.tools import LassoSelectTool
def get_tools():
    # Normally holding SHIFT switches the selection-mode to 'append'... but that doesn't work in Jupyter for some reason
    # according to this bug report https://github.com/bokeh/bokeh/issues/11324
    # So switching the mode to 'append' for Jupyter
    return ['wheel_zoom',
            'pan',
            LassoSelectTool(mode='append' if is_notebook() else 'replace'),
            'reset']

def make_widget(model_paramset, param_name, center):
    true_units=model_paramset.get_units(param_name)
    disp_units=model_paramset.get_display_units(param_name)
    name_with_units=param_name+(f" [{disp_units}]"
                                if (disp_units is not None and disp_units!="") else "")
    bounds=model_paramset.get_bounds(param_name)
    center=spicenum_to_float(center)
    #if bounds[1] is not None and bounds[1]<1e-16: raise Exception(f"Gonna have step error for {param_name}")
    if (dtype:=model_paramset.get_dtype(param_name)) is float:
        try:
            disp_scale=(ureg.parse_expression(true_units)/ureg.parse_expression(disp_units)).to("").magnitude
        except DimensionalityError as e:
            raise Exception(f"Units of {true_units} and Display Units of {disp_units}"\
                            f" for {param_name} are not compatible. Pint says \"{str(e)}\"")
        except Exception as e:
            raise Exception(f"Unit error on {param_name}. Pint says \"{str(e)}\"")
        try:
            #disp_scale=ureg.parse_expression(true_units).to(disp_units).magnitude
            #w=pn.widgets.FloatInput(name=name_with_units,
            #                             start=bounds[0]*disp_scale,
            #                             step=bounds[1]*disp_scale,
            #                             value=(center if center is not None\
            #                                 else spicenum_to_float( model_paramset.get_default(param_name)))*disp_scale,
            #                             sizing_mode='stretch_width',
            #                             format=BasicTickFormatter(precision=5)
            #                        )
            value=(center if center is not None\
                else spicenum_to_float( model_paramset.get_default(param_name)))*disp_scale
            w=pn.widgets.TextInput(name=name_with_units,value=f"{value:.5g}",sizing_mode='stretch_width')
            return w,disp_scale
        except:
            print(name_with_units,bounds,center)
            raise
    elif dtype is int:
        assert true_units=='', "Units should be none for integer parameter"
        assert disp_units=='', "Units should be none for integer parameter"
        if bounds[0] is not None and bounds[2] is not None:
            #print(f'hi there {param_name} {bounds} {center}')
            w=pn.widgets.Select(name=name_with_units,
                                options=list(range(int(bounds[0]),int(bounds[2]+1))),
                                value=int(center),
                                sizing_mode='stretch_width'
                                )
            return w,1
        else:
            raise Exception(f"What to do for {param_name}?")
    else:
        raise Exception(f"What is dtype {dtype} for {param_name}?")


@staticmethod
def fig_legend_config(fig,location='bottom_right'):
    fig.legend.margin=0
    fig.legend.spacing=0
    fig.legend.padding=4
    fig.legend.label_text_font_size='8pt'
    fig.legend.label_height=10
    fig.legend.label_text_line_height=10
    fig.legend.glyph_height=10
    fig.legend.location=location

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