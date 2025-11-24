from __future__ import annotations
from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Optional
from compyct.backends.backend import get_va_path
from compyct.paramsets import GuessedSubcktParamSet, ParamPatch, ParamSet
from compyct.templates import TemplateGroup
from compyct import OUTPUT_DIR, SAVE_DIR, logger
import panel as pn; pn.extension()
#pn.config.theme = 'dark' # type: ignore
if TYPE_CHECKING:
    from compyct.optimizer import SemiAutoOptimizerGui
    from compyct.backends.backend import ModelCardWriter

class ModelSuite():
    playback_ps_class:type[GuessedSubcktParamSet]

    def __init__(self, element_name:str, release_name:str,
                 submodel_split:dict[str, str|None]={'all': None},
                 instance_subset_names:dict[str, list[str]]={},
                 measurement_subset_names:dict[str, list[str]]={},
                 ):
        """A model and the associated templates for comparing it with data.

        Subclasses should implement get_template_group_explicit to return
        the appropriate TemplateGroup for the suite.
        
        Args:
            element_name: The name of the model as it should appear in the PDK modelcard file
            release_name: The name of the model release the suite belongs to (used for naming saves)
            submodel_split: A dictionary mapping submodel split names (e.g. "short-gate") to the
                conditions that define them in modelcard splits (e.g. "l<30e-9").
                A key that maps to None indicates the default condition (and must be present).
            instance_subset_names: A dictionary mapping instance subset names to lists of instances
                names to measure/simulate (an instance name identifies a set of device parameters).
            measurement_subset_names: A dictionary mapping measurement subset names to lists of
                measurement names (templates) to measure/simulate
        """
        self.element_name = element_name
        self.release_name = release_name
        self.submodel_split = submodel_split
        self.instance_subset_names = instance_subset_names
        self.measurement_subset_names = measurement_subset_names
        for k in instance_subset_names.keys():
            assert '-' not in k, "Subset names cannot contain hyphens"
        for k in measurement_subset_names.keys():
            assert '-' not in k, "Subset names cannot contain hyphens"
        for k in submodel_split.keys():
            assert '-' not in k, "Submodel split names cannot contain hyphens"
        if len(submodel_split) > 1:
            assert 'all' not in submodel_split.keys(), "'all' is a reserved submodel split name"
    
    def get_template_group(self, param_set:ParamSet, submodel_split_name:str='all',
                           instance_subset_name:Optional[str]=None,
                           measurement_subset_name:Optional[str]=None,
                           force_refresh_data:bool=False,
                           ) -> TemplateGroup:
        """Get a TemplateGroup for the specified subsets.

        Args:
            param_set: ParamSet the templates should use
            submodel_split_name: The name of the submodel split to use.
                Doesn't have to be in the listed splits (convenient for e.g. experimental fits
                to arbitrary conditions).  
            instance_subset_name: The name of the instance subset to use
            measurement_subset_name: The name of the measurement subset to use
            force_refresh_data: If True, forces calling get_data even if cached data exists.
        Returns:
            A TemplateGroup for the specified subsets.

        Note: if instance_subset_name is None, will fall back to the
            self.instance_subset_names[subset_split_name] if said key exists, or None otherwise.
            Similar for measurement_subset_name.
        """
        from compyct import CACHE_DIR
        cname=("native" 
            if hasattr(self,'param_set') and self.param_set == param_set # type: ignore
            else "playback")
        cache_path=CACHE_DIR/f"tg_{cname}"/\
                    f"{self.element_name}-{submodel_split_name}"\
                    f"-{instance_subset_name}-{measurement_subset_name}.pkl"
        if cache_path.exists() and not force_refresh_data:
            try:
                with open(cache_path, 'rb') as f: return pickle.load(f)
            except Exception as e:
                import traceback
                logger.warning(f"Exception loading cached TemplateGroup:"\
                               f" {e}\n{traceback.format_exc()}\nRegenerating.")
        else: cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        if instance_subset_name is not None:
            assert instance_subset_name in self.instance_subset_names, \
                f"Instance subset name {instance_subset_name} not found in suite {self.element_name}"
            instance_subset=self.instance_subset_names[instance_subset_name]
        else:
            instance_subset=self.instance_subset_names.get(submodel_split_name, None)
        if measurement_subset_name is not None:
            assert measurement_subset_name in self.measurement_subset_names, \
                f"Measurement subset name {measurement_subset_name} not found in suite {self.element_name}"
            measurement_subset=self.measurement_subset_names[measurement_subset_name]
        else:
            measurement_subset=self.measurement_subset_names.get(submodel_split_name, None)

        tg = self.get_template_group_explicit(instance_subset=instance_subset,
                                              measurement_subset=measurement_subset,
                                              param_set=param_set)
        with open(cache_path, 'wb') as f: pickle.dump(tg, f)
        return tg
    
    def get_template_group_explicit(self, param_set: ParamSet,
                instance_subset:Optional[list[str]]=None,
                measurement_subset:Optional[list[str]]=None) -> TemplateGroup:
        """Get a TemplateGroup for the specified subsets.

        Args:
            instance_subset: A list of instance names to include. If None, includes all instances.
            measurement_subset: A list of measurement names to include. If None, includes all measurements.
        Returns:
            A TemplateGroup for the specified subsets.
        """
        raise NotImplementedError("Subclasses must implement get_template_group_explicit")
    
    def get_modelcard_text(self, mcw: ModelCardWriter) -> str:
        raise NotImplementedError

    @property
    def va_includes(self) -> list[str]:
        raise NotImplementedError


class FittableModelSuite(ModelSuite):
    def __init__(self, element_name:str, param_set:ParamSet, release_name:str,
                 submodel_split:dict[str, str|None]={'all': None},
                 instance_subset_names:dict[str, list[str]]={},
                 measurement_subset_names:dict[str, list[str]]={},
                 measurement_specified_parameters:list[str]=None, # type: ignore
                 default_opt_kwargs:dict={},
                 default_gui_kwargs:dict={},
                 export_with_builtin:str|bool=False
                 ):
        """
        
        Args:
            measurement_specified_parameters: A list of parameter names that are specified by
                measurement conditions (e.g. Vds, Ids) and should not be modified in fitting patches.
        """
        super().__init__(element_name, release_name,
                         submodel_split=submodel_split,
                         instance_subset_names=instance_subset_names,
                         measurement_subset_names=measurement_subset_names)
        assert measurement_specified_parameters is not None, \
            "measurement_specified_parameters must be provided, even if empty"
        self.param_set = param_set
        self.measurement_specified_parameters = measurement_specified_parameters
        self.default_opt_kwargs = default_opt_kwargs
        self.default_gui_kwargs = default_gui_kwargs
        self.export_with_builtin = export_with_builtin

    def get_starting_patch(self) -> ParamPatch:
       return self.param_set.get_defaults_patch(ignore_keys=self.measurement_specified_parameters)
    
    def get_saved_params_path(self, submodel_split_name:str='all') -> Path:
        save_dir=SAVE_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir/f"{self.element_name}-{self.release_name}-{submodel_split_name}.yaml"
    
    def get_saved_patch_for_fitting(self, submodel_split_name:str='all') -> ParamPatch:
        import yaml
        with open(self.get_saved_params_path(submodel_split_name), 'r') as f:
            saved=yaml.safe_load(f)
        gv={k:v for k,v in saved['global_values'].items() if k in self.param_set._pdict}
        return self.param_set.make_complete_patch(**gv)
    
    def get_saved_patch_for_export(self, submodel_split_name:str='all') -> ParamPatch:
        return self.get_saved_patch_for_fitting(submodel_split_name)

    #def get_saved_as_patchgroup(self) -> dict[Optional[str], ParamPatch]:
    #    return {cond: self.get_saved_patch(ssn) for ssn,cond in self.submodel_split.items()}
    
    @property
    def va_includes(self) -> list[str]:
        if self.export_with_builtin: return []
        else: return self.param_set.va_includes
    
    def get_template_group(self, param_set:Optional[ParamSet]=None, submodel_split_name:str='all',
                           instance_subset_name:Optional[str]=None,
                           measurement_subset_name:Optional[str]=None,
                           force_refresh_data:bool=False,
                           ) -> TemplateGroup:
        return super().get_template_group(
            param_set=(param_set or self.param_set),
            submodel_split_name=submodel_split_name,
            instance_subset_name=instance_subset_name,
            measurement_subset_name=measurement_subset_name,
            force_refresh_data=force_refresh_data,)
    
    def get_opt_kwargs(self, starting_patch:ParamPatch) -> dict:
        return self.default_opt_kwargs
    
    def get_fitting_gui(self, submodel_split_name:str='all',
                           instance_subset_name:Optional[str]=None,
                           measurement_subset_name:Optional[str]=None,
                           force_refresh_data:bool=False,
                           backend='ngspice', opt_kwargs={}, gui_kwargs={}, threaded=False):
        from compyct.optimizer import SemiAutoOptimizer, SemiAutoOptimizerGui
        tg = self.get_template_group(submodel_split_name=submodel_split_name,
                                     instance_subset_name=instance_subset_name,
                                     measurement_subset_name=measurement_subset_name,
                                     force_refresh_data=force_refresh_data)
        starting_patch = self.get_starting_patch()
        save_path=self.get_saved_params_path(submodel_split_name=submodel_split_name)
        sao=SemiAutoOptimizer(backend=backend, output_dir=save_path.parent,
            global_template_group=tg, global_patch=starting_patch,
            **(self.get_opt_kwargs(starting_patch) | opt_kwargs))
        gui=SemiAutoOptimizerGui(sao, save_name=save_path.stem, **(self.default_gui_kwargs | gui_kwargs),)# load_before_run=True)
        gui._load_button_pressed(None)
        #gui._sao._mss.print_netlists(); exit()
        pn.serve(gui, title=f'{self.element_name} {submodel_split_name} Fit GUI', threaded=threaded)
        return gui
        
    def get_modelcard_text(self, mcw: ModelCardWriter) -> str:
        return mcw.simplifier_patch_group_to_modelcard_string()
    
    def get_submodel_modelcard_text(self, submodel_split_name: str, mcw: ModelCardWriter) -> str:
        if len(self.submodel_split)==1:
            element_name=self.element_name
        else:
            element_name=f"{self.element_name}_cond{submodel_split_name}"
        return mcw.simplifier_patch_to_modelcard_string(
            patch=self.get_saved_patch_for_export(submodel_split_name),
            element_name=element_name,
            netmap=self.get_netmap(), pcell_params=self.param_set.pcell_params, # type: ignore
            extra_text=self.get_extra_text(mcw), use_builtin=self.export_with_builtin)
    
    def get_netmap(self) -> dict[str,str]: return {}
    def get_extra_text(self, mcw: ModelCardWriter) -> str: return ""

class WrapperModelSuite(ModelSuite):
    def __init__(self, element_name:str, wrapped_suite:FittableModelSuite):
        self.wrapped_suite = wrapped_suite
        super().__init__(element_name=element_name, release_name=wrapped_suite.release_name,
                 submodel_split=wrapped_suite.submodel_split,
                 instance_subset_names=wrapped_suite.instance_subset_names,
                 measurement_subset_names=wrapped_suite.measurement_subset_names)
    def get_netmap(self) -> dict[str,str]: return {}
    def get_modelcard_text(self, mcw: ModelCardWriter) -> str:
        inner_pcell_params=self.wrapped_suite.param_set.pcell_params
        pass_params=dict(self.wrapped_suite.param_set.get_defaults_patch(only_keys=inner_pcell_params))
        return mcw.get_wrapper_modelcard_string(element_name=self.element_name, inner_element_name=self.wrapped_suite.element_name,
                                                pass_parameters=pass_params, eat_parameters={},
                                                extra_text=self.get_extra_text(mcw))
    def get_extra_text(self, mcw: ModelCardWriter) -> str: return ""
        
    @property
    def va_includes(self) -> list[str]: return []

class Bundle():

    _registry={}

    def __init__(self, pdk:str, release_name:str,
                 model_suites_by_file:dict[str,list[ModelSuite]],
                 header:str=''):
        self.pdk = pdk
        self.release_name = release_name
        self.model_suites = model_suites_by_file
        self.header = header
        for file, msuites in model_suites_by_file.items():
            for ms in msuites:
                assert ms.release_name == release_name,\
                    "All ModelSuites in a Bundle must have the same release_name"
            assert len([ms.element_name for ms in msuites])==len(msuites),\
                f"All ModelSuites in a single bundle file must have unique element_names"\
                f" violated in file {file} with {[ms.element_name for ms in msuites]}."
        Bundle._registry[(pdk, release_name)]=self
    
    def export(self, backend='spectre'):
        from compyct.backends.backend import ModelCardWriter
        mcw=ModelCardWriter.get_with_backend(backend)
        bundle_dir=OUTPUT_DIR/"bundles"/self.release_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting bundle to {bundle_dir}")
        for filename, msuites in self.model_suites.items():
            mcw.write_modelcard_file(bundle_dir/filename,
                                     header=self.header, model_suites=msuites,)
        va_includes=set(vai for msuites in self.model_suites.values() for ms in msuites for vai in ms.va_includes)
        for vafile in set(va_includes):    
            (bundle_dir/vafile).write_text(get_va_path(vafile).read_text())
            
    @staticmethod
    def get_bundle(pdk:str, release_name:str) -> 'Bundle':
        return Bundle._registry[(pdk, release_name)]
    @staticmethod
    def list_bundles() -> list[tuple[str,str]]:
        return list(Bundle._registry.keys())

    
