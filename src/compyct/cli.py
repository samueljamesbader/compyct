from pathlib import Path
import sys
from typing import Literal, overload
from datavac.util.cli import CLIIndex
from argparse import ArgumentParser


def entrypoint_compyct_cli():
    from compyct import initialize_bundles
    initialize_bundles()
    CLIIndex({
        'list': cli_list,
        'fit': cli_fit,
        'export': cli_export,
        'copy_param (cp)': cli_copy_param,
        'playback (pb)': cli_playback,
    })(*sys.argv)


@overload
def resolve_bundle_args(pdk:str|None, release_name:str|None, file:str|None, do_file:Literal[False], no_external=True) -> tuple[str,str,None]: ...
@overload
def resolve_bundle_args(pdk:str|None, release_name:str|None, file:str|None, do_file:Literal[True], no_external=True) -> tuple[str,str,str]: ...
@overload
def resolve_bundle_args(pdk:str|None, release_name:str|None, file:list[str]|None, do_file:Literal['list'], no_external=True) -> tuple[str,str,list[str]]: ...

def resolve_bundle_args(pdk=None, release_name=None, file=None, do_file=True, no_external=True):
    from compyct.model_suite import Bundle, ExternalBundle

    # Get available PDKs
    pdks = sorted(set(p for p, _ in Bundle.list_bundles()))

    # Validate or resolve PDK
    if pdk:
        if pdk not in pdks:
            raise ValueError(
                f"PDK '{pdk}' not found. Available PDKs: {', '.join(pdks)}"
            )
    else:
        if len(pdks) == 1:
            pdk = pdks[0]
            print(f"Using only available PDK: {pdk}")
        else:
            if len(pdks) == 0:
                raise ValueError("No PDKs available.  Make sure COMPYCT_PRELOAD_MODULES is set correctly.")
            print("Available PDKs:")
            for i, p in enumerate(pdks):
                print(f"{i+1}: {p}")
            idx = int(input("Select PDK by number: ")) - 1
            pdk = pdks[idx]

    # Get available releases for selected PDK
    releases = sorted(set(r for p, r in Bundle.list_bundles() if p == pdk
                          and ((not no_external) or (not isinstance(Bundle.get_bundle(p,r), ExternalBundle)))))

    # Validate or resolve release_name
    if release_name:
        if release_name not in releases:
            raise ValueError(
                f"Release '{release_name}' not found for PDK '{pdk}'. "
                f"Available releases: {', '.join(releases)}"
            )
    else:
        if len(releases) == 1:
            release_name = releases[0]
            print(f"Using only available release: {release_name}")
        else:
            print(f"Available releases for PDK '{pdk}':")
            for i, r in enumerate(releases):
                print(f"{i+1}: {r}")
            idx = int(input("Select release by number: ")) - 1
            release_name = releases[idx]

    bundle = Bundle.get_bundle(pdk, release_name)

    # Get available files in bundle
    files = list(bundle.model_suites.keys())

    # Validate or resolve file
    if do_file is True:
        if file:
            if file not in files:
                raise ValueError(
                    f"File '{file}' not found in bundle ({pdk}, {release_name}). "
                    f"Available files: {', '.join(files) if files else '[none]'}"
                )
        else:
            if len(files) == 1:
                file = files[0]
                print(f"Using only available modelcard file: {file}")
            elif files:
                print("Available modelcard files:")
                for i, f in enumerate(files):
                    print(f"{i+1}: {f}")
                idx = int(input("Select file by number: ")) - 1
                file = files[idx]
            else:
                file = input("Enter modelcard file name: ")
    elif do_file == 'list':
        if file:
            for f in file:
                if f not in files:
                    raise ValueError(
                        f"File '{f}' not found in bundle ({pdk}, {release_name}). "
                        f"Available files: {', '.join(files) if files else '[none]'}"
                    )
        else:
            print(f"Using all available modelcard files: {files}")
            file = files

    return pdk, release_name, file

def cli_fit(*args):
    parser = ArgumentParser(description="Compyct Fit CLI")
    parser.add_argument('--pdk', type=str, nargs='?', help='PDK name (optional if only one)')
    parser.add_argument('--release_name', type=str, nargs='?', help='Release name (optional if only one)')
    parser.add_argument('--file','-f', type=str, nargs='?', help='Modelcard file within release (optional if only one)')
    parser.add_argument('--element','-e', type=str, nargs='?', help='Element name to fit')
    parser.add_argument('--submodel_split_name', '-ssub', type=str, default=None, nargs='?', help='Submodel split name')
    parser.add_argument('--instance_subset_name', '-isub', type=str, nargs='?', help='Instance subset name')
    parser.add_argument('--measurement_subset_name', '-msub', type=str, nargs='?', help='Measurement subset name')
    parser.add_argument('--force_refresh_data', '-rd', action='store_true', help='Force refresh data (default: False)')
    parser.add_argument('--backend', type=str, nargs='?', default='ngspice', help='Backend to use (e.g., spectre, ngspice)')
    parsed_args = parser.parse_args(args)

    pdk, release_name, file = resolve_bundle_args(parsed_args.pdk, parsed_args.release_name, parsed_args.file, do_file=True)

    from compyct.model_suite import Bundle
    bundle = Bundle.get_bundle(pdk, release_name)
    if parsed_args.element is None:
        if len(bundle.model_suites[file]) == 1:
            element = bundle.model_suites[file][0].element_name
            print(f"Using only available element: {parsed_args.element}")
        else:
            print("Available elements:")
            for i, ms in enumerate(bundle.model_suites[file]):
                print(f"{i+1}: {ms.element_name}")
            idx = int(input("Select element by number: ")) - 1
            element = bundle.model_suites[file][idx].element_name
    else: element = parsed_args.element
    try:
        ms=next(ms for ms in bundle.model_suites[file] if ms.element_name == element)
    except StopIteration:
        raise Exception(f"Element {element} not found in file {file} of bundle ({pdk}, {release_name})")
    
    from compyct.model_suite import FittableModelSuite
    assert isinstance(ms, FittableModelSuite), f"Element {element} in file {file} of bundle ({pdk}, {release_name}) is not a FittableModelSuite"
    ms.get_fitting_gui(
        submodel_split_name=parsed_args.submodel_split_name,
        instance_subset_name=parsed_args.instance_subset_name,
        measurement_subset_name=parsed_args.measurement_subset_name,
        force_refresh_data=parsed_args.force_refresh_data,
        backend=parsed_args.backend
    )

def cli_export(*args):
    parser = ArgumentParser(description="Compyct Export CLI")
    parser.add_argument('--pdk', type=str, nargs='?', help='PDK name (optional if only one)')
    parser.add_argument('--release_name', type=str, nargs='?', help='Release name (optional if only one)')
    parsed_args = parser.parse_args(args)
    pdk, release_name, _ = resolve_bundle_args(parsed_args.pdk, parsed_args.release_name, file=None, do_file=False)
    from compyct.model_suite import Bundle
    bundle = Bundle.get_bundle(pdk, release_name)
    bundle.export()

def cli_copy_param(*args):
    parser = ArgumentParser(description="Compyct Copy Parameter CLI")
    parser.add_argument('--from_pdk','-fp', type=str, nargs='?', help='Source PDK name (optional if only one)')
    parser.add_argument('--from_release_name','-fr', type=str, nargs='?', help='Source Release name (optional if only one)')
    parser.add_argument('--from_file','-ff', type=str, nargs='?', help='Source Modelcard file within release (optional if only one)')
    parser.add_argument('--from_element', '-fe', type=str, nargs='?', help='Source Element name')
    parser.add_argument('--from_submodel_split_name', '-fs', type=str, nargs='?', default=None, help='Source Submodel split name (optional)')
    parser.add_argument('--to_pdk','-tp', type=str, nargs='?', help='Target PDK name (optional if only one)')
    parser.add_argument('--to_release_name','-tr', type=str, nargs='?', help='Target Release name (optional if only one)')
    parser.add_argument('--to_file','-tf', type=str, nargs='?', help='Target Modelcard file within release (optional if only one)')
    parser.add_argument('--to_element', '-te', type=str, nargs='?', help='Target Element name')
    parser.add_argument('--to_submodel_split_name', '-ts', type=str, nargs='*', default=[None], help='Target Submodel split name (optional)')
    parser.add_argument('params', type=str, nargs='+', help='Parameter names to copy')
    parsed_args = parser.parse_args(args)
    from compyct.model_suite import Bundle, FittableModelSuite

    from_pdk, from_release_name, from_file = resolve_bundle_args(parsed_args.from_pdk, parsed_args.from_release_name, parsed_args.from_file, do_file=True)
    from_bundle = Bundle.get_bundle(from_pdk, from_release_name)
    try:
        from_ms=next(ms for ms in from_bundle.model_suites[from_file] if ms.element_name == parsed_args.from_element)
    except StopIteration:
        raise Exception(f"Element {parsed_args.from_element} not found in file {from_file} of bundle ({from_pdk}, {from_release_name})")
    assert isinstance(from_ms, FittableModelSuite), f"Element {parsed_args.from_element} is not a FittableModelSuite"
    from_ssn = parsed_args.from_submodel_split_name or from_ms.default_submodel_split_name
    assert from_ssn is not None, f"Source submodel split name must be specified for element {parsed_args.from_element}"
    from_patch = from_ms.get_saved_patch_for_fitting(submodel_split_name=from_ssn)

    to_pdk, to_release_name, to_file = resolve_bundle_args(parsed_args.to_pdk, parsed_args.to_release_name, parsed_args.to_file, do_file=True)
    to_bundle = Bundle.get_bundle(to_pdk, to_release_name)
    try:
        to_ms=next(ms for ms in to_bundle.model_suites[to_file] if ms.element_name == parsed_args.to_element)
    except StopIteration:
        raise Exception(f"Element {parsed_args.to_element} not found in file {to_file} of bundle ({to_pdk}, {to_release_name})")
    assert isinstance(to_ms, FittableModelSuite), f"Element {parsed_args.to_element} is not a FittableModelSuite"
    to_ssns = [(tssn or to_ms.default_submodel_split_name) for tssn in parsed_args.to_submodel_split_name]
    assert all(to_ssn is not None for to_ssn in to_ssns),\
        f"Target submodel split name must be specified for element {parsed_args.to_element}"
    
    for to_ssn in to_ssns:
        import yaml
        with open(to_ms.get_saved_params_path(to_ssn),'r') as f:
            orig_yaml=yaml.safe_load(f)
        for param in parsed_args.params:
            print(f"Copying parameter {param}={from_patch[param]} "\
                  f"from {parsed_args.from_element}:{from_ssn} to {parsed_args.to_element}:{to_ssn}")
            if param not in from_patch:
                raise Exception(f"Parameter {param} not found in source patch of element {parsed_args.from_element}")
            orig_yaml['global_values'][param]=from_patch[param]
        with open(to_ms.get_saved_params_path(to_ssn),'w') as f:
            yaml.safe_dump(orig_yaml,f)

    #raise NotImplementedError("cli_copy_param not yet implemented")


def cli_playback(*args):
    parser = ArgumentParser(description="Compyct Playback CLI")
    parser.add_argument('--pdk', type=str, nargs='?', help='PDK name (optional if only one)')
    parser.add_argument('--release_name', type=str, nargs='?', help='Release name (optional if only one)')
    parser.add_argument('--file','-f', type=str, nargs='*', default=None, help='Modelcard file within release (optional if only one)')
    parser.add_argument('--override_file_path', type=str, nargs='?', default=None, help='Override modelcard file path (optional)')
    parser.add_argument('--element_names', '-e', type=str, nargs='*', default=None, help='Element names to playback')
    parser.add_argument('--circuit_names', '-c', type=str, nargs='*', default=None, help='Circuit collection names to playback')
    parser.add_argument('--instance_subset_name', '-isub', type=str, nargs='?', default=None, help='Instance subset names')
    parser.add_argument('--measurement_subset_name', '-msub', type=str, nargs='?', default=None, help='Measurement subset names')
    parser.add_argument('--force_refresh_data', '-rd', action='store_true', help='Force refresh data (default: False)')
    parser.add_argument('--allow_external', action='store_true', default=False, help='Allow external bundles (default: False)')
    parsed_args = parser.parse_args(args)
    pdk, release_name, files = resolve_bundle_args(parsed_args.pdk, parsed_args.release_name, parsed_args.file,
                                                  do_file='list', no_external=(not parsed_args.allow_external))
    from compyct.model_suite import Bundle
    bundle = Bundle.get_bundle(pdk, release_name)
    from compyct.model_suite import FittableModelSuite
    element_names = parsed_args.element_names
    circuit_names = parsed_args.circuit_names
    for file in files:
        print(f"Playing back bundle ({pdk}, {release_name}) for file: {file}")
        model_suites = [ms for ms in bundle.model_suites[file] if (element_names is None or ms.element_name in element_names)]

        from compyct import OUTPUT_DIR
        bundle_dir=bundle.get_bundle_path()
        actual_path=Path(parsed_args.override_file_path) if parsed_args.override_file_path is not None else bundle_dir/file
        if not ((element_names is None) and (circuit_names is not None)):
            tgse={ms.element_name:
                ms.get_template_group(param_set=ms.playback_ps_class(model=ms.element_name,file=actual_path,section='tttt'),
                                      instance_subset_name=parsed_args.instance_subset_name,
                                      measurement_subset_name=parsed_args.measurement_subset_name,
                                      force_refresh_data=parsed_args.force_refresh_data)
                    for ms in model_suites
                        if (element_names is None or (ms.element_name in element_names))
            }
        else: tgse={}
        if not ((circuit_names is None) and (element_names is not None)):
            tgsc={cc.collection_name:
                cc.get_template_group(includes=[(actual_path,'section=tttt')],
                                      force_refresh_data=parsed_args.force_refresh_data)
                    for cc in bundle.circuits[file]
                        if (circuit_names is None or (cc.collection_name in circuit_names))
            }
        else: tgsc={}
        tgs={**tgse, **tgsc}
            
        from compyct import logger
        from compyct.backends.backend import MultiSimSesh
        from compyct.optimizer import rerun_with_params
        import panel as pn
        for elt, tg in tgs.items():
            save_path=OUTPUT_DIR/f"playback/{pdk}-{release_name}-{file}-{elt}.html"    
            save_path.parent.mkdir(parents=True,exist_ok=True)
            logger.info(f"Running playback simulations for {elt}...")
            with MultiSimSesh.get_with_backend(simtemps=tg.only_simtemps(), backend='spectre') as mss:
                from compyct.backends.spectre_backend import SpectreMultiSimSesh
                assert isinstance(mss, SpectreMultiSimSesh)
                with open(save_path.parent/"netlists.txt",'w') as f:
                    mss.print_netlists(file=f)
                rerun_with_params(None, None, tg, mss)
            major_tabnames=list(dict.fromkeys(k.split("|||")[0] for k in tg))
            fig_layout_params=dict(width=200,height=250)
            playback=pn.Tabs(*[
                (majortabname,
                        tg.only(*[k for k in tg if k.startswith(majortabname+"|||")]).get_figure_pane(
                            fig_layout_params=fig_layout_params,gridplot_options={'ncols':6},do_update=True)
                )
                for majortabname in major_tabnames
            ])
            logger.info(f"Saving playback html to {save_path}...")
            playback.save(save_path)
            


def cli_list(*args):
    from compyct.model_suite import Bundle
    print("Available Bundles (pdk, release_name):")
    for pdk, release_name in Bundle.list_bundles():
        print(f" - ({pdk}, {release_name}): {Bundle.get_bundle(pdk, release_name).get_bundle_path()}")

if __name__ == "__main__":
    entrypoint_compyct_cli()