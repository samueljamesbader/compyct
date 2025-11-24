import sys
import os
import importlib
from typing import Literal, overload
from datavac.util.cli import CLIIndex
from argparse import ArgumentParser



def entrypoint_compyct_cli():

    dotpaths=os.environ.get("COMPYCT_PRELOAD_MODULES","").split(',')
    for dp in dotpaths:
        if dp.strip()!='':
            importlib.import_module(dp)
            preload=getattr(importlib.import_module(dp),'compyct_preload',None)
            assert preload is not None, f"Preload module {dp} must define a compyct_preload() function"
            preload()

    CLIIndex({
        'list': cli_list,
        'fit': cli_fit,
        'export': cli_export,
        'playback (pb)': cli_playback,
    })(*sys.argv)


@overload
def resolve_bundle_args(pdk:str|None, release_name:str|None, file:str|None, do_file:Literal[False]) -> tuple[str,str,None]: ...
@overload
def resolve_bundle_args(pdk:str|None, release_name:str|None, file:str|None, do_file:Literal[True]) -> tuple[str,str,str]: ...

def resolve_bundle_args(pdk=None, release_name=None, file=None, do_file=True):
    from compyct.model_suite import Bundle

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
            print("Available PDKs:")
            for i, p in enumerate(pdks):
                print(f"{i+1}: {p}")
            idx = int(input("Select PDK by number: ")) - 1
            pdk = pdks[idx]

    # Get available releases for selected PDK
    releases = sorted(set(r for p, r in Bundle.list_bundles() if p == pdk))

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
    if do_file:
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

    return pdk, release_name, file

def cli_fit(*args):
    parser = ArgumentParser(description="Compyct Fit CLI")
    parser.add_argument('--pdk', type=str, nargs='?', help='PDK name (optional if only one)')
    parser.add_argument('--release_name', type=str, nargs='?', help='Release name (optional if only one)')
    parser.add_argument('--file', type=str, nargs='?', help='Modelcard file within release (optional if only one)')
    parser.add_argument('--element','-e', type=str, nargs='?', help='Element name to fit')
    parser.add_argument('--submodel_split_name', '-ssub', type=str, default='all', nargs='?', help='Submodel split name')
    parser.add_argument('--instance_subset_name', '-isub', type=str, nargs='?', help='Instance subset name')
    parser.add_argument('--measurement_subset_name', '-msub', type=str, nargs='?', help='Measurement subset name')
    parser.add_argument('--force_refresh_data', '-rd', action='store_true', help='Force refresh data (default: False)')
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
        force_refresh_data=parsed_args.force_refresh_data
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
    
def cli_playback(*args):
    parser = ArgumentParser(description="Compyct Playback CLI")
    parser.add_argument('--pdk', type=str, nargs='?', help='PDK name (optional if only one)')
    parser.add_argument('--release_name', type=str, nargs='?', help='Release name (optional if only one)')
    parser.add_argument('--file', type=str, nargs='?', help='Modelcard file within release (optional if only one)')
    parser.add_argument('--element_names', '-e', type=str, nargs='*', default=None, help='Element names to playback')
    parser.add_argument('--instance_subset_name', '-isub', type=str, nargs='?', default=None, help='Instance subset names')
    parser.add_argument('--measurement_subset_name', '-msub', type=str, nargs='?', default=None, help='Measurement subset names')
    parser.add_argument('--force_refresh_data', '-rd', action='store_true', help='Force refresh data (default: False)')
    parsed_args = parser.parse_args(args)
    pdk, release_name, file = resolve_bundle_args(parsed_args.pdk, parsed_args.release_name, parsed_args.file, do_file=True)
    from compyct.model_suite import Bundle
    bundle = Bundle.get_bundle(pdk, release_name)
    from compyct.model_suite import FittableModelSuite
    element_names = parsed_args.element_names
    model_suites = [ms for ms in bundle.model_suites[file] if (element_names is None or ms.element_name in element_names)]

    from compyct import OUTPUT_DIR
    bundle_dir=OUTPUT_DIR/"bundles"/release_name
    tgs={ms.element_name:
        ms.get_template_group(param_set=ms.playback_ps_class(model=ms.element_name,file=bundle_dir/file,section='tttt'),
                              instance_subset_name=parsed_args.instance_subset_name,
                              measurement_subset_name=parsed_args.measurement_subset_name,
                              force_refresh_data=parsed_args.force_refresh_data)
            for ms in model_suites
                if (element_names is None or (ms.element_name in element_names))
    }
    
    from compyct import logger
    from compyct.backends.backend import MultiSimSesh
    from compyct.optimizer import rerun_with_params
    import panel as pn
    for elt, tg in tgs.items():
        logger.info(f"Running playback simulations for {elt}...")
        with MultiSimSesh.get_with_backend(simtemps=tg.only_simtemps(), backend='spectre') as mss:
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
        save_path=OUTPUT_DIR/f"playback/{pdk}-{release_name}-{file}-{elt}.html"    
        save_path.parent.mkdir(parents=True,exist_ok=True)
        logger.info(f"Saving playback html to {save_path}...")
        playback.save(save_path)


def cli_list(*args):
    from compyct.model_suite import Bundle
    print("Available Bundles (pdk, release_name):")
    for pdk, release_name in Bundle.list_bundles():
        print(f" - ({pdk}, {release_name})")