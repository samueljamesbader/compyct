__version__='0.0.1'

import sys

from dotenv import load_dotenv
load_dotenv()

from pint import UnitRegistry
ureg=UnitRegistry()
ureg.define("square = 1 = sq")

from .util import logger, set_logging_callback, unset_logging_callback

import os
from pathlib import Path
from platformdirs import user_cache_dir, user_data_dir
if (CACHE_DIR:=os.environ.get('COMPYCT_CACHE_DIR',None)) is None:
    CACHE_DIR=Path(user_cache_dir("compyct"))
else: CACHE_DIR=Path(CACHE_DIR)
if (SAVE_DIR:=os.environ.get('COMPYCT_SAVE_DIR',None)) is None:
    SAVE_DIR=Path(user_data_dir("compyct"))/"saved_params"
else: SAVE_DIR=Path(SAVE_DIR)
if (OUTPUT_DIR:=os.environ.get('COMPYCT_OUTPUT_DIR',None)) is None:
    OUTPUT_DIR=Path(user_data_dir("compyct"))/"outputs"
else: OUTPUT_DIR=Path(OUTPUT_DIR)


def initialize_bundles():
    import importlib
    dotpaths=os.environ.get("COMPYCT_PRELOAD_MODULES","").split(',')
    for dp in dotpaths:
        if dp.strip()!='':
            importlib.import_module(dp)
            preload=getattr(importlib.import_module(dp),'compyct_preload',None)
            assert preload is not None, f"Preload module {dp} must define a compyct_preload() function"
            preload()
