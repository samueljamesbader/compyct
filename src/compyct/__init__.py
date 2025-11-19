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
if (CACHE_DIR:=os.environ.get('COMPACT_CACHE_DIR',None)) is None:
    from platformdirs import user_cache_dir
    CACHE_DIR=Path(user_cache_dir("compyct"))
else: CACHE_DIR=Path(CACHE_DIR)