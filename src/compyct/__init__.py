__version__='0.0.1'

import sys

from dotenv import load_dotenv
load_dotenv()

from pint import UnitRegistry
ureg=UnitRegistry()
ureg.define("square = 1 = sq")

from .util import logger, set_logging_callback, unset_logging_callback
