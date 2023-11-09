from dotenv import load_dotenv
from pint import UnitRegistry

__version__='0.0.1'

load_dotenv()
ureg=UnitRegistry()
ureg.define("square = 1 = sq")