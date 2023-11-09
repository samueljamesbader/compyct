import sys
from contextlib import redirect_stderr, contextmanager
from functools import cache
from io import StringIO
import re


# Copied from https://stackoverflow.com/a/22424821
# under license https://creativecommons.org/licenses/by-sa/4.0/
@cache
def is_notebook() -> bool:
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


@contextmanager
def catch_stderr(ignore_regexes,unmatched='stderr'):
    sio=StringIO()
    with redirect_stderr(sio):
        yield
    sio.seek(0)
    for l in sio.readlines():
        if l.strip()=="": continue
        if any((re.match(ir,l.strip())  for ir in ignore_regexes)):
            continue
        else:
            if unmatched=='stderr':
                print(l,file=sys.stderr)
            elif unmatched=='raise':
                raise Exception(l)
            else:
                raise Exception(f"What is '{unmatched}'?")
