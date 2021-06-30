"""
DIMPy
=====
A Discrete Interaction Model for Python.
----------------------------------------
(c) Dhabih V. Chulhai, 2020


Python version by:
^^^^^^^^^^^^^^^^^^

* Dhabih V. Chulhai (<chulhaid@uindy.edu>)

Orginal Fortran version by:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Lasse Jensen
* LinLin Jensen
* Justin E. Moore
* Seth M. Morton

How to cite:
^^^^^^^^^^^^

* some paper
* some other paper
"""

from .analyzer import Analyzer
from .dimpy import run
from .dimpy_error import DIMPyError
from .input_file import ReadInput
from .methods import DDAs, DDAsPBC, DDAr, DDArPBC
from .methods import DIMs, DIMsPBC
from .nanoparticle import Nanoparticle
from ._version import __version__

__package__ = 'dimpy'
