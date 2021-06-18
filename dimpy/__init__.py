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
from .method_dynamic_dda import DDAr
from .method_dynamic_dda_pbc import DDArPBC
from .method_static_dda import DDAs
from .method_static_dda_pbc import DDAsPBC
from .method_static_dim import DIMs
from .nanoparticle import Nanoparticle
from .printer import Output
from .read_input_file import ReadInput
from ._version import __version__

__package__ = 'dimpy'
