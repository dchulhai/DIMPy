*****
DIMPy
*****

A Discrete Interaction Model for Python

Citing this Program
===================

**Python code**:

* D.V. Chulhai; M. Magee; "A Discrete Interaction Model for Python" GitHub 2020,
  https://github.com/dchulhai/DIMPy

**Original fortran code**:

* L.L. Jensen; L. Jensen; "Electrostatic Interaction Model for the
  Calculation of the Polarizability of Large Noble Metal Nanoclusters"
  J. Phys. Chem. C 2008, 112, 15697.

Installing and Running DIMPy
============================

Download and install requirements
---------------------------------

.. code-block:: bash

    git clone https://github.com/dchulhai/DIMPy.git
    cd DIMPy/
    make

Running DIMPy
-------------

To run DIMPy, first make sure that the ``DIMPy`` directory is part of your
``PYTHONPATH`` environment variable. For example, add the following line
to your ``.bashrc``

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:\location\to\DIMPy

Then source your bashrc using

.. code-block:: bash

    . ~/.bashrc

You should now be able to import dimpy from within Python using

.. code-block:: python

    >>> import dimpy

To run DIMPy from the command line, type

.. code-block:: bash

    python3 -m dimpy dimpy_input_file.dimpy

where ``dimpy_input_file.dimpy`` is the DIMPy input file.
You may also run the same input file from within python as

.. code-block:: python

    >>> import dimpy
    >>> calc = dimpy.run('dimpy_input_file.dimpy')

Compiling documentation
-----------------------

.. code-block:: bash

    cd documentation/
    make html # to make html documents
    make latexpdf # to make PDF document

Copyright
=========

    A Discrete Interation Model for Python (DIMPy) simulates the
    electrodynamics interation in an atomistic nanoparticle.
    Copyright (C) 2020 Dhabih V. Chulhai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    | You may contact me using the email: chulhaid@uindy.edu
    | Or the address:
    | Department of Chemistry,
    | University of Indianapolis
    | 1400 E Hanna Ave,
    | Indianapolis, IN 46227
