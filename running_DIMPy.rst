*************
Running DIMPy
*************

Ways to Run
###########

DIMPy may be run in one of two ways:

1.  From a DIMPy input file (see `DIMPy Input File`_).
    For example, from the command line

    .. code-block:: bash

        python -m dimpy filename.dimpy

    or from python

    .. code-block:: python

        >>> import dimpy
        >>> calc = dimpy.run('filename.dimpy')

    where ``filename.dimpy`` is a DIMPy input file.

2.  From within python, explicitly specifying the nanoparticle
    and calculation type, for example

    .. code-block:: python

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Ag 0 0 0; Ag 0 0 1.89')
        >>> nano.build()
        >>> calc = dimpy.DIMr(nano, kdir='y')
        >>> calc.run()


DIMPy Input File
################

The DIMPy input file is a file with extension ``.dimpy`` 
that contains all the information necessary to run a DIMPy calculation.
In this documentation, we will go over the options in a DIMPy Input.

See files in ``DIMPy/examples/`` for examples for DIMPy input files for specfic
types of calculation.

Minimal Input
=============

The following is the minimal input to perform a static (or
frequency-independent), discrete interaction model, quasi-static
approximation, molecular calculation on 2 silver atoms, where the
coordinates are assumed to be in Angstrom. The radius of the 
silver atoms are taken to be the van der Walls radius. This minimal input
file (and all examples in this section)
can be found in the ``DIMPy/examples/`` directory.

.. code-block:: console

    NANOPARTICLE
      Atoms
        Ag 0 0 0 
        Ag 0 0 1.89
      End 
    ENDNANOPARTICLE

    METHOD
      Interaction DDA 
    ENDMETHOD

``NANOPARTICLE`` (case insensitive) specificies the information for the
nanoparticle, with the ``Atoms`` starting the coordinates input sub-block
for the nanoparticle. See `Nanoparticle Options`_ for more information.
The ``Atoms`` sub-block is ended with the ``End`` key while the ``NANOAPARTICLE``
block is ended with the ``ENDNANOPARTICLE`` key.

``METHOD`` specifies the calculation method and is ended with ``ENDMETHOD``.
In this example, we specify the type of atom interaction with ``Interaction DDA``.
See  `Method Options`_ for more information.

You may specify other input keys that are not unique to the nanoparticle or
calculation method. For more details, see `Other Input Options`_.

Nanoparticle Options
====================

The general ``NANOPARTICLE`` block is

.. code-block:: console

    NANOPARTICLE
      <Coordinates>
      [PBC]
      [Unit]
      [AtomParam]
    ENDNANOPARTICLE

*   ``Coordinates`` (required) specifies the coordinates of the nanoparticle,
    and may be done in one of two ways: (1) specifying the coordinates explicitly
    using the ``Atoms`` sub-block; or (2) specifying an ``.xyz`` file using
    the ``XYZFile`` keyword. Examples are below:

    1.  Using the ``Atoms`` sub-block:

        .. code-block:: console

            Atoms
              atom1 x1 y1 z1
              atom2 x2 y2 z1
              ...
            End

        where ``atom1`` is the symbol for an atom with coordinates (``x1``,
        ``y1``, ``z1``), and so on.

    2.  Using the ``XYZFile`` keyword:

        .. code-block:: console

            XYZFile filename.xyz

        where ``filename.xyz`` (case sensitive) is the (relative) path
        to the ``.xyz`` file that contains the coordinates of the nanoparticle.

*   ``PBC`` (optional) specifies the periodic lattice parameters using a sub-block.
    The nanoparticle may be a single molecule (default, in absence of the ``PBC``
    sub-block) or an infinite one- or two-dimensional array of nanoparticles.
    For a one-dimensional, periodically repeating nanoparticle, use 

    .. code-block:: console

        PBC
          <Ux> <Uy> <Uz>
        End

    where ``<Ux>``, ``<Uy>``, and ``<Uz>`` specify the *x*, *y*, and *z* 
    coordinates of the lattice vector.

    For nanoparticles repeating periodically
    in two dimensions with another lattice vector (``<Vx>``, ``<Vy>``, ``<Vz>``),
    use

    .. code-block:: console

        PBC
          <Ux> <Uy> <Uz>
          <Vx> <Vy> <Vz>
        End

*   ``Unit`` (optional) specifies whether input coordinates and lattice vectors
    are given in Angstroms (the default) or Bohrs (atomic units). To specify
    that coordinates are given in Angstroms (again, this is the default behavior):

    .. code-block:: console

        Unit Angstrom

    To specify that coordinates are in atomic units, you may use:

    .. code-block:: console

        Unit Bohr

*   ``AtomParam`` (optional) specifies an atom-specific parameter and may be
    repeated as many times as necessary. The general useage is:

    .. code-block:: console

        AtomParam Xx key value

    where ``Xx`` is the atomic symbol, ``key`` is the type of parameter
    that you're setting (examples include ``rad`` and ``exp``) , and
    ``value`` is the value for that key.

    Let's say you want to give silver atoms a radius of 1.72 Angstroms
    (this is the default van-der Waals radius):

    .. code-block:: console

        AtomParam Ag rad 1.72

    Or, let's say you want to use the ``Au_jc`` dielectric function for 
    gold atoms (needed for frequency-dependent calculations):

    .. code-block:: console

        AtomParam Au exp Au_jc

Method Options
==============

The general ``METHOD`` block is

.. code-block:: console

    METHOD
      <Interaction>
      [Excitation]
      [Kdir]
      [Solver]
    ENDMETHOD

*   ``Interaction`` (required) denotes the type of interaction between
    atoms. For a discrete interaction model (DIM) calculation that uses
    screened interaction tensors, use:

    .. code-block:: console

        Interaction DIM

    For a discrete dipole approximation (DDA) calculation that uses
    un-screened interaction tensors, use:

    .. code-block:: console

        Interaction DDA

*   ``Excitation`` (optional) specify the excitation frequency (or frequencies).
    The default (no excitation specification) performs a static (or 
    frequency-independent) calculation. The excitaiton frequencies may be specified
    in one of two ways:

    1.  You may specify a calculation at a specified frequency (or frequencies)
        using

        .. code-block:: console

            Frequency <unit> <value> [<value2> <value3> ...]

        where ``<unit>`` may be one of ``ev`` (electron volts), ``nm`` (nanometers),
        ``hz`` (hertz), ``cm-1`` (wave numbers), ``hartree`` (hatrees),
        or ``au`` (atomic units or hartrees).
        ``<value>`` is the frequency to use. You may specify as many frequencies
        (separated by spaces) as needed, though specifying more than one frequency
        this way is optional.

    2.  Alternatively, you may also specify a range of frequencies using

        .. code-block:: console

            FreqRange <unit> <start> <end> <number>

        where ``<start>`` is the starting frequency, ``<end>`` is the 
        ending frequency, and ``<number>`` is the number of frequencies
        between ``<start>`` and ``<end>`` to use (inclusively).


*   ``Kdir`` (optional) specifies the direction of the **k**-vector. In its
    absence, we assume the quasi-static behavior, where the electric
    fields do not change phase with distance (the default behavior).
    To include phase-changes (or retardation) effects, include the following
    in the input file

    .. code-block:: console

        Kdir <dir>

    where ``<dir>>`` is one of either ``x``, ``y``, or ``z`` and gives the direction
    of the **k**-vector for retardation effects. The magnitude of the **k**-vector
    is determined automatically from the frequency (see above).
    Note that the ``Kdir`` key does nothing when the frequency is zero (i.e. a
    static calculation).

*   ``Solver`` (optional) chooses the type of solver to use. The solvers used
    are all part of the :mod:`scipy.sparse.linalg` module. To manually choose
    a solver, include the following in the DIMPy input file

    .. code-block:: console

        Solver <option>

    where ``<option>`` may be one of (see the full list at
    :attr:`dimpy.read_input_file.ReadInput.solvers`):

    *   ``direct`` - direct inversion. Fast, but uses a lot of memory. Good for
        small nanoparticles (maximum of ~12k atoms with ~32GB of RAM).

    *   ``bicg`` - BIConjugate Gradient iteration

    *   ``bicgstab`` - BIConjugate Gradient STABilized iteration

    *   ``cg`` - Conjugate Gradient iteration (unstable)

    *   ``cgs`` - Conjugate Gradient Squared iteration (unstable)

    *   ``gmres`` - Generalized Minimal RESidual interation

    *   ``lgmres`` - Linear Generalized Minimal RESidual interation

    *   ``minres`` - MINimum RESidual iteration (unstable/incorrect)

    *   ``qmr`` - Quasi-Minimal Residual iteration

    *   ``gcrotmk`` - Generalized Conjugate Residual with inner Orthogonalization and
        outer Truncation method (default)


Other Input Options
===================

You may include any of the following in your DIMPy input file (outside of either
of the ``NANOPARTICLE`` or ``METHOD`` blocks) to add (or remove)
functionalities:

*   ``TITLE <calculation title>`` - This lets you give the name
    ``<calculation title>`` to your calculation. Without it, you
    calculation title will be ``None``.

*   ``DEBUG`` - This will print additional timing and memory information into
    both the output and log files. This will also print the most verbose output
    and it overrides options for ``PRINT`` and ``NOPRINT`` (below).

*   ``VERBOSE <num>`` - This determines how much information is included in the
    output file. This keyword is ignored if ``DEBUG`` is given.
    Options for ``<num>`` are:

    *   ``0`` - Nothing is printed to the output file.

    *   ``1`` - Only the most important results are printed.

    *   ``2`` - Important results as well as key information are printed
        (default).

    *   ``3`` - Additional information, such as atomic dipoles, are
        printed at this level.

    *   ``4`` - Even more information is printed. ``VERBOSE 4`` is almost
        the same as ``DEBUG``, with the exception of printing additional
        timing and memory information.

Examples
========

All files listed in this section can be found in the ``DIMPy/examples/``
directory. All calculations in this section may be analyzed using the
:class:`dimpy.analyzer.Analyzer` class.

Absorbance / scattering spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example calculates the absorbance spectrum in the visible
region using DDA for a small 100-atom gold nanoparticle using the dielectric
function found in ``Au_jc``.
The calculation calculates the properties of the nanoparticle at 40 frequency
steps between 400 nm and 800 nm.
This example uses input file ``gold_absorption_dda.dimpy``
and the associated `.xyz` file ``Au_147_ico.xyz``.

.. code-block:: console

    TITLE Gold absorption spectrum

    NANOPARTICLE
      XYZFile Au_147_ico.xyz
      AtomParam Au exp Au_jc
    ENDNANOPARTICLE

    METHOD
      Interaction DDA 
      FreqRange nm 400 800 40
    ENDMETHOD

You may run the calculation using the command line with

.. code-block:: console

    python -m dimpy gold_absorption_dda.dimpy

or as part of a python script with

.. code-block:: python

    >>> import dimpy

    >>> calc = dimpy.run('gold_absorption_dda.dimpy',
    ...                  output_filename='gold_absorption_dda.out')

1-Dimensional gold chain
^^^^^^^^^^^^^^^^^^^^^^^^

Let's calculate the absorbance for a 1-dimensional periodically repeating
gold chain that is 1-atom thick at 545 nm. This input file is
``gold_chain_dda.dimpy``.

.. code-block:: console

    TITLE 1-Dimensional single-atom gold chain

    NANOPARTICLE
      Atoms
        Au 0 0 0 
      End 
      AtomParam Au exp Au_jc
      PBC 
        0 0 2.48
      End 
    ENDNANOPARTICLE

    METHOD
      Interaction DDA 
      Frequency nm 545 
    ENDMETHOD

2-Dimensional silver sheet
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's calculate the absorbance for a 2-dimensional periodically repeating
silver sheet that is 1-atom thick at 400 nm. The input file is
``silver_sheet_dda.dimpy``.

.. code-block:: console

    TITLE 2-Dimensional single-atom silver sheet

    NANOPARTICLE
      Atoms
        Ag 0 0 0 
      End 
      PBC 
        0 0 1.89
        0 1.89 0
      End 
      AtomParam Ag exp Ag_jc
    ENDNANOPARTICLE

    METHOD
      Interaction DDA 
      Frequency nm 545 
    ENDMETHOD

DDA with retartdation effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input file is ``method_dda_retardation.dimpy``.

.. code-block:: console

    TITLE DDA with retardation

    NANOPARTICLE
      XYZFile Au_147_ico.xyz
      AtomParam Au exp Au_jc
    ENDNANOPARTICLE

    METHOD
      Interaction DDA 
      Frequency nm 545 
      Kdir x
    ENDMETHOD

