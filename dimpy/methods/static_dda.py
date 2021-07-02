from .base import CalcMethodBase

class DDAs(CalcMethodBase):
    """A static (without retardation effects) discrete dipole
    approximation (DDA) method.

    See :class:`dimpy.methods.base.CalcMethodBase` for full documentation.

    **Examples:**

    Molecular nanoparticle at a static frequency.
    This is the same as running the example in 
    ``DIMPy/examples/minimal_input_dda.dimpy``::

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Ag 0 0 0; Ag 0 0 1.89', verbose=0)
        >>> nano.build()
        >>> calc = dimpy.DDAs(nano)
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        12.575719

    Periodic nanoparticle excited at 450 nm.
    This is the same as running the example in
    ``DIMPy/examples/gold_chain_dda.dimpy``::

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Au 0 0 0', pbc=[[0, 0, 3.32]],
        ...        atom_params={'Au': {'exp': 'Au_jc', 'rad': 1.66}})
        >>> nano.verbose = 0
        >>> nano.build()
        >>> calc = dimpy.DDAsPBC(nano, freqs=0.10125189)
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        (4.5884604+34.322826j)

    """

    interaction = 'DDA'
    """Discrete Dipole Approximation"""

    model = 'PIM'
    """Polarizability Interaction Model"""

