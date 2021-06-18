from .calc_method import CalcMethod

class DDAs(CalcMethod):
    """A static (without retardation effects) discrete dipole
    approximation (DDA) method.

    See :class:`dimpy.calc_method.CalcMethod` for full documentation.

    **Example:** (this is the same as running the example in 
    ``DIMPy/examples/minimal_input.dimpy``::

        >>> import dimpy
        >>> nano = dimpy.Nanoparticle('Ag 0 0 0; Ag 0 0 1.89', verbose=0)
        >>> nano.build()
        >>> calc = dimpy.DDAs(nano)
        >>> calc.run()
        >>> calc.isotropic_polarizabilities[0]
        12.575719

    """

    interaction = 'DDA'
    model = 'PIM'
