def quick_test(fd=False):
    """\
    Creates a temporary sample input file, runs it, then returns a DIM
    object with the collected output data.

    Parameters
    ----------
    fd : :obj:`bool`
        If :obj:`True`, runs a frequency-dependent calculation, otherwise
        it runs a static calculation.

    Returns
    -------
    instance : :class:`DIM`
        A :class:`DIM` class instancd with the filled with the output data
        of the calculation

    """
    from .run import run
    from .dimclass import DIM
    from tempfile import mkstemp
    from textwrap import dedent
    # Frequency-dependent
    if fd: 
        fd_input = '.'.join([mkstemp()[1], 'dim'])
        with open(fd_input,  'w') as fdi:
            fdi.write(dedent("""\
                             ALGORITHM DIRECT
                             PIM
                             FREQUENCY EV 3.5 3.7
                             PRINT EFF
                             Ag
                               EXP Ag
                               RAD 1.4445
                             END
                             XYZ
                               Ag 0.0 0.0 0.0
                               Ag 0.0 0.0 4.0
                             END
                             """))
        fd_output = fd_input.replace('.dim', '.out')
        run(fd_input, fd_output, nolog=True)
        fd = DIM(fd_output)
        fd.collect()
        return fd
    # Static
    else:
        static_input = '.'.join([mkstemp()[1], 'dim'])
        with open(static_input,  'w') as si: 
            si.write(dedent("""\
                            ALGORITHM DIRECT
                            PIM
                            Ag
                              RAD 1.4445
                            END
                            XYZ
                              Ag 0.0 0.0 0.0
                              Ag 0.0 0.0 4.0
                            END
                            """))
        static_output = static_input.replace('.dim', '.out')
        run(static_input, static_output, nolog=True)
        static = DIM(static_output)
        static.collect(static_output)
        return static
