import re

from .constants import NOCONV, ANGSTROM2BOHR, ELEMENTS


def read_input_file(input_filename):
    """Reads information from a formatted input file."""

    from input_reader import InputReader

    # Initializes reader for a DIMPy input
    # Valid comment characters are `::`, `#`, `!`, and `//`
    reader = InputReader(comment=['!', '#', '::', '//'],
             case=False, ignoreunknown=True)

    #####################
    # Some general keys #
    #####################
    reader.add_line_key('title', type=[], glob={'len': '*', 'join': True, },
                        case=True)
    # TODO: debug currently does nothing
    reader.add_boolean_key('debug', action=True, default=False)

    ####################
    # Solver algorithm #
    ####################
    # These algorithms are all found in the `scipy.sparse.linalg` module
    algorithms = ('spsolve', 'spsolve_triangular', 'bicg', 'bicgstab',
                  'cg', 'cgs', 'gmres', 'lgmres', 'minres', 'qmr', 'direct')
    reader.add_line_key('algorithm', type=algorithms, default='spsolve')

    reader.add_line_key('niter', type=int, default=0)
    reader.add_line_key('grid', type=(3, 5, 7, 9, 11,
                                      'extracoarse', 'coarse', 'medium',
                                      'fine', 'extrafine'), default=7)
    reader.add_line_key('atomspercell', type=int, default=4)
    reader.add_boolean_key('coorddepend', action=True, default=False)

    # k-vector direction (when present, this is a retardation calculation)
    reader.add_line_key('kdir', type=('x', 'y', 'z'), default=None)

    #define media refactive index and scale factor for bulk plasma osillation
    reader.add_line_key('nsolv', type=float, default=1.0)
    reader.add_line_key('gfactor', type=float, default=7.0)

    # TODO: noninteracting currently does nothing
    reader.add_boolean_key('noninteracting', action=True, default=False)
    reader.add_boolean_key('bohr', action=NOCONV, default=ANGSTROM2BOHR,
                           dest='distconv')
    reader.add_line_key('tolerance', type=float, dest='tol', default=1.0E-6)
    reader.add_boolean_key('dda', action=True, default=False)
    reader.add_boolean_key('noprecon', dest='precon', action=False,
                           default=True)
    reader.add_line_key('precon', type=('none', 'perfect', 'jacobi', float),
                        default='none')
    reader.add_line_key('direction', glob={'len':'*'}, default=None)

    volume = reader.add_mutually_exclusive_group()
    volume.add_line_key('volume', type=float, default=-1.0)
    volume.add_line_key('multiplier', type=[float, float, float],
                        default=(1.0, 0.5, 1.0))
    reader.add_line_key('scscale', type=float, default=1.0)

    ####################
    # Printing control #
    ####################
    reader.add_line_key('printlevel', type=(0, 1, 2, 3), default=0)
    reader.add_line_key('print', type=('atmdip', 'coords', 'energy',
                                       'pol', 'timing', 'timingverbose',
                                       'timingveryverbose',
                                       'input', 'eff'), repeat=True)
    reader.add_line_key('noprint', type=('atmdip', 'coords', 'energy',
                                         'pol', 'timing', 'timingverbose',
                                         'timingveryverbose',
                                         'input', 'eff'), repeat=True)

    # The calculation mode (PIM vs CPIM)
    # TODO: CPIM not implemented
    mode = reader.add_mutually_exclusive_group(required=True)
    mode.add_boolean_key('cpim', action=True, default=False)
    mode.add_boolean_key('pim', action=True, default=False)
   
    # Keys that depend on CPIM
    reader.add_boolean_key('nopol', action=True, default=False, depends='cpim')
    reader.add_boolean_key('nochar', action=True, default=False,
                           depends='cpim')
    reader.add_boolean_key('nocross', action=True, default=False,
                           depends='cpim')
    reader.add_line_key('totalcharge', type=float, default=0.0, depends='cpim')

    # Frequency control
    freq = reader.add_mutually_exclusive_group()
    freq.add_line_key('frequency',
                      type=('ev', 'nm', 'hz', 'cm-1', 'hartree', 'au'),
                      glob={'len': '+', 'type': float})
    freq.add_line_key('freqrange',
                      type=[('ev', 'nm', 'hz', 'cm-1', 'hartree', 'au'),
                            float, float, int])

    ##################################
    # The elemental parameter blocks #
    ##################################
    unitglob = {'len': '?',
                'type': ('ev', 'hz', 'cm-1', 'hartree', 'au'),
                'default': 'hartree'}
    for el in ELEMENTS:  # Elements defined from constants module
        e = reader.add_block_key(el)
        die_or_pol = e.add_mutually_exclusive_group()
        die_or_pol.add_boolean_key('dielectric', action=True, default=False)
        die_or_pol.add_boolean_key('polarizability', action=True, default=False)

        # define the coordination dependent element, element binding to plasmonics, and the minimal and maximal radii of the bonds
        e.add_boolean_key('coorddepend', action=True, default=False)
        e.add_boolean_key('binding', action=True, default=False)
        e.add_boolean_key('static', action=True, default=False)
        e.add_line_key('romin',  type=float, default=2.5)
        e.add_line_key('romax',  type=float, default=3.5)

        e.add_line_key('pol',   type=float)
        e.add_line_key('cap',   type=float)
        e.add_line_key('om1',   type=float)
        e.add_line_key('om2',   type=float)
        e.add_line_key('gm1',   type=float)
        e.add_line_key('gm2',   type=float)
        e.add_line_key('size',  type=float)
        e.add_line_key('rad',   type=float, required=True)
        e.add_line_key('rmin',  type=float, default=3,)
        e.add_line_key('rmax',  type=float, default=5)
        e.add_line_key('rsurf', type=float, default=2.2)
        e.add_line_key('rbulk', type=float, default=1.4445)
        e.add_line_key('cnmax',  type=float,   default=12)
        e.add_line_key('exp',   type=str, case=True, default=None)
        e.add_line_key('bound', type=float, default=1.0)
        e.add_line_key('sc',    type=float, default=-110)
        e.add_line_key('wpin',  type=float, default=1)
        e.add_line_key('drude', type=[float, float],
                       glob=unitglob)
        e.add_line_key('lrtz',  type=[float, float, float], repeat=True,
                       glob=unitglob)
        e.add_line_key('lrtz1', type=[float, float, float], repeat=True,
                       glob=unitglob)
        e.add_line_key('lrtz2', type=[float, float, float, float], repeat=True,
                       glob=unitglob)
        e.add_line_key('lrtz3', type=[float, float, float], repeat=True,
                       glob=unitglob)
        e.add_line_key('fermi', type=float, default=0.0, depends='drude')
        e.add_line_key('spillout', type=float, default=0.0)

    ################################
    # Periodic Boundary Conditions #
    ################################
    pbc_string = re.compile(r"""
                             \s*
                             ((?i)pbc)
                             (\s+[0-9.]+)
                             (\s+[0-9.]+)?
                             (\s+[0-9.]+)?
                             """, re.VERBOSE)
    reader.add_regex_line('pbc', pbc_string)

    # Read the input and return the options read in
    input_options = reader.read_input(input_filename)
    return input_options

