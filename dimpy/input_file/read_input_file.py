import os
import re

from input_reader import InputReader
import numpy as np

from ..tools.constants import EV2HART, NM2HART, HZ2HART, WAVENUM2HART
from ..tools.constants import ELEMENTS, NOCONV, HART2NM, BOHR2NM
from ..tools.constants import HART2EV, HART2WAVENUM, HART2HZ
from ..tools.constants import ANGSTROM2BOHR, ELEMENTS
from ..dimpy_error import DIMPyError


class ReadInput(object):
    """Read information from a formatted DIMPy input file
    using non-standard module :mod:`input_reader`

    :param filename: DIMPy input filename to read
    :type filename: str, optional

    **Examples**::

    >>> from dimpy import ReadInput
    >>> input_options = ReadInput('filename.dimpy').read()

    **or**::

    >>> from dimpy import ReadInput
    >>> input_options = ReadInput().read('filename.dimpy')

    """

    def __init__(self, filename=None):
        """Initialize the class."""

        self.filename = filename

        # Initializes reader for a DIMPy input
        # Valid comment characters are `::`, `#`, `!`, and `//`
        reader = InputReader(comment=['!', '#', '::', '//'],
                             case=False, ignoreunknown=True)

        reader.add_line_key('title', type=[], glob={'len': '*', 'join': True, },
                            case=True)

        # debug and verbose keys
        reader.add_boolean_key('debug', action=True, default=False)
        reader.add_line_key('verbose', type=int, default=2)

        self.reader = reader

        self._check_nanoparticle_block()
        self._check_method_block()

    def _check_nanoparticle_block(self):
        """Add the nanoparticle block keys."""
        reader = self.reader

        # initialize the nanoparticle block in the input file
        nano = reader.add_block_key('nanoparticle', end='endnanoparticle',
                                    ignoreunknown=True, required=True, case=False)

        # read in coords either explicitly or via an xyz file
        coords = nano.add_mutually_exclusive_group(required=True)
        coords.add_line_key('xyzfile', type=str, case=True)
        atoms = coords.add_block_key('atoms')
        coord = re.compile(r"""
                             \s*           # Leading whitespace
                             \d*           # Atomic ## (optional)
                             .*            # Separator
                             ([A-Z][a-z]?) # The element
                             \s+           # Whitespace
                             ([-0-9.]+)    # X coord
                             \s+           # etc...
                             ([-0-9.]+)
                             \s+
                             ([-0-9.]+)
                            """, re.VERBOSE)
        atoms.add_regex_line('coords', coord, repeat=True)

        # Periodic Boundary Conditions
        pbc = nano.add_block_key('pbc')
        pbc.add_regex_line('vector',
                           '\s*(\-?\d+.?\d*)\s+(\-?\d+.?\d*)\s+(\-?\d+.?\d*)',
                           repeat=True)

        # coordinate / pbc lattice units
        nano.add_line_key('unit', type=('bohr', 'au', 'b', 'a', 'ang',
                          'angstrom'), default='angstrom')

        # Atom parameters
        # is of the form 'ATOMPARAM Xx KEY ##.###'
        nano.add_line_key('atomparam', repeat=True, case=True,
                          type=[re.compile(r'[A-z][a-z]?', re.VERBOSE),
                                ('rad', 'exp'),
                                re.compile(r'\S+', re.VERBOSE)])

    # define solvers here
    solvers = ('direct', 'solve', 'bicg', 'bicgstab',
               'cgs', 'gmres', 'lgmres', 'qmr', 'gcrotmk')
    """Available options for the 'Solver' key in the DIMPy input."""

    def _check_method_block(self):
        """Add the calculation block keys."""
        reader = self.reader

        # initialize the method block in the input file
        method = reader.add_block_key('method', end='endmethod',
                                      ignoreunknown=True, required=True)

        # DIM or DDA?
        method.add_line_key('interaction', type=('dda', 'dim'), required=True)

        # k-vector direction (when present, this is a retardation calculation)
        method.add_line_key('kdir', type=('x', 'y', 'z'), default=None)

        # Frequency control
        freq = method.add_mutually_exclusive_group()
        freq.add_line_key('frequency',
                          type=('ev', 'nm', 'hz', 'cm-1', 'hartree', 'au'),
                          glob={'len': '+', 'type': float})
        freq.add_line_key('freqrange',
                          type=[('ev', 'nm', 'hz', 'cm-1', 'hartree', 'au'),
                                float, float, int])

        # Solver algorithm
        # These algorithms are all found in the `scipy.sparse.linalg` module
        method.add_line_key('solver', type=self.solvers, default='gcrotmk')

    def _read_nanoparticle_block(self, input_options=None):
        """Read the nanoparticle block in a DIMPy input file."""
        if input_options is None:
            input_options = self.input_options
        options = input_options.nanoparticle

        # read atom_params
        atom_params = {}
        params = options.atomparam
        if params is not None:
            for i in range(len(params)):
                atom = params[i][0].capitalize()
                key = params[i][1]
                value = list(params[i][2:])
                if len(value) == 1:
                    try:
                        value = float(value[0])
                    except ValueError:
                        value = value[0]
                else:
                    for j in range(len(value)):
                        try:
                            value[j] = float(value[j])
                        except ValueError:
                            pass
                try:
                    atom_params[atom][key] = value
                except KeyError:
                    atom_params[atom] = {key: value}
        options.atom_params = atom_params

        # set pbc
        if options.pbc is not None:
            lattice = []
            for r in options.pbc.vector:
                lattice.append(np.array([r.group(1), r.group(2),
                    r.group(3)], dtype=float))
            options.pbc = np.array(lattice, dtype=np.float32)

        # format the atoms to be used by the nanoparticle
        if options.atoms is not None:

            coords = options.atoms.coords
            atoms = [[coord.group(1), [float(coord.group(i)) for i in range(2,5)]]
                     for coord in coords]
            options.atoms = atoms

        else:
            options.atoms = options.xyzfile

    def _read_method_block(self, input_options=None):
        """Read the frequency from a DIMPy input file and convert
        to atomic units.
        """

        if input_options is None:
            input_options = self.input_options
        options = input_options.method

        conversion = {'ev': EV2HART,
                      'nm': NM2HART,
                      'hz': HZ2HART,
                      'cm-1': WAVENUM2HART,
                      'hartree': NOCONV,
                      'au': NOCONV}

        # convert individually given frequency
        if options.frequency is not None:
            convert = conversion[options.frequency[0]]
            freqs = []
            for freq in options.frequency[1:]:
                freqs.append(convert(freq))
            freqs = np.array(freqs, dtype=float)

        # expand the frequency range and convert appropriately
        elif options.freqrange is not None:
            convert = conversion[options.freqrange[0]]
            start, stop, num = options.freqrange[1:]
            freqs = convert(np.linspace(start, stop, num))
            freqs = np.sort(freqs) # sort in order of increasing energy

        # assume this is a static frequency
        else:
            freqs = np.zeros((1))

        options.freqs = freqs

    def read(self, filename=None):
        """Read the input and return the input option.

        :param filename: DIMPy input filename to read
        :type filename: str, optional

        :returns: ``input_reader`` object with input options

        """
        if filename is None:
            filename = self.filename
        # first check that file exists
        if filename is None:
            raise DIMPyError('No DIMPy input filename given!')
        elif not os.path.isfile(filename):
            raise DIMPyError(f'File `{filename}` does not exist!')

        reader = self.reader
        self.filename = filename
        self.input_options = reader.read_input(filename)
        self._read_nanoparticle_block()
        self._read_method_block()

        return self.input_options

