from copy import deepcopy
from os.path import splitext
from .constants import elem, atomic_mass, atomic_radius, \
    HART2EV, HART2NM, AVOGADRO, PI, LIGHT_AU, NOCONV, ANGSTROM2BOHR, \
    ANGSTROM2NM, ANGSTROM2CM, ANGSTROM2M
from numpy import where, concatenate, vectorize, average, dot, eye, array, \
    sum, trace, empty, zeros, absolute, power, diagflat, zeros_like, \
    asarray, argsort, errstate
from numpy import sqrt as npsqrt
from numpy.linalg import eig, LinAlgError
from math import radians, cos, sin, sqrt
from .dimtools import calc_bonds, minmax_pdist
import sys
import os
import re


class DIMError(Exception):
    '''Error class for DIM errors.

    Parameters
    ----------
    msg : :obj:`str`
        The message to give to the user.

    Examples
    --------

        >>> import dim
        >>> try:
        ...     filedata = dim.quick_test('file.dim')
        ... except dim.DIMError as d:
        ...     sys.exit(str(d))

    '''
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class DIM(object):
    '''\
    Stores the DIM output data.

    The :class:`DIM` class reads in the data of a DIM output file and
    provides a high-level interface to help with data manipulation
    and displaying the data.

    Parameters
    ----------
    name : :obj:`str`
        The name of the DIM output file to associate with this class.

    Attributes
    ----------
    filename : :obj:`str`
        The name if the DIM output file associated with this class.
    filetype : :obj:`str`
        The file type, { 'input', 'output' }
    key : :obj:`dict`
        A dictionary containing the input keys from the input block
    calctype : :obj:`str`
        The type of calculation that was run { 'STATIC', 'FD' }
    natoms : :obj:`int`
        The number of atoms in the nanoparticle.
    coordinates : :obj:`numpy.ndarray` of shape ``natoms`` x 3
        The cartesian coordinates of the atoms in the nanoparticle
    atoms : :obj:`numpy.ndarray` of length ``natoms``
        The name of each atom in the nanoparticle
    elements : :obj:`set`
        The elements in the nanoparticle
    npol : :obj:`int`
        The number of frequency-dependent (FD) polarizabilities calculated.
    e_frequencies : :obj:`numpy.ndarray` of length ``npol``
        The electric frequencies the polarizabilities were calculated at
    polarizability : :obj:`numpy.ndarray` of shape ``npol`` x 3 x 3
        The polarizability tensor for each frequency.
    efficiencies : :obj:`numpy.ndarray` of shape ``npol`` x 3
        The spectral efficiencies.  For each frequency, the first
        element is the scattering, the second is the absorbance, and
        the third is the exctinction. This is only valid for FD
        calculations.
    cross_sections : :obj:`numpy.ndarray` of shape ``npol`` x 3
        The spectral cross sections.  For each frequency, the first
        element is the scattering, the second is the absorbance, and the
        third is the exctinction. This is only valid for FD calculations.
    dipoles : :obj:`numpy.ndarray` of shape ``npol`` x ``natoms`` x 3
        The dipole on each atom of the nanoparticle.
        Only available if these were printed in the output file
    charges : :obj:`numpy.ndarray` of shape ``npol`` x ``natoms``
        The charge on each atom of the nanoparticle.
        Only available if these were printed in the output file
        and the calculation was CPIM
    start : :obj:`datetime.datetime`
        The starting time of the calculation
    real_time : :obj:`datetime.timedelta`
        The wall time of the calculation
    cpu_time : :obj:`datetime.timedelta`
        The cpu time of the calculation
    routine_time : :obj:`dict`
        The time it took to perform each routine in the DIM calculation.
    termination : :obj:`str`
        A message that indicates how the DIM calculation ended
    host : :obj:`str`
        The host the calculation was run on
    nprocs : :obj:`int`
        The number of processors used to perform the calculation

    Methods
    -------
    tensor_isotropic(tensors)
        Return the isotropic polarizability for each given polarizability
        tensor
    tensor_anisotropic2(tensors)
        Return the anisotropic polarizability squared for each given
        polarizability tensor

    .. note:: The above two methods are static methods of the DIM class.

    Examples
    --------
    When you instantiate a new DIM class, every field is empty except
    for the filename and filetype attributes

        >>> import dim
        >>> static = dim.quick_test()
        >>> static.filename
        '...out'
        >>> static.filetype
        'output'

    .. note:: All :class:`DIM` examples use the :func:`quick_test` function
              to generate the test cases.

    '''

    def __init__(self, name):
        '''\
        Initiallize the DIM class.

        :param name: The name of the DIM output file to associate with this class.
        '''
        # Find extention
        ftype = splitext(name)[1]
        if ftype not in ('.out', '.inp', '.dim'):
            raise ValueError(ftype+' not a recognized extention')
        self.filetype = 'output' if ftype == '.out' else 'input'
        self.filename = name

        # List of input keys.
        self.blockkeys = ('XYZ',)
        # Each element is a possible key
        for e in elem[1:]:
            self.blockkeys += (e.upper(),)
        self.linekeys = ('TOLERANCE', 'TOTALCHARGE', 'DAMPING', 'PRINTLEVEL',
                         'ALGORITHM', 'NOPRINT', 'PRINT', 'FREQRANGE',
                         'FREQUENCY', 'OUTPUT',)
        self.singlekeys = ('CPIM', 'PIM', 'NOPOL', 'NOCHAR', 'NONINTERACTING',
                           'DEBUG', 'BOHR',)

        self.calctype = set()
        self.subkey = set()
        self.key = {}
        self.natoms = None
        self.coordinates = None
        self.atoms = None
        self.elements = None
        self.npol = None
        self.e_frequencies = None
        self.polarizability = None
        self.efficiencies = None
        self.cross_sections = None
        self.dipoles = None
        self.charges = None
        self.start = None
        self.real_time = None
        self.cpu_time = None
        self.routine_time = None
        self.termination = None
        self.host = None
        self.nprocs = None
        self._diagonalized = False
        self._minmax = None
        self._bonds = None

    def collect(self, abort=False):
        '''\
        Read the data from a :class:`DIM` input or output file

        Parameters
        ----------
        abort : :obj:`bool`
            If :obj:`True`, the collector will raise an error when an collection
            error is encountered.  If :obj:`False`, the error will be
            ignored and it will continue collection the remainder of the file.

        Raises
        ------
        DIMError
            An expected section of the output was missing.
            (Other exceptions may also be raised, but are difficult to predict
            because of the nature of parsing an open-ended text file.)

        Examples
        --------
        The :meth:`collect()` method can collect output files:

            >>> import dim
            >>> # .collect() is called as the last step of quick_test()
            >>> # Example
            >>> # run(inputfile, outputfile)
            >>> # data = DIM(outputfile)
            >>> # data.collect()
            >>>
            >>> static = dim.quick_test()
            >>> static.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> static.atoms # doctest: +NORMALIZE_WHITESPACE
            array(['Ag', 'Ag'], dtype='|S2')
            >>> static.natoms
            2
            >>> static.elements
            set(['Ag'])
            >>> static.nelements
            1
            >>> static.key['XYZ']
            ('Ag 0.0 0.0 0.0', 'Ag 0.0 0.0 4.0')

        Note that the results of a static and frequency-dependent calculation will be
        in a slightly different format.

            >>> static.npol
            1
            >>> static.e_frequencies
            array([ 0.])
            >>> static.polarizability
            array([[[ 71.324,   0.   ,   0.   ],
                    [  0.   ,  71.324,   0.   ],
                    [  0.   ,   0.   ,  93.755]]])
            >>> static.efficiencies
            >>> static.cross_sections
            >>> fd = dim.quick_test(fd=True)
            >>> fd.npol
            2
            >>> fd.e_frequencies
            array([ 0.128623,  0.135972])
            >>> fd.polarizability
            array([[[ 200.794+264.06j ,    0.000  +0.j   ,    0.000  +0.j   ],
                    [   0.000  +0.j   ,  200.794+264.06j ,    0.000  +0.j   ],
                    [   0.000  +0.j   ,    0.000  +0.j   , -188.906+296.316j]],
            <BLANKLINE>
                   [[ -81.283 +99.26j ,    0.000  +0.j   ,    0.000  +0.j   ],
                    [   0.000  +0.j   ,  -81.283 +99.26j ,    0.000  +0.j   ],
                    [   0.000  +0.j   ,    0.000  +0.j   ,  -78.875 +57.359j]]])
            >>> # The following are only collected for FD
            >>> # and if 'PRINT EFF' was included in input file
            >>> fd.efficiencies
            array([  8.09150000e-02,   1.84020000e-12])
            >>> fd.cross_sections
            array([  7.22310000e-04,   1.64260000e-14])

        '''

        # Collect from file
        try:
            # Set the abort flag
            self._abort = abort
            # Read in file
            from .read_file import read_file
            f, indices = read_file(self)
            # Read input block
            from .input_block import collect_input
            collect_input(self, f, indices)
            # Determine calculation type
            self.__det_calc_type()
            # Techical properties
            from .dimproperties import collect_technical
            collect_technical(self, f, indices)
            # Collect all exciting information
            from .dimproperties import collect_dim
            collect_dim(self, f, indices)
            # Collect timing
            from .dimproperties import collect_timing
            collect_timing(self, f, indices)
        # If a known error occured
        except DIMError as d:
            if abort:
                raise DIMError(str(d) + ': ' + self.filename)
        # If a known unknown error occured.
        except (IndexError, StopIteration, KeyError):
            msg = 'Did the calculation end prematurely?'
            if abort:
                print('/\\'*int(len(msg)/2), file=sys.stderr)
                print(msg, file=sys.stderr)
                print('/\\'*int(len(msg)/2), file=sys.stderr)
                print(file=sys.stderr)
                raise
        except ValueError:
            msg = 'Possibly a number too large for the format (********)\n'
            msg += 'or a string and number running together (WORD1.785)'
            if abort:
                print('/\\'*27, file=sys.stderr)
                print(msg, file=sys.stderr)
                print('/\\'*27, file=sys.stderr)
                print(file=sys.stderr)
                raise

    def copy(self):
        '''\
        Returns a copy of the current instance.
        Equivalent to :obj:`copy.deepcopy` (self).

        Returns
        -------
        copy : :class:`DIM`
            A copy of the current :class:`DIM` instance.

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> example1 = static
            >>> example1 is static
            True
            >>> example2 = static.copy()
            >>> example2 is static
            False

        '''
        return deepcopy(self)

    def printCoords(self, mode=None, a1=1, a2=None, file=None, latex=False):
        '''Prints the geometry coordinates to screen.

        Parameters
        ----------
        mode : {'xyz', 'xyz_title', 'num', 'dimblock'}
            Determines how the coordinates will be printed.  If
            omitted, the coordinates will be printed without numbers.

            Valid options are:

                - 'num':       Prints the number of each atom with the element.
                - 'xyz':       Prints the total number of atoms, then a space, then the coordinates.
                - 'xyz_title': Same as 'xyz', but prints the title instead of a space, if a title is available.
                - 'dimblock':  Same as 'xyz', but omits the space altogether.

        a1 : :obj:`int`
            The number of the lowest atom to print
        a2 : :obj:`int`, :obj:`None`
            The number of the highest atom to print.
            If given :obj:`None`, the number of the last atom is used
        file : :obj:`str`, :obj:`file`
            Where to print to.  If omitted, it will print to standard
            output.  You may give the name of a file or an already open
            :obj:`file` object to write to. Or, if `file` is 'xyz', then
            it will create a ``.xyz`` file based on the :attr:`filename`
            attribute and print there.
        latex : :obj:`bool`
            Causes the output to be printed in a LaTeX table type of format.

        Raises
        ------
        ValueError
            An invalid mode is entered

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.printCoords()
            Ag    0.00000000    0.00000000    0.00000000
            Ag    0.00000000    0.00000000    4.00000000
            >>> static.printCoords(mode='num')
            1 Ag    0.00000000    0.00000000    0.00000000
            2 Ag    0.00000000    0.00000000    4.00000000
            >>> static.printCoords(mode='xyz')
            2
            <BLANKLINE>
            Ag    0.00000000    0.00000000    0.00000000
            Ag    0.00000000    0.00000000    4.00000000
            >>> static.printCoords(latex=True)
             Ag   &    0.000000 &    0.000000 &    0.000000\\\\
             \\hline
             Ag   &    0.000000 &    0.000000 &    4.000000\\\\
             \\hline

        '''
        valid_modes = ('xyz', 'xyz_title', 'num', 'dimblock')
        if mode is not None and (mode not in valid_modes):
            raise ValueError('printCoords(): Invalid mode: '+mode)

        # Create an xyz file if file is True
        if file == 'xyz':
            # Split current extention off filename, then replace with .xyz
            file = '.'.join([os.path.splitext(self.filename)[0], 'xyz'])
            # Open, and remember we need to close after
            file = open(file, 'w')
            closebool = True
        # If file is None, use standard out
        elif file is None:
            file = sys.stdout
            closebool = False
        # Otherwise, try to open the file
        else:
            try:
                file = open(file, 'w')
            # If it fails, then it was already an open file
            except TypeError:
                closebool = False
            # If it suceeds, then remmeber that we must close the file
            else:
                closebool = True

        # Convert atom number to index
        a1 -= 1
        # Default to all.
        if a2 is None:
            a2 = len(self.atoms)

        # Determine how to format the line, and make numbers correct
        if mode == 'num':
            # Find string length of largest number to be printed and make that
            # number a string
            maxlen = str(len(str((a2 - a1) + 1)))
            fmt = '{0:<'+maxlen+'} {1:<2}{2[0]:14.8f}{2[1]:14.8f}{2[2]:14.8f}'
            num = 1
        else:
            if latex:
                fmt = '{0:>3}{2:>4}{1[0]:12.6f}{2:>2}{1[1]:12.6f}{2:>2}{1[2]:12.6f}{3:<2}'
            else:
                fmt = '{0:<2}{1[0]:14.8f}{1[1]:14.8f}{1[2]:14.8f}'

        # Make number of atoms
        natoms = len(self.atoms[a1:a2])

        # Print the number of atoms if the mode calls for it
        if mode in ('xyz', 'xyz_title', 'dimblock'):
            print(natoms, file=file)

        # Print a space if 'xyz', or if 'xyz_title' and there is no title
        if mode == 'xyz' or (mode == 'xyz_title' and ('title' not in self or not self.title)):
            print(file=file)
        # Print the title if 'xyz_title' and there is a title
        elif mode == 'xyz_title':
            print(self.title, file=file)

        # Print the coordinates
        for i in range(a1, a2):
            if mode == 'num':
                print(fmt.format(num, self.atoms[i], self.coordinates[i]), file=file)
                num += 1
            else:
                if latex:
                    print(fmt.format(self.atoms[i], self.coordinates[i], '&', r'\\'), file=file)
                    print(r' \hline', file=file)
                else:
                    print(fmt.format(self.atoms[i], self.coordinates[i]), file=file)

        # Close the file if appropriate
        if closebool:
            file.close()

    def writeCoords(self, a1=1, a2=None):
        '''This is a shortcut for
        :func:`printCoords(mode='xyz', file='xyz') <printCoords>`.

        This will write the coordinates to a .xyz file based on the
        name in the filename attribute.

        '''
        self.printCoords(mode='xyz', file='xyz', a1=a1, a2=a2)

    def writePDB(self, a1=1, a2=None):
        '''Writes the coordinates to a .pdb file with the same name
        as the current file.

        Parameters
        ----------
        a1 : :obj:`int`
            The number of the lowest atom to print
        a2 : :obj:`int`, :obj:`None`
            The number of the highest atom to print.
            If given :obj:`None`, the number of the last atom is used

        '''

        # Define title format
        ft = 'CMPND  {0}'
        # Define coordinate format
        fc = ('HETATM{0:>5d}{1:>3}   LIG     1    '
              '{2[0]:>8.3f}{2[1]:>8.3f}{2[2]:>8.3f}  1.00  0.00          '
              '{1:>2}  ')
        # Define bond format
        fb = 'CONECT{0:>5d}'

        # Convert atom number to index
        a1 -= 1
        # Default to all.
        if a2 is None:
            a2 = len(self.atoms)

        # Open the file as pdb with same name
        filename = '.'.join([os.path.splitext(self.filename)[0], 'pdb'])
        with open(filename, 'w') as fl:

            # pbd Starts with some info.  Give the title if there is one
            if 'title' in self and self.title:
                title = ft.format('MOLECULE: '+self.title)
            else:
                title = ft.format('UNNAMED')
            print(title, file=fl)

            # Place the coordinates in the file
            i = 1
            for atom, coord in zip(self.atoms[a1:a2], self.coordinates[a1:a2]):
                print(fc.format(i, atom, coord), file=fl)
                i += 1

            # Place the bonds in the file.  The bonding is redundant, so each
            # atom must be specified and bonds will be listed mutlitple times.
            for i in range(self.natoms):

                # Skip atoms we don't want to show a bond to
                if i < a1 or i > a2:
                    continue

                # Print the bonding keyword
                print(fb.format(i+1), end='', file=fl)

                # Find everywhere that this atoms is in the first column
                indx = where(self.bonds[:, 0] == i)[0]

                # Print off all atoms this one is bonded to
                for j in indx:
                    # Skip atoms we don't want to show a bond to
                    if self.bonds[j, 1] < a1 or self.bonds[j, 1] > a2:
                        continue
                    print('{0:>5d}'.format(self.bonds[j, 1]+1), end='', file=fl)

                # Repeat for the second column
                indx = where(self.bonds[:, 1] == i)[0]
                for j in indx:
                    # Skip atoms we don't want to show a bond to
                    if self.bonds[j, 0] < a1 or self.bonds[j, 0] > a2:
                        continue
                    print('{0:>5d}'.format(self.bonds[j, 0]+1), end='', file=fl)

                # New line
                print(file=fl)

            # End the file
            print('END   ', file=fl)

    def join(self, other):
        '''Concatenates one molecule into the current one.

        The attributes that are concatenated are:

          - :attr:`~.coordinates`
          - :attr:`~.atoms`
          - ``natoms``
          - :attr:`~.elements`
          - :attr:`~.nelements`

        All other properties would be unphysical to concatenate
        and are emptied.

        Parameters
        ----------
        other : :class:`DIM`
            The other :class:`DIM` object to concatenate with this one

        Examples
        --------

            >>> import dim
            >>> np1 = dim.quick_test()
            >>> np1.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> np2 = dim.quick_test()
            >>> np2.translate_coordinates([3.0, 0.0, 0.0])
            >>> np2.coordinates
            array([[ 3.,  0.,  0.],
                   [ 3.,  0.,  4.]])
            >>> np1.join(np2)
            >>> np1.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.],
                   [ 3.,  0.,  0.],
                   [ 3.,  0.,  4.]])
            >>> np1.atoms
            array(['Ag', 'Ag', 'Ag', 'Ag'], 
                  dtype='|S2')

        '''
        self.coordinates = concatenate((self.coordinates, other.coordinates))
        self.atoms = concatenate((self.atoms, other.atoms))
        self.natoms = len(self.atoms)
        self.elements.update(other.elements)
        self.nelements = len(self.elements)
        self.empty(ignore=['coordinates', 'atoms', 'natoms',
                           'elements', 'nelements'])
        self._bonds = None
        self._minmax = None

    def find_center(self, type='geometrical'):
        '''Locates the center of the nanoparticle.

        Parameters
        ----------
        type : {'geometrical', 'center-of-mass'}
            The type of the center-of-mass to locate

        Returns
        -------
        center : :obj:`float`
            The center-of-mass of the nanoparticle

        Raises
        -------
        ValueError
            An invalid type is given.

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> static.find_center()
            array([ 0.,  0.,  2.])

        '''

        if type == 'center-of-mass':
            wt = self.masses
        elif type == 'geometrical':
            wt = None
        else:
            raise ValueError('find_center(): Invalid type: '+type)

        # Now find the center of mass
        return average(self.coordinates, axis=0, weights=wt)

    def shift_to_origin(self, type='geometrical'):
        '''\
        Shifts the coordinates in place so that the center of the
        molecule is at the origin.

        Parameters
        ----------
        type : {'geometrical', 'center-of-mass'}
            Determines the center type.  See :meth:`find_center`.

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> static.shift_to_origin()
            >>> static.coordinates
            array([[ 0.,  0., -2.],
                   [ 0.,  0.,  2.]])

        '''
        self.translate_coordinates(-self.find_center(type))

    def radii(self, set):
        '''\
        Returns the radii for each atom in the nanoparticle.

        Parameters
        ----------
        set : {'vis', 'vdw'}
            Specifies from which set of data you wish
            to collect the radii.  The options are:

                -vis: These radii are good for molecular visualizations but are not physical
                -vdw: These use the van Der Waals radii and are intended to be physical, however not all elements are available.

        Returns
        -------
        radii : :obj:`numpy.ndarray` of length ``natoms``.
            The radii for each atom

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.atoms
            array(['Ag', 'Ag'], 
                  dtype='|S2')
            >>> static.radii('vis')
            array([ 1.44,  1.44])

        '''

        # Find radii for each atom
        rad = zeros(self.natoms)
        for i in range(self.natoms):
            rad[i] = atomic_radius(self.atoms[i], set)
        return rad

    def translate_coordinates(self, transvec):
        '''\
        Translates the coordinates in place.

        Parameters
        ----------
        transvec : :obj:`numpy.ndarray` of length 3
            The translation vector by which to translate the coordinates

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> static.translate_coordinates([3.0, 1.0, 0.0])
            >>> static.coordinates
            array([[ 3.,  1.,  0.],
                   [ 3.,  1.,  4.]])

        '''
        self.coordinates += transvec

    def rotate_coordinates(self, rotmat=None, angle=None, dir=None, rad=False):
        '''Rotates the coordinates.

        Parameters
        ----------
        rotmat : :obj:`numpy.ndarray` of shape 3 x 3
            The rotation matrix to use to rotate the nanoparticle.
            Mutually exclusive with the `angle` parameter.
        angle : :obj:`float`, :obj:`list` of :obj:`float`
            An angle by which to rotate the coordinates.  If given
            as a :obj:`list`, the coordinates are rotated by each
            angle in the order they are given.
            Mutually exclusive with the `rotmat` parameter.
        dir : {'X', 'Y', 'Z'}
            The direction in which to rotate the coordinates
            for a given angle. Required when the `angle` parameter
            is given. If `angle` is given as a :obj:`list`, `dir`
            must be a :obj:`list` of the same length
        rad : :obj:`bool`
            Tells if the angle is in radians or note.

        Raises
        ------
        ValueError
            Invalid arguments are given

        Examples
        --------

            >>> import dim
            >>> import numpy
            >>> np1 = dim.quick_test()
            >>> np1.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> np1.rotate_coordinates(angle=90, dir='X')
            >>> numpy.around(np1.coordinates)
            array([[ 0.,  0.,  0.],
                   [ 0.,  4.,  0.]])
            >>> np2 = dim.quick_test()
            >>> np2.rotate_coordinates(angle=[45, 45], dir=['X', 'Z'])
            >>> numpy.around(np2.coordinates, decimals=5)
            array([[ 0.     ,  0.     ,  0.     ],
                   [ 0.     ,  2.82843,  2.82843]])
            >>> np3 = dim.quick_test()
            >>> np3.rotate_coordinates(rotmat=numpy.array([[0.5, 1.0, 1.0],
            ...                                            [1.0, 0.5, 0.0],
            ...                                            [1.0, 0.0, 0.5]]))
            >>> numpy.around(np3.coordinates, decimals=5)
            array([[ 0.,  0.,  0.],
                   [ 4.,  0.,  2.]])

        '''

        if rotmat is None and angle is None:
            ValueError("rotate_coordinates(): "
                       "Must choose one of 'rotmat' or 'angle'")
        if rotmat is not None and angle is not None:
            ValueError("rotate_coordinates(): "
                       "Must choose only one of 'rotmat' or 'angle'")

        # Create matrix if not given explicitly
        # Create a general 3-D rotation
        if rotmat is None:
            if dir is None:
                raise ValueError("rotate_coordinates(): "
                                 "'dir' must not be empty with 'angle'")
            if isinstance(angle, list) and not isinstance(dir, list):
                raise ValueError("rotate_coordinates(): "
                                 "'angle' and 'dir' must both be either "
                                 "a 'list' or 'str'")
            if not isinstance(angle, list) and isinstance(dir, list):
                raise ValueError("rotate_coordinates(): "
                                 "'angle' and 'dir' must both be either "
                                 "a 'list' or 'str'")
            # Make sure lists are the same length, and turn non-list in to list
            if isinstance(angle, list):
                if len(angle) != len(dir):
                    raise ValueError("rotate_coordinates(): "
                                     "'angle' and 'dir' must be the same length")
            else:
                angle = [angle]
                dir = [dir]

            # Generate the general rotation matrix
            rotmat = eye(3)
            for a, d in zip(angle, dir):
                if not rad:
                    a = radians(a)
                c = cos(a)
                s = sin(a)
                if d.lower() == 'x':
                    temp = array([[ 1,  0,  0 ],
                                  [ 0,  c, -s ],
                                  [ 0,  s,  c ]], dtype=float)
                elif d.lower() == 'y':
                    temp = array([[ c,  0,  s ],
                                  [ 0,  1,  0 ],
                                  [-s,  0,  c ]], dtype=float)
                elif d.lower() == 'z':
                    temp = array([[ c, -s,  0 ],
                                  [ s,  c,  0 ],
                                  [ 0,  0,  1 ]], dtype=float)
                else:
                    raise ValueError("Unknown value for 'dir': "+str(d))

                # Create the rotmat in rotation order. Not that this is
                # backwards from what you would expect because internally the
                # rotation is done with the coordinates first.
                rotmat = dot(temp, rotmat)

        # Rotate. Use fast MKL routine to do so
        #self.coordinates = rotate(rotmat, self.coordinates)
        self.coordinates = dot(self.coordinates, rotmat)

    def order_coords(self, atom=None, coord=None):
        '''\
        Reorders the coordinates according to proximity to either
        a specific atom or a point in space.

        Parameters
        ----------
        atom : :obj:`int`
            The atom to reorder by.
        coord : :obj:`numpy.ndarray` of length 3
            The point in space to reorder with respect to.

        Raises
        ------
        ValueError
            One and only one of `atom` and `coord` is not given.

        Examples
        --------

            >>> import dim
            >>> np1 = dim.quick_test()
            >>> np1.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> np1.order_coords(coord=[0.0, 0.0, 5.0])
            >>> np1.coordinates
            array([[ 0.,  0.,  4.],
                   [ 0.,  0.,  0.]])
            >>> np2 = dim.quick_test()
            >>> np2.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> np2.order_coords(atom=2)
            >>> np2.coordinates
            array([[ 0.,  0.,  4.],
                   [ 0.,  0.,  0.]])

        '''

        if atom is None and coord is None:
            raise ValueError('order_coords(): Must choose one of atoms or coords.')
        if atom and coord:
            raise ValueError('order_coords(): Cannot use both atoms and coords.')

        # First determine the distance from the points
        if atom is not None:
            c = self.coordinates[atom-1]
        else:
            c = coord

        # Sort the distances and return the sorted indices.
        index = argsort(npsqrt(sum((c - self.coordinates)**2, axis=1)))
        # Sort atoms coodinates and modes.
        self.atoms = self.atoms[index]
        self.coordinates = self.coordinates[index]
        self._bonds = None

    def printTensor(self, iso=False, ani=False, unit='au', p1=1, p2=None):
        '''\
        Pretty print the polarizability tensor to standard output.  One
        tensor is printed for each frequency.

        If the calculation was FD,
        then the real and imaginary tensors are printed alongside each other.

        Parameters
        ----------
        iso : :obj:`bool`
            Prints the isotropic polarizability along with the tensor
        ani : :obj:`bool`
            Prints the anisotropic polarizability along with the tensor
        unit : {'au', 'ev', 'nm'}
            Specifies the unit to print the frequency that corresponds
            to each polarizability tensor.
        p1 : :obj:`int`
            The number of the lowest polarizability tensor to print
        a2 : :obj:`int`, :obj:`None`
            The number of the highest polarizability tensor to print.
            If given :obj:`None`, the number of the last atom is used

        Raises
        ------
        DIMError
            No polarizabilities were collected.

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.printTensor(iso=True, ani=True, unit='ev')
            <BLANKLINE>
            Static Polarizability
                            X               Y               Z
               X          71.324           0.000           0.000
               Y           0.000          71.324           0.000
               Z           0.000           0.000          93.755
            <BLANKLINE>
            ISOTROPIC PART    =       78.801
            ANISOTROPIC PART  =       22.431
            <BLANKLINE>
            >>> fd = dim.quick_test(fd=True)
            >>> fd.printTensor(iso=True, ani=True, unit='ev')
            <BLANKLINE>
            FD Polarizability: Energy 3.50001 eV
                                           Real                                                Imaginary
                            X               Y               Z                      X               Y               Z
               X         200.794           0.000           0.000      X         264.060           0.000           0.000
               Y           0.000         200.794           0.000      Y           0.000         264.060           0.000
               Z           0.000           0.000        -188.906      Z           0.000           0.000         296.316
            <BLANKLINE>
            ISOTROPIC PART    =       70.894 + 274.812j
            ANISOTROPIC PART  =      391.033
            <BLANKLINE>
            FD Polarizability: Energy 3.69999 eV
                                           Real                                                Imaginary
                            X               Y               Z                      X               Y               Z
               X         -81.283           0.000           0.000      X          99.260           0.000           0.000
               Y           0.000         -81.283           0.000      Y           0.000          99.260           0.000
               Z           0.000           0.000         -78.875      Z           0.000           0.000          57.359
            <BLANKLINE>
            ISOTROPIC PART    =     -80.4803 +  85.293j
            ANISOTROPIC PART  =      41.9701
            <BLANKLINE>

        '''
        if 'polarizability' not in self:
            raise ValueError('printTensor(): '
                             'No polarizabilities were collected.')
        # Convert pol number to index
        p1 -= 1
        # Default to all.
        if p2 is None:
            p2 = self.npol

        # Set the unit and energy type
        if not re.match(r'au|ev|nm', unit, re.I):
            raise ValueError('printTensor() : Invalid unit: '+unit)
        else:
            unit = {'au': 'a.u.', 'ev': 'eV', 'nm': 'nm'}[unit.lower()]
            etyp = {'a.u.': 'Frequency',
                    'eV': 'Energy',
                    'nm': 'Wavelength'}[unit]

        # Set the tensor
        prop = 'Polarizability'

        # Start the formatting of the label
        if 'RAMAN' in self.calctype:
            lbl = self.v_frequencies
        else:
            with errstate(divide='ignore'):
                lbl = {'a.u.': self.e_frequencies,
                       'eV': HART2EV(self.e_frequencies),
                       'nm': HART2NM(self.e_frequencies)}[unit]

        # Diagonalized or not
        d = ', Diagonalized' if self._diagonalized else ''

        # Print selected tensors in class
        print()
        for n in range(p1, p2):

            t = self.polarizability[n]
            r = t.real
            i = t.imag

            # Prints out the proper heading depending on the energy type
            # and the unit.
            # First, make the number at most 7 digits with 'good' formatting
            # Strip off whitespace
            s = '{0:7g}'.format(lbl[n]).strip()
            # Now place this number in the heading
            head = 'FD {4}: {0} {1} {2}{3}'.format(etyp, s, unit, d, prop)
            # Replace the heading if this was a static calculation
            if s == '0':
                head = 'Static {0}'.format(prop)
            print(head)

            # Pretty-print the tensor
            if self.calctype == 'FD':
                label = '{0:>35}{1:35}{2:>22}'
                head = '{0:>17}{1:>16}{2:>16}{3:6}{0:>17}{1:>16}{2:>16}'
                fr = '{0:>4}{1[0]:16.3f}{1[1]:16.3f}{1[2]:16.3f}  '
                fi = '{0:>4}{1[0]:16.3f}{1[1]:16.3f}{1[2]:16.3f}'
                print(label.format('Real', '', 'Imaginary'))
                print(head.format('X', 'Y', 'Z', ''))
                print(fr.format('X', r[0, :]), fi.format('X', i[0, :]))
                print(fr.format('Y', r[1, :]), fi.format('Y', i[1, :]))
                print(fr.format('Z', r[2, :]), fi.format('Z', i[2, :]))
            else:
                head = '{0:>17}{1:>16}{2:>16}'
                f = '{0:>4}{1[0]:16.3f}{1[1]:16.3f}{1[2]:16.3f}'
                print(head.format('X', 'Y', 'Z'))
                print(f.format('X', r[0, :]))
                print(f.format('Y', r[1, :]))
                print(f.format('Z', r[2, :]))

            # Add the invariants at the end if requested
            if iso or ani:
                print()
            if iso:
                print('ISOTROPIC PART    = ', end='')
                iso = self.isotropic[n]
                if 'FD' in self.calctype:
                    print('{0:12g} + {1:7g}j'.format(iso.real, iso.imag))
                else:
                    print('{0:12g}'.format(iso))
            if ani:
                ani = self.anisotropic2[n]
                print('ANISOTROPIC PART  = {0:12g}'.format(sqrt(ani)))

            print()

    def printOptical(self, property='cross-section',
                     pathlength=None, concentration=None, unit='au',
                     absunit='angstroms'):
        '''Prints a list of optical properties.

        Parameters
        ----------
        property : {'absorption', 'absorptivitty', 'absorbance', 'transmittance'}
            Specifies the optical property to print.
            The choices are:

                - cross-section-> :meth:`cross_section`
                - absorptivitty -> :meth:`molar_absorptivitty`
                - absorbance -> :meth:`absorbance`
                - transmittance -> :meth:`transmittance`

        pathlength : :obj:`float`
            If the `property` is 'absorbance' or 'transmission',
            you will need to specify the pathlength for Beer's
            law in centimeters.
        concentration : :obj:`float`
            If the `property` is 'absorbance' or 'transmission',
            you will need to specify the concentration for Beer's
            law in molarity.
        unit : {'au', 'ev', 'nm'}
            This is the unit to print the frequencies. The options are:

                - 'au' -> atomic units, Hartrees
                - 'ev' -> electron volts
                - 'nm' -> nanometers

        absunit : :obj:`str`
            This is the unit that that absorption cross section
            is calculated at.  See :meth:`cross_section`.

        Raises
        ------
        ValueError
            Invalid arguments given

        Examples
        --------

            >>> import dim
            >>> fd = dim.quick_test(fd=True)
            >>> fd.printOptical(property='cross-section', unit='nm', absunit='cm')  # doctest: +NORMALIZE_WHITESPACE
            Wavelength (nm)   Absorption Cross-Section (cm^2/molecule)
                354.2395                       3.24138E-02
                335.0936                       1.06350E-02
            >>> fd.printOptical(property='transmittance', pathlength=1.0, concentration=1e-6)  # doctest: +NORMALIZE_WHITESPACE
            Pathlength=1.00 cm, Concentration=1.000E-06 M
             Frequency (a.u.)            Transmittance (unitless)
               1.2862E-01                      8.22670E-01
               1.3597E-01                      9.37962E-01

        '''

        # Prep for printing
        if property == 'cross-section':
            values = self.cross_section(unit=absunit)
            label = 'Absorption Cross-Section'
            if absunit == 'angstroms':
                u = unicode(u'\u212B^2/molecule')
            elif absunit == 'bohr':
                u = 'bohr^2/molecule'
            elif absunit == 'nm':
                u = 'nm^2/molecule'
            elif absunit == 'cm':
                u = 'cm^2/molecule'
            elif absunit == 'm':
                u = 'm^2/molecule'
        elif property == 'absorptivitty':
            values = self.molar_absorptivitty()
            label, u = 'Molar Absorptivitty', 'L mol^{-1} cm^{-1}'
        elif property == 'absorbance':
            values = self.absorbance(pathlength, concentration)
            label, u = 'Absorbance', 'unitless'
        elif property == 'transmittance':
            values = self.transmittance(pathlength, concentration)
            label, u = 'Transmittance', 'unitless'
        else:
            raise ValueError('printOptical(): '
                             'invalid property ('+property+')')

        # Choose frequency unit
        if not re.match(r'au|ev|nm', unit, re.I):
            raise ValueError('printTensor() : Invalid unit: '+unit)
        unit = {'au': 'a.u.', 'ev': 'eV', 'nm': 'nm'}[unit.lower()]
        etyp = {'a.u.': 'Frequency',
                'eV': 'Energy',
                'nm': 'Wavelength'}[unit]
        frequencies = {'a.u.': self.e_frequencies,
                       'eV': HART2EV(self.e_frequencies),
                       'nm': HART2NM(self.e_frequencies)}[unit]

        # Title
        if property in ('absorbance', 'transmittance'):
            string = 'Pathlength={0:.2f} cm, Concentration={1:.3E} M'
            print(string.format(pathlength, concentration))
        string = unicode(u'{0:>10} {1:<6} {2:>24} {3:<20}')
        print(string.format(etyp, '('+unit+')', label, '('+u+')'))

        # A format string
        if unit == 'a.u.':
            fmt = '{0:^17.4E} {1:^45.5E}'
        else:
            fmt = '{0:^17.4f} {1:^45.5E}'

        # Print the properties
        for freq, val in zip(frequencies, values):
            print(fmt.format(freq, val))

    def cross_section(self, unit='angstroms'):
        '''Returns the absorption cross-section of the system.

        Parameters
        ----------
        unit : {'angstroms', 'nm', 'bohr', 'cm', 'm'}
            The unit in which the cross section will be returned

                - 'angstroms' -> angstroms^2/molecule
                - 'nm' -> nm^2/molecule
                - 'bohr' -> bohr^2/molecule
                - 'cm' -> cm^2/molecule
                - 'm' -> m^2/molecule

        Returns
        -------
        cross_section : :obj:`numpy.ndarray` of length :attr:`~.npol`
            The absorption cross section for each frequency

        Raises
        ------
        ValueError
            Invalid arguments given
        DIMError
            Not an FD calculation

        Examples
        --------

            >>> import dim
            >>> fd = dim.quick_test(fd=True)
            >>> fd.cross_section()
            array([ 3.24137681,  1.06350141])
            >>> fd.cross_section(unit='nm')
            array([ 0.03241377,  0.01063501])
            >>> fd.cross_section(unit='cm')
            array([  3.24137681e-16,   1.06350141e-16])

        '''

        # Assign the correct unit converter
        if unit.lower() == 'angstroms':
            conv = NOCONV
        elif unit.lower() == 'bohr':
            conv = ANGSTROM2BOHR
        elif unit.lower() == 'nm':
            conv = ANGSTROM2NM
        elif unit.lower() == 'cm':
            conv = ANGSTROM2CM
        elif unit.lower() == 'm':
            conv = ANGSTROM2M
        else:
            raise ValueError('cross_section(): Invalid unit ('+unit.lower()+')')

        if self.calctype == 'FD':
            iso = self.isotropic.imag
            # Conversion is performed twice because the property is squared
            return conv(conv((4 * PI * self.e_frequencies / LIGHT_AU) * iso))
        else:
            raise DIMError('cross_section(): Not a FD calculation')

    def molar_absorptivitty(self):
        '''Calculates the molar absorptivitty of the system in units of
        L mol^{-1} cm^{-1} or M^{-1} cm^{-1} (these are equivalent).

        Returns
        -------
        molar_absorptivitty : :obj:`numpy.ndarray` of length :attr:`~.npol`
            The molar absorptivitty for each frequency in

        Raises
        ------
        ValueError
            Invalid arguments given
        DIMError
            Not an FD calculation

        Examples
        --------

            >>> import dim
            >>> fd = dim.quick_test(fd=True)
            >>> fd.molar_absorptivitty()
            array([ 84774.41649881,  27814.63448711])

        '''
        from math import log as ln

        # Conversion factor to get from angstroms^2 to L/cm.
        # L == dm^3
        # dm^2 == 1E18 angstroms^2
        # dm^2 == 0.1 dm^3/cm
        # 0.1 dm^3/cm = 1E18 angstroms^2
        # 1E-19 = dm^3/(cm*angstroms^2)
        CONVFACTOR = 1E-19

        # Get the absorption cross-section in angstroms^2/molecule
        acs = self.cross_section()
        # Use this to calculate the molar absorptivity
        # LN(10) accounts for Beer's law
        return AVOGADRO * CONVFACTOR * acs / ln(10)

    def absorbance(self, pathlength, concentration):
        '''\
        Calculates the (unitless) absorbance of the system using
        Beer's law.

        Parameters
        ----------
        pathlength : :obj:`float`
            The pathlength in centimeters of the light
        concentration : :obj:`float`
            The solution concentration in molarity

        Returns
        -------
        absorbance : :obj:`numpy.ndarray` of length :attr:`~.npol`
            The unitless absorbance for each frequency

        Raises
        ------
        ValueError
            Invalid arguments given
        DIMError
            Not an FD calculation

        Examples
        --------

            >>> import dim
            >>> fd = dim.quick_test(fd=True)
            >>> fd.absorbance(1.5, 1.5e-6)
            array([ 0.19074244,  0.06258293])

        '''
        eps = self.molar_absorptivitty()
        return eps * pathlength * concentration

    def transmittance(self, pathlength, concentration):
        '''\
        Calculates the transmittance from the absorbance on a scale
        from 0 to 1.  The user must convert to percent if she so desires.

        Parameters
        ----------
        pathlength : :obj:`float`
            The pathlength in centimeters of the light
        concentration : :obj:`float`
            The solution concentration in molarity

        Returns
        -------
        transmittance : :obj:`numpy.ndarray` of length :attr:`~.npol`
            The normalized transmittance for each frequency in

        Raises
        ------
        ValueError
            Invalid arguments given
        DIMError
            Not an FD calculation

        Examples
        --------

            >>> import dim
            >>> fd = dim.quick_test(fd=True)
            >>> fd.transmittance(1.5, 1.5e-6)
            array([ 0.64455141,  0.86579899])

        '''
        absorb = self.absorbance(pathlength, concentration)
        return power(10, -absorb)

    def pol_minmax(self):
        '''\
        Finds the minimum and maximum isotropic polarizabilities

        Returns
        -------
        pol_min : :obj:`float`
            Minimum isotropic polarizability
        pol_max : :obj:`float`
            Maximum isotropic polarizability

        Examples
        --------

            >>> import dim
            >>> fd = dim.quick_test(fd=True)
            >>> minmax = fd.pol_minmax()
            >>> fd.isotropic
            array([ 70.89400000+274.812j, -80.48033333 +85.293j])
            >>> round(minmax[0].real, 4)
            -80.4803
            >>> round(minmax[0].imag, 4)
            85.293
            >>> round(minmax[1].real, 4)
            70.894
            >>> round(minmax[1].imag, 4)
            274.812

        '''
        return (self.isotropic.min(),
                self.isotropic.max())

    def pol_diagonalize(self):
        '''Diagonalizes the polarizability tensors in place.

        Returns
        -------
        rotmat : :obj:`numpy.ndarray` of shape :attr:`~.npol` x 3 x 3
            Rotation matrix after diagonalization

        Raises
        ------
        DIMError
            Diagonalization does not converge

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.polarizability
            array([[[ 71.324,   0.   ,   0.   ],
                    [  0.   ,  71.324,   0.   ],
                    [  0.   ,   0.   ,  93.755]]])
            >>> static.pol_diagonalize()
            array([[[ 1.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0.,  1.]]])
            >>> static.polarizability
            array([[[ 71.324,   0.   ,   0.   ],
                    [  0.   ,  71.324,   0.   ],
                    [  0.   ,   0.   ,  93.755]]])
            >>> # Note: This example shows no difference  before and
            >>> # after diagonalization because the tensor was already
            >>> # diagonalized because the nanoparticle is linear on the
            >>> # z-axis.

        '''

        # Find the eigenvector and eigenvalues of the tensor
        # The eigenvector is the diagonal of the tensor, and the
        # eigenvalues are the rotation matrix.
        rotation = zeros_like(self.polarizability)

        for n in range(self.npol):
            try:
                eigenvec, eigenval = eig(self.polarizability[n])
            except LinAlgError:
                raise DIMError('Diagonalization of polarizability '
                               'tensor does not converge')
            else:
                self.polarizability[n] = diagflat(eigenvec)
                rotation[n] = eigenval

        # Remember the state
        self._diagonalized = True

        return rotation

    def absorb(self, other, ignore=set()):
        '''\
        Method to "absorb" all of the data from another
        :py:class:`DIM` instance to the current DIM instance.

        Data in `other` overwrite data in `self`.
        Empty parameters in `other` will be ignored.

        Parameters
        ----------
        other : :class:`DIM`
            The :class:`DIM` object to be absorbed.
        ignore : :obj:`set`
            Attributes that will **NOT** be absorbed.

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> fd = dim.quick_test(fd=True)

        '''
        # Not DIM instance
        if not isinstance(other, DIM):
            raise DIMError('absorb(): other != DIM()')
        # Make sure ignore is a set
        ignore = set(ignore)

        # Set of attributes to absorb
        attr = set(['calctype', 'subkey', 'key', 'natoms', 'coordinates',
                    'atoms', 'elements', 'npol', 'e_frequencies',
                    'polarizability', 'efficiencies', 'cross_section',
                    'dipoles', 'charges', 'start', 'real_time', 'cpu_time',
                    'routine_time', 'termination', 'host', 'nprocs'])

        # Run over all attributes on the attribute list (minus the ones
        # in ignore) and copy into this instance.
        # Don't copy things that are None.
        for k in attr.difference(ignore):
            if k not in other:
                continue
            setattr(self, k, deepcopy(getattr(other, k)))
        if 'coordinates' not in ignore:
            self._minmax = None
            self._bonds = None

    def empty(self, ignore=set()):
        '''\
        Empties all attributes in the current :class:`DIM` instance
        except those on the ignore list.

        Parameters
        ----------
        ignore : :obj:`set`
            A set of attributes to **NOT** empty

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> fd = dim.quick_test(fd=True)

        '''
        # Make sure ignore is a set
        ignore = set(ignore)

        # Set of attributes to empty
        attr = set(['calctype', 'subkey', 'key', 'natoms', 'coordinates',
                    'atoms', 'elements', 'npol', 'e_frequencies',
                    'polarizability', 'efficiencies', 'cross_section',
                    'dipoles', 'charges', 'start', 'real_time', 'cpu_time',
                    'routine_time', 'termination', 'host', 'nprocs'])

        # Run over all attributes and empty them, except for the ones
        # in ignore.
        for k in attr.difference(ignore):
            if k in ('calctype', 'subkey'):
                setattr(self, k, set())
            elif k == 'key':
                setattr(self, k, {})
            else:
                setattr(self, k, None)
        if 'coordinates' not in ignore:
            self._minmax = None
            self._bonds = None


    #################
    # MANAGED METHODS
    #################

    @property
    def masses(self):
        """The atomic mass of each atom in the nanoparticle

        .. note:: This is a property method, meaning that this is called
                  as though it were an attribute of :class:`DIM`, so
                  no parentheses are used.

        Returns
        -------
        masses : :obj:`numpy.ndarray` of length ``natoms``
            The atomic masses in the system

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.masses
            array([ 106.905093,  106.905093])

        """
        # Make the atomic mass function broadcastable on a numpy array
        atm_mass = vectorize(atomic_mass)
        # Return the mass for each atom
        return atm_mass(self.atoms)

    @property
    def molecular_mass(self):
        """The total mass of the nanoparticle

        .. note:: This is a property method, meaning that this is called
                  as though it were an attribute of :class:`DIM`, so
                  no parentheses are used.

        Returns
        -------
        molecular_mass : :obj:`float`
            The molecular mass of the nanoparticle

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> round(static.molecular_mass, 6)
            213.810186

        """
        # Sum masses and return the total molecular mass
        return self.masses.sum()

    @property
    def bonds(self):
        """Returns a list of the bonds in the system

        Given that ``N`` is the number of bonds, this is an ``N`` x 2 numpy
        array containing the index of atom 1 and atom 2 of each bond.
        Therefore, bond ``i`` goes from ``self.coordinates[self.bonds[i,0]]``
        to ``self.coordinates[self.bonds[i,1]]``

         .. note:: This is a property method, meaning that this is called
                  as though it were an attribute of :class:`DIM`, so
                  no parentheses are used.

        Returns
        -------
        bonds :  :obj:`numpy.ndarray` of shape ``N`` x 2
            The array of bonds

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.bonds
            array([], shape=(0, 2), dtype=int64)
            >>> # This is empty because the atoms in this
            >>> # example are too far apart to be considered bonding

        """
        # Use a C subroutine to find bonds.
        if self._bonds is None:
            try:
                self._bonds = asarray(calc_bonds(self.coordinates,
                                                 self.radii('vis'), 1.1).T, dtype=int)
            except AttributeError:
                return None
        return self._bonds

    @property
    def maxdist(self):
        """The maximum distance between two atoms in the nanoparticle

        .. note:: This is a property method, meaning that this is called
                  as though it were an attribute of :class:`DIM`, so
                  no parentheses are used.

        Returns
        -------
        maxdist : :obj:`float`
            The distance

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.natoms
            2
            >>> # Only 2 atoms, so maxdist is the distance between those two
            >>> static.maxdist
            4.0

        """
        # Return the maximum distance between atoms in the system.
        if self._minmax is None:
            self._minmax = minmax_pdist(self.coordinates)
        return self._minmax[1]

    @property
    def mindist(self):
        """The minimum distance between two atoms in the nanoparticle

        .. note:: This is a property method, meaning that this is called
                  as though it were an attribute of :class:`DIM`, so
                  no parentheses are used.

        Returns
        -------
        mindist : :obj:`float`
            The distance

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.natoms
            2
            >>> # Only 2 atoms, so mindist is the distance between those two
            >>> static.mindist
            4.0

        """
        # Return the mimimum distance between atoms in the system.
        if self._minmax is None:
            self._minmax = minmax_pdist(self.coordinates)
        return self._minmax[0]

    @property
    def isotropic(self):
        """The isotropic polarizability for each polarizability tensor.

        .. note:: This is a property method, meaning that this is called
                  as though it were an attribute of :class:`DIM`, so
                  no parentheses are used.

        Returns
        -------
        iso : :obj:`numpy.ndarray` of length :attr:`~.npol`
            Isotropic polarizabilities

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> static.npol
            1
            >>> static.isotropic
            array([ 78.801])
            >>> fd = dim.quick_test(fd=True)
            >>> fd.npol
            2
            >>> fd.isotropic
            array([ 70.89400000+274.812j, -80.48033333 +85.293j])

        """
        return self.tensor_isotropic(self.polarizability)

    @property
    def anisotropic2(self):
        """The anisotropic polarizability squared for each
        polarizability tensor.

        .. note:: This is a property method, meaning that this is called
                  as though it were an attribute of :class:`DIM`, so
                  no parentheses are used.

        Returns
        -------
        ani : :obj:`numpy.ndarray` of length :attr:`~.npol`
            Anisotropic polarizabilities squared

        Examples
        --------

            import dim
            >>> static = dim.quick_test()
            >>> static.npol
            1
            >>> static.anisotropic2
            array([ 503.149761])
            >>> fd = dim.quick_test(fd=True)
            >>> fd.npol
            2
            >>> fd.anisotropic2
            array([ 152906.539536,    1761.492265])

        """

        # Calculates the anisotropic polarizability squared
        return self.tensor_anisotropic2(self.polarizability)

    ####################
    # OPERATOR OVERRIDES
    ####################

    def __str__(self):
        '''Defines what is printed when the class is printed.'''
        return 'DIM object for file {0}'.format(self.filename)

    def __contains__(self, key):
        '''\
        Boolean for if a field is filled or not.

        Parameters
        ----------
        key : :obj:`str`
            The :class:`DIM` attribute to check

        Returns
        -------
        contains : :obj:`bool`
            :obj:`True` if the field is not empty, :obj:`False` otherwise

        Examples
        --------

            >>> import dim
            >>> static = dim.quick_test()
            >>> 'efficiencies' in static
            False
            >>> 'coordinates' in static
            True

        '''
        if getattr(self, key, None) is not None:
            try:
                return bool(getattr(self, key, None))
            except ValueError:
                return True
        else:
            return False

    def __add__(self, other):
        '''\
        Concatenates two nanoparticles together and returns the new molecule.

        The attributes that are concatenated are:

          - :attr:`coordinates`
          - :attr:`atoms`
          - :attr:`natoms`
          - :attr:`elements`
          - :attr:`nelements`

        All other properties would be unphysical to concatenate
        and are emptied.

        Parameters
        ----------
        other : :class:`DIM`
            The other nanoparticle

        Returns
        -------
        new : :class:`DIM`
            The new :class:`DIM` instance with the data
            concatenated together

        Examples
        --------

            >>> import dim
            >>> np1 = dim.quick_test()
            >>> np1.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> np2 = dim.quick_test()
            >>> np2.translate_coordinates([3.0, 0.0, 0.0])
            >>> np2.coordinates
            array([[ 3.,  0.,  0.],
                   [ 3.,  0.,  4.]])
            >>> np3 = np1 + np2
            >>> np3.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.],
                   [ 3.,  0.,  0.],
                   [ 3.,  0.,  4.]])
            >>> np3.atoms
            array(['Ag', 'Ag', 'Ag', 'Ag'], 
                  dtype='|S2')

        '''
        from numpy import concatenate
        from dimpy import DIM
        new = DIM(self.filename)
        new.coordinates = concatenate((self.coordinates, other.coordinates))
        new.atoms = concatenate((self.atoms, other.atoms))
        new.elements = self.elements.union(other.elements)
        new.natoms = len(new.atoms)
        new.nelements = len(new.elements)

        return new

    def __iadd__(self, other):
        '''\
        Alternative syntax to :meth:`~.join`

        Parameters
        ----------
        other : :class:`DIM`
            The other nanoparticle

        Returns
        -------
        new : :class:`DIM`
            The new :class:`DIM` instance with the data
            concatenated together

        Examples
        --------

            >>> import dim
            >>> np1 = dim.quick_test()
            >>> np1.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.]])
            >>> np2 = dim.quick_test()
            >>> np2.translate_coordinates([3.0, 0.0, 0.0])
            >>> np2.coordinates
            array([[ 3.,  0.,  0.],
                   [ 3.,  0.,  4.]])
            >>> np1 += np2
            >>> np1.coordinates
            array([[ 0.,  0.,  0.],
                   [ 0.,  0.,  4.],
                   [ 3.,  0.,  0.],
                   [ 3.,  0.,  4.]])
            >>> np1.atoms
            array(['Ag', 'Ag', 'Ag', 'Ag'], 
                  dtype='|S2')
        '''
        self.join(other)

    #################
    # PRIVATE METHODS
    #################

    def __det_calc_type(self):
        '''Determine the calculation type based on what is found in the keys'''
        if 'PRINT' in self.key:
            if 'ATMDIP' in self.key['PRINT']:
                self.subkey.add('ATMDIP')
            if 'ENERGY' in self.key['PRINT']:
                self.subkey.add('ENERGY')
        if 'NOPRINT' in self.key:
            if 'ATMDIP' in self.key['NOPRINT']:
                self.subkey.discard('ATMDIP')
            if 'ENERGY' in self.key['NOPRINT']:
                self.subkey.discard('ENERGY')
        if 'CPIM' in self.key:
            self.subkey.add('CPIM')
        elif 'PIM' in self.key:
            self.subkey.add('PIM')
        if 'FREQRANGE' in self.key or 'FREQUENCY' in self.key:
            self.calctype = 'FD'
        else:
            self.calctype = 'STATIC'

    def _raise_or_pass(self, msg):
        '''Subroutine to pass or raise on collection error.
        Makes termination more specific by adding error message.

        '''
        if self.termination is None or 'ERROR:' not in self.termination:
            self.termination = msg
        if self._abort:
            raise DIMError(msg)
        else:
            pass

    @staticmethod
    def tensor_isotropic(tn):
        return trace(tn, axis1=1, axis2=2) / 3

    @staticmethod
    def tensor_anisotropic2(tn):
        # Calculate anisotripoc polarizability squared using eq.14 from
        # J. Chem. Phys, 75, 5615:
        # \frac{3}{4}\left[\sum_{ij}\alpha_{ij}\alpha_{ij)^\ast
        #                + \sum_{ij}\alpha_{ij}\alpha_{ji)^\ast\right]
        # - \frac{1}{2}\sum_{ij}\alpha_{ii}\alpha_{jj}^\ast

        # It should be noted that there is no complex anisotropic
        # polarizability like there is isotropic polarizability.  This is due
        # to the fact that the anisotropic polarizaibiltiy is instrinsically
        # a magnatude due to the squared factor, and as such must be real.
        cj = tn.conjugate()
        r = range(3)
        t = empty(len(tn), dtype=float)
        for n in range(len(tn)):
            cross1 = sum([tn[n, i, j] * cj[n, i, j] for i in r for j in r])
            cross2 = sum([tn[n, i, j] * cj[n, j, i] for i in r for j in r])
            diag = sum([tn[n, i, i] * cj[n, j, j] for i in r for j in r])
            t[n] = absolute((3 / 4) * (cross1 + cross2) - (1 / 2) * diag)
        return t
