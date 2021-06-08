
import math
import os
import re

import input_reader
import numpy as np
import scipy as sp
from scipy import interpolate, spatial

from .constants import ELEMENTS, elemset as ELEMSET, atomic_mass, atomic_radius, \
                       ANGSTROM2BOHR, HART2NM, BOHR2NM
from .dimpy_error import DIMPyError
from .memory import Memory, check_memory
from .printer import Output, print_header
from .timer import Timer, check_time


class Nanoparticle(object):
    """\
    Stores the information necessary to describe the nanoparticle.

    The information is also stored in a way to match the properties used
    by PySCF's Mole object.

    Required Arguments:
        atoms_list_or_string : list or string
            Gives a way to find the nanoparticle coordinates. can be
            either a string or list of atoms like in PySCF.Mole or can be
            the filename of a DIMPy input file or an XYZ file.

    Optional agruments:
        unit : string (default is 'Angstrom')

        atom_params : dict
            List of parameters for each atom type. Format is:
            | atom_params = {'Ag': {'rad': 1.445, 'exp': 'Ag_jc'},
            |                ...
            |                'Au': {'rad': 1.222, 'exp': 'Au_jc'}}

        rcut : float
            Distance between atoms to ignore interactions (default is 100 bohr).

    Attributes:
        natom : int
            Number of atoms
        atoms : array
            Array of atomic symbols (compatibility with DIM / Chem Package)
        coordinates : array
            Array of atomic coordinates (compatibility with DIM / Chem Package)
        atomic_radii : array
            Array of the radius of each atom
        atomic_static_polarizabilities : array
            Array of the static polarizability of each atom

    Attributes to match PySCF.Mole object:
        atom : list
            To define the molecular structure. Format is:
            | atom = [[atom1, (x, y, z)],
            |         [atom2, (x, y, z)],
            |         ...
            |         [atomN, (x, y, z)]]

        unit : str
            'Angstrom' by default
        natm : int
            Number of atoms

    Example with explicitly passing in the atoms as a string:
    >>> nanoparticle = Nanoparticle('Ag 0 0 0; Ag 0 0 2.5')        

    Example with explicity passing in the atoms as a list:
    >>> nanoparticle = Nanoparticle([['Ag', (0, 0, 0)], ['Ag'

    Example using a DIMPy input file:
    >>> nanoparticle = Nanoparticle('dimpy_input.dim')

    Example using an XYZ file:
    >>> nanoparticle = Nanoparticle('coordinates_file.xyz')
    """

    def __init__(self, atoms_list_or_string, unit='Angstrom',
                 atom_params=None, rcut=100., output_filename=None,
                 log_filename=None, pbc=None, debug=False):
        '''Initializes the Nanoparticle class.
        '''

        # create the output and log files
        self.out = Output(filename=output_filename)
        self.log = Output(filename=log_filename, logfile=True)

        # initialize memory management
        self._memory = Memory()

        # start reading from input file
        self._timer = Timer()
        start_time = self._timer.startTimer('Total Process', short='DIMPy')
        self.log('Initializing nanoparticle',
            time=self._timer.startTimer('Nanoparticle.__init__'))

        # set passed in attributes here
        self.unit = unit
        self.rcut = rcut
        self.output_filename = output_filename
        self.log_filename = log_filename
        self.xyz_filename = None
        self.atom_params = atom_params
        self.debug = debug

        # check if input is a string and that it's a filename
        if (isinstance(atoms_list_or_string, str) and
           (os.path.splitext(atoms_list_or_string)[1].lower()
           in ('.dim', '.inp', '.xyz'))):
            self.input_filename = atoms_list_or_string
            self._read_coordinates(atoms_list_or_string)

        # else this is the atoms given in a pyscf.Mole format
        else:
            self.input_filename = None
            self._format_atoms(atoms_list_or_string)

        # get atomic radii
        # check if argument is passed first:
        if atom_params is not None:
            self._get_atomic_radii_from_atom_params(atom_params)
        else:
            self._get_atomic_radii_from_dimpy_input()

        # periodic boundary conditions
        self.pbc = pbc

        # Other attributes. These should never have to be set by the user
        self._molecular_mass = None
        self._static_polarizabilities = None
        self._r_vec = None
        self._distances = None
        self._volume = None
        self._radmult = None

        # end the timer
        end_time = self._timer.endTimer('Nanoparticle.__init__')
        self.log('Finished Initializing nanoparticle, '
                 '{0:.3f} seconds'.format(end_time[1]),
                 time=end_time[0])


    @check_memory(log='debug')
    @check_time(log='debug')
    def _read_coordinates(self, filename=None):
        '''\
        Determines whether this is a DIMPy input or an XYZ file
        and calls the appropriate function to read them.
        '''

        if filename is None: filename = self.input_filename

        # first check that file exists
        if not os.path.isfile(filename):
            raise DIMPyError(f'File `{filename}` does not exist!')

        # is this a DIMPy input?
        if os.path.splitext(filename)[1].lower() in ('.dim', '.inp'):
            self._read_coordinates_from_dimpy_input(filename)

        # is this an XYZ file?
        if os.path.splitext(filename)[1].lower() == '.xyz':
            self._read_coordinates_from_xyz_file(filename)


    @check_memory(log='debug')
    @check_time(log='debug')
    def _read_coordinates_from_dimpy_input(self, filename=None):
        '''\
        Reads the coordinates from a DIMPy input file.
        '''

        if filename is None: filename = self.input_filename

        # Initializes reader for a DIMPy input
        # Valid comment characters are `::`, `#`, `!`, and `//`
        reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
                 case=False, ignoreunknown=True)

        # Read from the coordinates block of the DIMPy input
        xyz = reader.add_block_key('xyz', required=True)
        xyz.add_regex_line('natoms', r'^\s*(\d+)$', repeat=False)
        meg = xyz.add_mutually_exclusive_group(required=True)
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
        meg.add_regex_line('coords', coord, repeat=True, depends='natoms')
        meg.add_regex_line('file', r'(.*\.xyz)$', repeat=False)

        # read the file and collect the options
        options = reader.read_input(filename).xyz


        # If we gave the coordinates in the DIMPy input, set them here
        if options.coords is not None:

            if options.natoms is not None:
                self.natoms = int(options.natoms.group(0))
            else:
                self.natoms = len(options.coords)

            if self.natoms != len(options.coords):
                raise DIMPyError('NATOMS does not match the number of atoms '
                                f'given in the DIMPy input file `{filename}`!')

            # set the atoms
            self.atoms = np.array([atom.group(1) for atom in options.coords])

            # set the coordinates
            self.coordinates = np.array([[atom.group(i) for i in range(2,5)]
                                      for atom in options.coords], dtype=np.float32)

        # if the DIMPy input file has a link to an XYZ file, then
        # we need to read the XYZ file
        if options.file is not None:
            xyz_filename = options.file.group(0)
            self._read_coordinates_from_xyz_file(xyz_filename)

        # in either of these two cases, the coordinates are in Angstroms,
        # but we want to store them internally in Bohr, so we convert here
        self.coordinates *= 1.8897261328856432
        self.unit = 'bohr'


    @check_memory(log='debug')
    @check_time(log='debug')
    def _read_coordinates_from_xyz_file(self, filename=None):
        '''Reads the atom coordinates from an XYZ file.
        '''

        if filename is None: filename = self.input_filename
        self.xyz_filename = filename

        # Initializes reader for a DIMPy input
        # Valid comment characters are `::`, `#`, `!`, and `//`
        reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
                 case=False, ignoreunknown=True)

        # Read from the coordinates block of the DIMPy input
        reader.add_regex_line('natoms', r'^\s*(\d+)$', repeat=False, required=True)
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
        reader.add_regex_line('coords', coord, repeat=True, required=True)

        # read the file and collect the options
        options = reader.read_input(filename)

        self.natoms = int(options.natoms.group(0))

        if self.natoms != len(options.coords):
            raise DIMPyError('Number of atoms does not match the number of atom '
                            f'coordinates given in the XYZ file `{filename}`!')

        # set the atoms
        self.atoms = np.array([atom.group(1) for atom in options.coords])

        # set the coordinates
        self.coordinates = np.array([[atom.group(i) for i in range(2,5)]
                                  for atom in options.coords], dtype=np.float32)


    @check_memory(log='debug')
    @check_time(log='debug')
    def _format_atoms(self, atoms_list_or_string):
        '''Convert the input :attr:`Nanoparticle.atoms_list_or_string` to the internal
        data format. The format is similar to pyscf.Mole atoms input.
        '''

        def get_atom_symbol(number_or_symbol):
            '''Returns the atomic symbol if given the atomic number.'''
            if (isinstance(number_or_symbol, str) and
                number_or_symbol in ELEMSET):
                return number_or_symbol
            elif isinstance(number_or_symbol, (int, float)):
                return ELEMENTS[int(number_or_symbol)]
            else:
                raise DIMPyError('Cannot recognize atom in input!')

        def str2atm(line):
            data = line.split()
            if len(data) != 4: raise DIMPyError('Error in atoms string input!')
            symbol = get_atom_symbol(data[0])
            coord  = [float(x) for x in data[1:4]]
            return symbol, coord

        # input is a string
        if isinstance(atoms_list_or_string, str):
            atoms = str(atoms_list_or_string.replace(';','\n').replace(
                        ',',' ').replace('\t',' '))
            atoms = atoms.split('\n')
            self.atoms = []
            self.coordinates = []
            for atom in atoms:
                atom_symbol, atom_coord = str2atm(atom.strip())
                self.atoms.append(atom_symbol)
                self.coordinates.append(atom_coord)

        # otherwise this should be a list / array
        else:
            self.atoms = []
            self.coordinates = []
            for atom in atoms_list_or_string:
                # atom about still be a string
                if isinstance(atom, str):
                    symbol, coord = str2atm(atom.replace(',', ' '))

                # atom is a list / array
                else:
                    symbol = get_atom_symbol(atom[0])

                    # two options for atom coordinates
                    if isinstance(atom[1], (int, float)):
                        coord = atom[1:4]
                    else:
                        coord  = atom[1]

                self.atoms.append(symbol)
                self.coordinates.append(coord)
        self.atoms = np.array(self.atoms)
        self.coordinates = np.array(self.coordinates, dtype=np.float32)
        self.natoms = len(self.atoms)


    @check_memory(log='debug')
    @check_time(log='debug')
    def _get_atomic_radii_from_atom_params(self, atom_params):
        self.atomic_radii = np.zeros((self.natoms))
        unique_atoms = np.unique(self.atoms)
        for atom in unique_atoms:
            atom_index = np.where(self.atoms == atom)
            try:
                self.atomic_radii[atom_index] = atom_params[atom]['rad']
            except (KeyError, TypeError):
                raise DIMPyError(f'`rad` key for atom `{atom}` not given '
                                 'in atom_params!')


    @check_memory(log='debug')
    @check_time(log='debug')
    def _get_atomic_radii_from_dimpy_input(self, filename=None):

        if filename is None: filename = self.input_filename

        # Initializes reader for a DIMPy input
        # Valid comment characters are `::`, `#`, `!`, and `//`
        reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
                 case=False, ignoreunknown=True)

        for el in ELEMENTS:  # Elements defined from constants module
            e = reader.add_block_key(el)

            # required keys
            e.add_line_key('rad',   type=float, required=True)

            # optional keys with defaults

            # optional keys without defaults
            e.add_line_key('exp',   type=str, case=True, default=None)

        # read and collect the options
        options = reader.read_input(filename)

        # set atom_params
        atom_params = {}
        for atom in options.keys():
            if getattr(options, atom) is not None:
                atom_para = vars(getattr(options, atom))
                atom_params[atom.capitalize()] = atom_para
        self.atom_params = atom_params

        # set atomic radii
        self.atomic_radii = np.zeros((self.natoms))
        unique_atoms = np.unique(self.atoms)
        for atom in unique_atoms:
            atom_index = np.where(self.atoms == atom)
            self.atomic_radii[atom_index] = getattr(options, atom.lower()).rad

    #################
    # MANAGED METHODS
    #################

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def atom(self):
        '''Atom list format from pyscf.Mole'''
        return [[self.atoms[i], np.array([self.coordinates[i][0],
                 self.coordinates[i][1], self.coordinates[i][2]], dtype=np.float32)]
                for i in range(self.natoms)]

    @property
    def natm(self):
        '''Number of atoms from pyscf.Mole'''
        return self.natoms

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
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
        atm_mass = np.vectorize(atomic_mass)
        # Return the mass for each atom
        return atm_mass(self.atoms) 

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
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
        if self._molecular_mass is None:
            self._molecular_mass = self.masses.sum()
        return self._molecular_mass


    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def static_polarizabilities(self):
        '''The polarizabilities of each atom caculated from their radii.'''
        if self._static_polarizabilities is None:
            # FIXME: I'm using this factor for now, the exact formula should
            # be used.
            factor = 12.88837
            self._static_polarizabilities = factor * self.atomic_radii**3
        return self._static_polarizabilities


    @check_memory(log='debug')
    @check_time(log='debug')
    def atomic_polarizabilities(self, omega=None):
        '''The atomic polarizabilities.'''

        # if this is a static calculation, then nothing needs to be done
        if omega is None or omega == 0:
            return self.static_polarizabilities

        # return the complex frequency-dependent polarizabilities
        # FIXME: right now, only experimental dielectrics are allowed
        polarizabilities = np.zeros((self.natoms), dtype=complex)
        unique_atoms = np.unique(self.atoms)
        for atom in unique_atoms:
            atom_index = np.where(self.atoms == atom)
            die_file = self.atom_params[atom]['exp']
            epsilon = self._read_experimental_dielectric(die_file, omega)
            atomic_pol = (epsilon - 1) / (epsilon + 2)
            polarizabilities[atom_index] = atomic_pol
        return polarizabilities * self.static_polarizabilities


    @check_memory(log='debug')
    @check_time(log='debug')
    def _read_experimental_dielectric(self, die_file, omega):

        # get the experimental dielectric file and open it
        abs_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'dielectrics', die_file)
        exp_data = np.loadtxt(abs_file, unpack=True)

        # interpolate the data
        interpolate_real = sp.interpolate.interp1d(exp_data[0],
                            exp_data[1], kind='cubic')
        interpolate_imag = sp.interpolate.interp1d(exp_data[0],
                            exp_data[2], kind='cubic')

        # get new experimental dielectrics
        die_real = interpolate_real(HART2NM(omega))
        die_imag = interpolate_imag(HART2NM(omega))
        return die_real + 1j * die_imag

    ###########################################################
    # Matrix and Tensor Functions
    # FIXME: These are not written to be efficient right now
    # consider using scipy.sparse to make them memory efficient
    ###########################################################

    @property
    @check_memory
    @check_time(log='once')
    def r_vec(self):
        '''The vector distance between two points.'''
        if self._r_vec is None:

            self._r_vec = (self.coordinates[np.newaxis,:,:] -
                     self.coordinates[:,np.newaxis,:])
            if self.unit in ('A', 'Ang', 'Angstrom'):
                self._r_vec *= 1.8897261328856432

        return self._r_vec

    @property
    @check_memory
    @check_time(log='once')
    def distances(self):
        '''Get the distances between atoms in atomic units.'''
        if self._distances is None:

            dists = spatial.distance.pdist(self.coordinates)
            self._distances = spatial.distance.squareform(dists)
            self._distances = np.array(self._distances, dtype=np.float32)
            if self.unit in ('A', 'Ang', 'Angstrom'):
                self._distances *= 1.8897261328856432

        return self._distances

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def volume(self):
        '''Get the volume.'''

        if self._volume is None:

            self._radmult = - 1.0 / (self.natoms**0.5) + 1.0
            radmult_cubed = self._radmult**3

            rad_cubed = self.atomic_radii * self.atomic_radii * self.atomic_radii
            rad_cubed = rad_cubed.sum()

            self._volume = 4.0 * math.pi * radmult_cubed * rad_cubed / 3.0
            self._volume *= BOHR2NM(1)**3 * ANGSTROM2BOHR(1)**3


        return self._volume
        

    #################################
    # End Matrix and Tensor functions
    #################################

    @check_memory(log='debug')
    @check_time(log='debug')
    def _print_nanoparticle(self, output=None):
        '''Prints all relevant information about the nanoparticle.'''

        if output is None: output = self.out
        unique_atoms = np.unique(self.atoms)
        n_unique_atoms = len(unique_atoms)

        # print overview information about nanoparticle
        print_header('Nanoparticle Input', output)
        output(f'Input file   : {self.input_filename}')
        output(f'# Atoms      : {self.natoms:>7d}')
        output(f'# Atom Types : {n_unique_atoms:>7d}')
        output(f'XYZ file     : {self.xyz_filename}')
        output()

        # atomic properties
        string = 'Atom Type '
        printable_params = ['rad', 'exp', 'others']
        params_to_print = []
        for atom in self.atom_params.keys():
            for key in self.atom_params[atom]:
                if key in printable_params and not key in params_to_print:
                    params_to_print.append(key)
                    string = string + key.center(10)
        output(string)
        output('-'*len(string))
        string = ''
        for atom in unique_atoms:
            string += atom.ljust(10)
            for key in params_to_print:
                if key in self.atom_params[atom].keys():
                    string += ('{0}'.format(self.atom_params[atom][key])).center(10)
                else:
                    string += '---'.center(10)
        output(string)
        output()

        # print the volume
        volume = self.volume
        print_header('Volume', output)
        output((f'Multiplier   : {self._radmult:10.4f}     ').center(79))
        output((f'Volume       : {volume:10.4e} nm^3').center(79))
        output()

        # print coordiantes only if an xyz file doesn't exist
        if self.xyz_filename is None:
            print_header('G E O M E T R Y', output)
            output('       Atom                     Coordinates          ')
            output('                      X              Y              Z')
            output('_______________________________________________________________')
            for i in range(self.natoms):
                iatom = i+1
                atom = self.atoms[i]
                x, y, z = self.coordinates[i,:]
                output(f'{i+1:>7}   {atom:<2s} {x:14.8f} {y:14.8f} {z:14.8f}')
            output()
