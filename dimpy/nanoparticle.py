
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
#from .read_input_file import read_input_file
from .read_input_file import ReadInput
from .timer import Timer, check_time


class Nanoparticle(object):
    """Store the information necessary to describe the nanoparticle.

    The information is also stored in a way to match the properties used
    by PySCF's Mole object.

    :param input_list_or_string: Give the nanoparticle coordinates.
        Can be either a string or list of atoms like in ``PySCF.Mole``
        or can be the filename of an XYZ file.
    :type input_list_or_string: list or str

    :param unit: Unit to use for conversion, default is "Angstrom"
    :type unit: str, optional

    :param atom_params: Dict of parameters for each atom type. Format is:
        ``atom_params = {'Ag': {'rad': 1.445, 'exp': 'Ag_jc'},
        ...,
        'Au': {'rad': 1.222, 'exp': 'Au_jc'}}``
    :type atom_params: dict, optional

    :param output_filename: The filename to print output data to, default
        is standard output
    :type output_filename: str of file object, optional

    :param log_filename: The filename to print log data to, default
        is standard output
    :type log_filename: str of file object, optional

    :param pbc: Array of lattice vectors. Could be None, a 1x3 or 2x3 array.
        Default is None
    :type pbc: numpy.ndarray or None, optional

    :param debug: Whether to print extra debug statements, default is False
    :type debug: bool, optional

    :cvar out: Output file object. Prints to output using ``out("string")``
    :vartype out: file object

    :cvar log: Log file object. Prints to log using ``log("string")``
    :vartype log: file object

    :cvar natom: Number of atoms
    :vartype natom: int

    :cvar natm: Number of atoms
    :vartype natom: int

    :cvar atoms: Atomic symbols (compatibility with DIM / Chem Package)
    :vartype atoms: numpy.ndarray

    :cvar atom: The molecular structure (compatibility with PySCF). The Format
        is ``atom = [[atom1, (x1, y1, z1)], [atom2, (x2, y2, z1)], ...]``
    :vartype atom: list

    :cvar coordinates: Atomic coordinates (compatibility with DIM / Chem Package)
    :vartype coordinates: numpy.ndarray

    :cvar atomic_radii: Radius of each atom
    :vartype atomic_radii: numpy.ndarray

    :cvar static_polarizabilities: Static polarizability of each atom
    :vartype static_polarizabilities: numpy.ndarray

    Example with explicitly passing in the atoms as a string::

        >>> from dimpy import Nanoparticle
        >>> nano = Nanoparticle('Ag 0 0 0; Ag 0 0 2.5')        
        >>> nano.build()

    Example with explicity passing in the atoms as a list::

        >>> nano = Nanoparticle([['Ag', (0, 0, 0)], ['Ag',
        ...                      (0, 0, 1.89)]])
        >>> nano.build()

    Example using an XYZ file::

        >>> nano = Nanoparticle('coordinates_file.xyz')
        >>> nano.build()

    """

    def __init__(self, input_list_or_string, unit='angstrom',
                 atom_params={}, output_filename=None,
                 log_filename=None, pbc=None, debug=False,
                 verbose=2):
        """Initialize the Nanoparticle class."""

        # create the output and log files
        self.out = Output(filename=output_filename)
        self.log = Output(filename=log_filename, logfile=True)
        self.verbose = verbose
        self.debug = debug

        # initialize memory management
        self._memory = Memory()

        # start reading from input file
        self._timer = Timer()
        start_time = self._timer.startTimer('Total Process', short='DIMPy')
        start_time = self._timer.startTimer('Nanoparticle.__init__')
        if self.verbose > 0 or self.debug:
            self.log('Initializing nanoparticle', time=start_time)

        # set passed in / default attributes here
        self.unit = unit
        self.atom_params = atom_params
        self.output_filename = output_filename
        self.log_filename = log_filename
        if pbc is not None:
            self.pbc = np.array(pbc, dtype=np.float32) * self._distconv
        else:
            self.pbc = None
        self.xyz_filename = None
        self.unformatted_atoms = None

        # check if input is a string and that it's a filename
        if (isinstance(input_list_or_string, str) and
            (os.path.splitext(input_list_or_string)[1].lower()
            in ('.xyz',))):
            self.xyz_filename = input_list_or_string

        # else this is the atoms given in a pyscf.Mole format
        else:
            self.unformatted_atoms = input_list_or_string

        # Other attributes. These should never have to be set by the user
        self._molecular_mass = None
        self._static_polarizabilities = None
        self._r_vec = None
        self._distances = None
        self._volume = None
        self._radmult = None

        # end the timer
        end_time = self._timer.endTimer('Nanoparticle.__init__')
        if self.verbose > 0 or self.debug:
            self.log('Finished Initializing nanoparticle, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])

    @check_memory
    @check_time
    def build(self):
        """Build the nanoparticle object."""
        if self.xyz_filename is not None:
            self._read_coordinates_from_xyz_file()
        elif self.unformatted_atoms is not None:
            self._format_atoms(self.unformatted_atoms)

        self._get_atomic_radii_from_atom_params()

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def _distconv(self):
        if self.unit.lower() in ('b', 'bohr', 'au'):
            return 1.0
        elif self.unit.lower() in ('a', 'ang', 'angstrom'):
            return ANGSTROM2BOHR(1)


    @check_memory(log='debug')
    @check_time(log='debug')
    def _read_coordinates_from_xyz_file(self, filename=None):
        """Read the atom coordinates from an XYZ file."""
        if filename is None:
            filename = self.xyz_filename

        # Initializes reader for a DIMPy input
        # Valid comment characters are `::`, `#`, `!`, and `//`
        reader = input_reader.InputReader(case=False, ignoreunknown=True)

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
        self.coordinates *= ANGSTROM2BOHR(1) # .xyz file always in Angstrom


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
        self.coordinates *= self._distconv
        self.natoms = len(self.atoms)


    @check_memory(log='debug')
    @check_time(log='debug')
    def _get_atomic_radii_from_atom_params(self, atom_params=None):
        if atom_params is None:
            atoms_params = self.atom_params
        self.atomic_radii = np.zeros((self.natoms))
        unique_atoms = np.unique(self.atoms)
        for atom in unique_atoms:
            atom_index = np.where(self.atoms == atom)
            try:
                self.atomic_radii[atom_index] = atom_params[atom]['rad']
            except (KeyError, TypeError):
                radius = atomic_radius(atom, set='vdw')
                self.atomic_radii[atom_index] = radius
                try:
                    self.atom_params[atom]['rad'] = radius
                except KeyError:
                    self.atom_params[atom] = {'rad': radius}


    #################
    # MANAGED METHODS
    #################

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def atom(self):
        """Atom list format from pyscf.Mole."""
        return [[self.atoms[i], np.array([self.coordinates[i][0],
                 self.coordinates[i][1], self.coordinates[i][2]], dtype=np.float32)]
                for i in range(self.natoms)]

    @property
    def natm(self):
        """Number of atoms from pyscf.Mole."""
        return self.natoms

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def masses(self):
        """
        :returns: Atomic masses of each atom
        :rtype: :py:obj:`numpy.ndarray` of length :py:attr:`Nanoparticle.natm`
        """
        # Make the atomic mass function broadcastable on a numpy array
        atm_mass = np.vectorize(atomic_mass)
        # Return the mass for each atom
        return atm_mass(self.atoms) 

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def molecular_mass(self):
        """
        :returns: Total molar mass of the nanoparticle
        :rtype: float
        """
        if self._molecular_mass is None:
            self._molecular_mass = self.masses.sum()
        return self._molecular_mass


    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def static_polarizabilities(self):
        """The polarizabilities of each atom caculated from their radii."""
        if self._static_polarizabilities is None:
            # FIXME: I'm using this factor for now, the exact formula should
            # be used.
            factor = 12.88837
            self._static_polarizabilities = factor * self.atomic_radii**3
        return self._static_polarizabilities


    @check_memory(log='debug')
    @check_time(log='debug')
    def atomic_polarizabilities(self, omega=None):
        """The atomic polarizabilities at wavelength ``omega``.

        :param omega: Excitation frequency in atomic units, default None.
            If None given, uses :py:meth:`Nanoparticle.static_polarizabilities`
        :type omega: float
        """

        # if this is a static calculation, then nothing needs to be done
        if omega is None or omega == 0:
            return self.static_polarizabilities

        # return the complex frequency-dependent polarizabilities
        # FIXME: right now, only experimental dielectrics are allowed
        polarizabilities = np.zeros((self.natoms), dtype=complex)
        unique_atoms = np.unique(self.atoms)
        for atom in unique_atoms:
            atom_index = np.where(self.atoms == atom)
            try:
                die_file = self.atom_params[atom]['exp']
            except KeyError as e:
                raise DIMPyError("'EXP' not specified for atom type "+str(e))
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
        """The vector distances between pairs of atoms in atomic units."""
        if self._r_vec is None:
            self._r_vec = (self.coordinates[np.newaxis,:,:] -
                     self.coordinates[:,np.newaxis,:])
        return self._r_vec

    @property
    @check_memory
    @check_time(log='once')
    def distances(self):
        """The distances between pairs of atoms in atomic units."""
        if self._distances is None:
            dists = spatial.distance.pdist(self.coordinates)
            self._distances = spatial.distance.squareform(dists)
            self._distances = np.array(self._distances, dtype=np.float32)
        return self._distances

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def volume(self):
        """Volume of the nanoparticle in atomic units."""

        if self._volume is None:

            self._radmult = - 1.0 / (self.natoms**0.5) + 1.0
            radmult_cubed = self._radmult**3

            rad_cubed = self.atomic_radii * self.atomic_radii * self.atomic_radii
            rad_cubed = rad_cubed.sum()

            self._volume = 4.0 * math.pi * radmult_cubed * rad_cubed / 3.0
            self._volume *= BOHR2NM(1)**3 * ANGSTROM2BOHR(1)**3

            # BUGFIX: this returns zero if there is only 1 atom
            # so let's set the volume to the volume of that 1 atom
            if self._volume == 0:
                self._volume = 4.0 * math.pi * self.atomic_radii.sum()**3 / 3.0

        return self._volume
        

    #################################
    # End Matrix and Tensor functions
    #################################

    @check_memory(log='debug')
    @check_time(log='debug')
    def print_nanoparticle(self, output=None):
        """Print all relevant information about the nanoparticle.

        :param output: Output file object to print information to, defaults
            to ``std.out``
        :type output: file object

        """

        if output is None: output = self.out
        unique_atoms = np.unique(self.atoms)
        n_unique_atoms = len(unique_atoms)

        # print overview information about nanoparticle
        print_header('Nanoparticle Information', output)
        output(f'# Atoms      : {self.natoms:<7d}')
        n_type = 'Molecular'
        if self.pbc is not None:
            if len(self.pbc) == 1:
                n_type = '1-D Periodic'
            else:
                n_type = '2-D Periodic'
        output(f'Type         : {n_type}')
        output(f'# Atom Types : {n_unique_atoms:<7d}')
        output(f'XYZ file     : {self.xyz_filename}')
        output()

        # periodic boundary conditions lattice vectors
        if self.pbc is not None:
            string = 'Periodic Lattice Vector(s)'
            output(string)
            output('-'*len(string))
            for i in range(len(self.pbc)):
                string = (f'{self.pbc[i,0]:8.4f} {self.pbc[i,1]:8.4f} '
                          f'{self.pbc[i,2]:8.4f}')
                output(string)
        output()

        # atomic parameters
        string = 'Atom Parameter Value(s)   '
        output(string)
        output('-'*len(string))

        for atom in self.atom_params:
            for key in self.atom_params[atom]:
                value = self.atom_params[atom][key]
                string = f'{atom:<2s}    {key:<9s} '
                if isinstance(value, (int, float)):
                    string = string + f'{value}'
                else:
                    for i in range(len(value)):
                        string = string + f'{value[i]}'
                output(string)
        output()

        # print the volume
        volume = self.volume
        print_header('Volume', output)
        output((f'Multiplier   : {self._radmult:10.4f}     ').center(79))
        output((f'Volume       : {volume:10.4e} nm^3').center(79))
        output()

        # print coordinates only if an xyz file doesn't exist
        if (self.xyz_filename is None or self.verbose > 2 or self.debug):
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
