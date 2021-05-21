
from os import path
from input_reader import InputReader
from .dimpy import DIMPyError
import numpy as np
from numpy import linalg
import scipy as sp
from scipy.sparse import linalg
from .printer import print_atomic_dipoles
from time import time

class CalcMethod(object):
    '''Calculation base class.

    Uses a discrete dipole approximation (DDA) method
    '''

    def __init__(self, nanoparticle, input_filename=None, output_filename=None,
                 log_filename=None, freqs=None, title=None):
        '''\
        Initializes the DIMPy class.

        :param nanoparticle: the dimpy.Nanoparticle object (Required).

        :param input_filename: The name of the DIMPy input file (Optional).
        :param output_filename: The name of the DIMPy output file (Optional).
        :param log_filename: The name of the DIMPy log file (Optional).
        '''

        from .timer import Timer
        from .printer import Output

        # use the nanoparticle attributes
        self.nanoparticle = nanoparticle
        self.out = nanoparticle.out
        self.log = nanoparticle.log
        self._timer = nanoparticle._timer

        self.log('Initializing calculation', time=self._timer.startTimer('Init Calc'))

        # set default attributes and parameters
        if input_filename is None:
            input_filename = nanoparticle.input_filename
        self.input_filename = input_filename
        if output_filename is None:
            output_filename = nanoparticle.output_filename
        self.output_filename = output_filename
        if log_filename is None:
            log_filename = nanoparticle.log_filename
        self.log_filename = log_filename
        self.title = title

        # check if given input file exists
        if input_filename is not None and not path.isfile(input_filename):
            raise DIMPyError(f'Input file `{input_filename}` does not exist!')

        # read frequency
        if freqs is not None:
            if isinstance(freqs, (int, float)):
                self.freqs = np.array([freqs])
            else:
                self.freqs = np.array(freqs)
            self.nFreqs = len(self.freqs)
        elif input_filename is not None:
            self._read_frequency()

        # assume a static polarizability
        else:
            self.freqs = np.zeros((1))
            self.nFreqs = 1

        # set attributes to be calculated later
        self._t0 = None
        self._t1 = None
        self._t2 = None
        self._A_matrix = None
        self.total_energy = None
        self.polarizability = None
        self._iFreq = None
        self._dtype = None
        self.cAbsorb  = None
        self.cScatter = None
        self.cExtinct = None
        self.qAbsorb  = None
        self.qScatter = None
        self.qExtinct = None

        # end the timer
        end_time = self._timer.endTimer('Init Calc')
        self.log('Finished Initializing calculation, '
                 '{0:.3f} seconds'.format(end_time[1]),
                 time=end_time[0])

    def _read_frequency(self, filename=None):
        '''Reads the frequency from a DIMPy input file and convert
        to atomic units.'''

        from .constants import EV2HART, NM2HART, HZ2HART, WAVENUM2HART, \
                               NOCONV, HART2NM

        if filename is None: filename = self.input_filename

        reader = self._initialize_input_reader()

        freq = reader.add_mutually_exclusive_group()
        freq.add_line_key('frequency',
                          type=('ev', 'nm', 'hz', 'cm-1', 'hartree', 'au'),
                          glob={'len': '+', 'type': float})
        freq.add_line_key('freqrange',
                          type=[('ev', 'nm', 'hz', 'cm-1', 'hartree', 'au'),
                                float, float, int])

        options = reader.read_input(filename)

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
            self.freqs = np.array(freqs, dtype=float)

        # expand the frequency range and convert appropriately
        elif options.freqrange is not None:
            convert = conversion[options.freqrange[0]]
            start, stop, num = options.freqrange[1:]
            self.freqs = convert(np.linspace(start, stop, num))
            self.freqs = np.sort(self.freqs) # sort in order of increasing energy

        self.nFreqs = len(self.freqs)


    def _initialize_input_reader(self):
        '''Initializes and returns the DIMPy input file reader object.
        I did it this way because I don't want to call this
        identical code many times.

        Valid comment characters are `::`, `#`, `!`, and `//`
        '''
        reader = InputReader(comment=['!', '#', '::', '//'],
                 case=False, ignoreunknown=True)
        return reader


    def _read_element_properties(self, filename=None):
        '''Read in the properties of each element.'''

        from .constants import ELEMENTS

        if filename is None: filename = self.input_filename

        reader = self._initialize_input_reader()

        # Add a parameter block for each element
        for element in ELEMENTS:
            e = reader.add_block_key(element)

            # required keys
            e.add_line_key('rad', type=float, required=True)

            # optional keys
            e.add_line_key('exp',   type=str, case=True, default=None)

        # read in all the element parameters
        options = reader.read_input(filename) 

    @property
    def wavelength_nm(self):
        '''Returns the frequencies as a wavelength in nm.'''
        from .constants import HART2NM
        return HART2NM(self.freqs)

    ###########################################################
    # Matrix and Tensor Functions
    # FIXME: These are not written to be efficient right now
    # consider using scipy.sparse to make them memory efficient
    ###########################################################

    @property
    def t0(self):
        '''The zeroth-order interaction tensor.'''
        if self._t0 is None:

            self.log('Generating T0 tensor',
                     time=self._timer.startTimer('T0 Tensor'))

            dists = self.nanoparticle.distances
            self._t0 = np.divide(1, dists, out=np.zeros_like(dists),
                                 where=dists!=0, dtype=np.float32)

            end_time = self._timer.endTimer('T0 Tensor')
            self.log('Finished generating T0 tensor, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])

            self.log('T0 Size (MB): ', self._t0.nbytes / (1024)**2)

        return self._t0

    @property
    def t1(self):
        '''The first-order interaction tensor.'''
        if self._t1 is None:

            self.log('Generating T1 tensor',
                     time=self._timer.startTimer('T1 Tensor'))

            r_vec = self.nanoparticle.r_vec
            r_inv = self.t0
            r3_inv = r_inv * r_inv * r_inv
            self._t1 = -1.0 * r3_inv[:,:,np.newaxis] * r_vec

            end_time = self._timer.endTimer('T1 Tensor')
            self.log('Finished generating T1 tensor, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])

        return self._t1

    @property
    def t2(self):
        '''The second-order interaction tensor.'''
        if self._t2 is None:

            self.log('Generating T2 tensor',
                     time=self._timer.startTimer('T2 Tensor'))

            # do the off diagonal terms
            r_vec = self.nanoparticle.r_vec
            r_inv = self.t0
            r3_inv = r_inv * r_inv * r_inv
            r5_inv = r3_inv * r_inv * r_inv

            self._t2 = 3 * np.einsum('ij,ijk,ijl->ijkl', r5_inv, r_vec,
                                     r_vec, dtype=np.float32, casting='same_kind')
            self._t2[:,:,0,0] -= r3_inv
            self._t2[:,:,1,1] -= r3_inv
            self._t2[:,:,2,2] -= r3_inv

            end_time = self._timer.endTimer('T2 Tensor')
            self.log('Finished generating T2 tensor, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])

            self.log('T2 Size (MB): ', self._t2.nbytes / (1024)**2)

        return self._t2

    def A_matrix(self, omega=None):
        '''Returns the A-matrix.'''
        # start the timer
        self.log('Generating A-matrix for this frequency',
                 time=self._timer.startTimer('A-matrix'))

        natoms = self.nanoparticle.natoms
        pol = 1 / self.nanoparticle.atomic_polarizabilities(omega)

        A_matrix = np.zeros_like(self.t2, dtype=self._dtype)
        A_matrix -= self.t2

        for i in range(natoms):
            A_matrix[i,i,:,:] = 0
            for a in range(3):
                A_matrix[i,i,a,a] = pol[i]

        temp = np.swapaxes(A_matrix, 1, 2)
        del(A_matrix)
        temp2 = temp.reshape(natoms * 3, natoms * 3)
        del (temp)
        self._A_matrix = temp2

        # end the timer
        end_time = self._timer.endTimer('A-matrix')
        self.log('Finished generating A-matrix, '
                 '{0:.3f} seconds'.format(end_time[1]),
                 time=end_time[0])

        self.log('Amat Size (MB): ', self._A_matrix.nbytes / (1024)**2)

        return self._A_matrix

    def solve_one_direction(self, Amat, dimension=0, x0=None):

        # get direction from dimension
        if dimension == 0:
            xyz = 'X'
        elif dimension == 1:
            xyz = 'Y'
        elif dimension == 2:
            xyz = 'Z'
        else:
            xyz = 'UNKNOWN'

        self.log(f'Solving {xyz} direction',
                 time=self._timer.startTimer(f'{xyz} direction'))

        natoms = self.nanoparticle.natoms

        # print iterations and residuals to logfile
        self._iteration = 0
        self._itertime = time()
        def report(xk):
            self._iteration += 1
            if self._iteration%10 == 0:
                ellapsed_time = time() - self._itertime
                self._itertime = time()
                self.log(f'Iter: {self._iteration//10:>4d}, Err: {xk:8.2e},'
                         f' {ellapsed_time:6.3f} s')

        E = np.zeros((natoms, 3), dtype=np.float32)
        E[:, dimension] = 1
        E = E.reshape(natoms * 3)
#        mu = np.linalg.solve(Amat, E)
        mu, info = sp.sparse.linalg.gmres(Amat, E, x0=x0, callback=report)
        E = E.reshape(natoms, 3)
        mu = mu.reshape(natoms, 3)

        self._print_atomic_dipoles(xyz, mu)
        if np.isrealobj(mu):
            self._calc_total_energy(dimension, mu, E)
        self._calc_polarizability(dimension, mu)

        # end the timer
        end_time = self._timer.endTimer(f'{xyz} direction')
        self.log(f'Finished {xyz} direction, '
                 '{0:.3f} seconds'.format(end_time[1]),
                 time=end_time[0])

        return mu

    def _calc_total_energy(self, dim, mu, E):
        if self.total_energy is None:
            self.total_energy = np.zeros((3))
        self.total_energy[dim] = -0.5 * (mu[:,dim] * E[:,dim]).sum()

    def _calc_polarizability(self, dim, mu):
        if self.polarizability is None:
            self.polarizability = np.zeros((3,3), dtype=self._dtype)
        self.polarizability[dim,:] = mu.sum(axis=0)

    def _calc_efficiencies(self, pol, omega):

        from .constants import HART2NM, BOHR2NM
        from math import pi

        wavelength  = HART2NM(omega)
        k           = 2 * pi /wavelength
        iso         = np.trace(pol) / 3.0
        iso        *= BOHR2NM(1)**3 # convert pol to au^3 to nm^3
        cAbsorb     = k * iso.imag
        cScatter    = k**4 * np.abs(iso.real**2 + iso.imag**2) / (6 * pi)
        cExtinct    = cAbsorb + cScatter

        # calculate the radius of a sphere with equivalent volume
        effrad      = (( 3.0 / ( 4.0 * pi )) * self.nanoparticle.volume )**(1./3.)
        effrad_sq   = effrad * effrad
        qAbsorb     = cAbsorb / (pi * effrad_sq)
        qScatter    = cScatter / (pi * effrad_sq)
        qExtinct    = cExtinct / (pi * effrad_sq)

        if self.cAbsorb is None:
            self.cAbsorb  = np.array([cAbsorb])
            self.cScatter = np.array([cScatter])
            self.cExtinct = np.array([cExtinct])
            self.qAbsorb  = np.array([qAbsorb])
            self.qScatter = np.array([qScatter])
            self.qExtinct = np.array([qExtinct])
        else:
            self.cAbsorb  = np.append(self.cAbsorb,  [cAbsorb])
            self.cScatter = np.append(self.cScatter, [cScatter])
            self.cExtinct = np.append(self.cExtinct, [cExtinct])
            self.qAbsorb  = np.append(self.qAbsorb,  [qAbsorb])
            self.qScatter = np.append(self.qScatter, [qScatter])
            self.qExtinct = np.append(self.qExtinct, [qExtinct])

        from .printer import print_efficiencies
        print_efficiencies(qAbsorb, qScatter, qExtinct, cAbsorb, cScatter,
                           cExtinct, self.out)

    #################################
    # End Matrix and Tensor functions
    #################################

    ########################
    # Begin main run routine
    ########################

    def run(self):
        '''Runs the calculation.'''

        from .printer import print_welcome, print_energy, \
                             print_polarizability

        # print header
        print_welcome(self.out)
        self.nanoparticle._print_nanoparticle()

        # generate the interaction tensors
        if self._t2 is None:
            trash = self.t2

        # cycle over each frequency
        for self._iFreq in range(self.nFreqs):

            omega = self.freqs[self._iFreq]

            # check if this is a static calculation
            if omega == 0 or omega is None:
                self._dtype = np.float32
            else:
                self._dtype = np.complex64

            # print frequency header
            self._print_freq_header(self.freqs[self._iFreq])

            # create the A matrix
            Amat = self.A_matrix(omega)

            # run each direction
            for ixyz in range(3):
                self.solve_one_direction(Amat, ixyz)

            # print energy terms
            if omega == 0 or omega is None:
                print_energy(self.total_energy, self.out)

            # print polarizability tensor
            print_polarizability(self.polarizability, self.out)

            # calculate and print efficiencies
            if omega is not None and omega != 0:
                self._calc_efficiencies(self.polarizability, omega)

        # print timing statistics
        self._timer.endAllTimers()
        self._timer.dumpTimings(verbosity=0, out=self.out)


    ######################
    # End main run routine
    ###################### 

    # Printing Methods specific to the calculation
    def _print_atomic_dipoles(self, xyz, mu):
        print_atomic_dipoles(xyz, self.nanoparticle.atoms, mu, self.out)

    def _print_freq_header(self, omega):

        from .constants import HART2EV, HART2NM, HART2WAVENUM, HART2HZ

        string = '*****  Performing calculation at:  *****'

        # get frequencies in all other units
        au = omega
        eV = HART2EV(omega)
        if omega > 0:
            nm = HART2NM(omega)
        else:
            nm = float('inf')
        cm = HART2WAVENUM(omega)
        Hz = HART2HZ(omega)

        # print the frequency header
        self.out(('*'*len(string)).center(79))
        self.out(string.center(79))
        self.out('***** ---------------------------- *****'.center(79))
        self.out(f'***** {au:8f} a.u. {eV:9f} eV   *****'.center(79))
        self.out(f'***** {nm:8.2f} nm {cm:11.2f} cm-1 *****'.center(79))
        self.out(f'***** {Hz:23.4e} Hz   *****'.center(79))
        self.out(('*'*len(string)).center(79))
        self.out()
        

DDA = CalcMethod
