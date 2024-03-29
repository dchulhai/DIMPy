import math
import time

import numpy as np
import scipy as sp
from scipy import linalg, sparse
from scipy.sparse import linalg
import numexpr as ne

from ..tools.constants import HART2NM, BOHR2NM, NM2BOHR
from ..tools.constants import HART2EV, HART2WAVENUM, HART2HZ
from ..dimpy_error import DIMPyError
from ..tools.memory import check_memory
from ..tools.printer import Output, print_atomic_dipoles
from ..tools.printer import print_efficiencies
from ..tools.printer import print_welcome, print_energy, print_header
from ..tools.printer import print_polarizability, print_input_file
from ..tools.timer import Timer, check_time


class CalcMethodBase(object):
    """Calculation method base class.

    Use a discrete dipole approximation (DDA) method to calculate
    induced atomic dipoles. Class also used as the DIMPy calculation base
    class.

    :param nanoparticle: Nanoparticle to use
    :type nanoparticle: :class:`dimpy.nanoparticle.nanoparticle.Nanoparticle`

    :param output_filename: The name of the DIMPy output file, default None
    :type output_filename: str or None, optional

    :param log_filename: The name of the DIMPy log file, default None
    :type log_filename: str or None, optional

    :param freqs: Frequencies at which to perform calculations in atomic
        units (Hartrees), default None (static, or zero, frequency)
    :type freqs: int or float or None, optional

    :param title: Title of this calculation, default None
    :type title: str or None, optional

    :param kdir: Unit k-vector for calculations with retardation
        effects, default is None (a non-retarded calculation)
    :type kdir: numpy.ndarray or list or None, optional

    :param solver: Linalg solver to use, default is "gmres". Possible options
        are given in :attr:`dimpy.input_file.read_input_file.ReadInput.solvers`
    :type solver: str, optional

    :param r_max_pbc: Maximum distance to calculate the interactions between
        periodic images of the unit cell, default is 500 bohrs
    :type r_max_pbc: float, optional

    :param verbose: Verbosity level, default is None (use verbose level from
        :class:`dimpy.nanoparticle.nanoparticle.Nanoparticle`)
    :type verbose: int or None, optional

    :param debug: Debug flag, default is None (use debug from
        :class:`dimpy.nanoparticle.nanoparticle.Nanoparticle`)
    :type debug: bool or None, optional

    :cvar float total_energy: Total energy of the nanoparticle

    :cvar numpy.ndarray polarizabilities: Polarizabilities calculated
        at each frequency

    :cvar numpy.ndarray isotropic_polarizabilities: Isotropic polarizabilities
        calculated at each frequency

    :cvar float cAbsorb: Absorbance cross-section

    :cvar float cScatter: Scattering cross-section

    :cvar float cExtinct: Extinction cross-section

    :cvar float qAbsorb: Absorbance efficiency

    :cvar float qScatter: Scattering efficiency

    :cvar float qExtinct: Extinction efficiency

    """

    def __init__(self, nanoparticle, output_filename=None,
                 log_filename=None, freqs=None, title=None, kdir=None,
                 solver='gmres', r_max_pbc=500, verbose=None, debug=None):
        """Initialize the DIMPy class."""

        # use the nanoparticle attributes
        try:
            self.nanoparticle = nanoparticle
            if output_filename is None:
                self.output_filename = nanoparticle.output_filename
                self.out = nanoparticle.out
            else:
                self.output_filename = output_filename
                self.out = Output(filename=output_filename)
            if verbose is None:
                self.verbose = nanoparticle.verbose
            else:
                self.verbose = verbose
            if debug is None:
                self.debug = nanoparticle.debug
            else:
                self.debug = debug
            self.out = nanoparticle.out
            self.log = nanoparticle.log
            self._memory = nanoparticle._memory
            self._timer = nanoparticle._timer
            self.debug = nanoparticle.debug
            self.pbc = nanoparticle.pbc
            self.natoms = self.nanoparticle.natoms
        except AttributeError:
            raise DIMPyError('AtributeError: Did you forget to `.build()`'
                             'the Nanoparticle object?')

        start_time = self._timer.startTimer('CalcMethodBase.__init__')
        if self.verbose > 0 or self.debug:
            self.log('Initializing calculation', time=start_time)

        # set default attributes and parameters
        if log_filename is None:
            log_filename = nanoparticle.log_filename
        self.log_filename = log_filename
        self.title = title
        self.solver = solver
        self.r_max_pbc = r_max_pbc

        # get k-direction
        if kdir is not None:
            if len(kdir) != 3:
                raise DIMPyError("'kdir' must be a length 3 array or list!")
            self.kdir = np.array(kdir) / sp.linalg.norm(kdir)
        else:
            self.kdir = None

        # read frequency
        if freqs is not None:
            if isinstance(freqs, (int, float)):
                self.freqs = np.array([freqs])
            else:
                self.freqs = np.array(freqs)
            self.nFreqs = len(self.freqs)

        # assume a static polarizability
        else:
            self.freqs = np.zeros((1))
            self.nFreqs = 1

            # you cannot have a k-direction for a static calculation
            if self.kdir is not None:
                raise DIMPyError("You cannot have a k-vector for a "
                                 "static calculation!")

        # set attributes to be calculated later
        self._t0 = None
        self._t1 = None
        self._t2 = None
        self._A_matrix = None
        self.total_energy = None
        self._polarizability = None
        self.polarizabilities = None
        self.atomic_dipoles = None
        self._iFreq = None
        self._dtype = None
        self.cAbsorb  = None
        self.cScatter = None
        self.cExtinct = None
        self.qAbsorb  = None
        self.qScatter = None
        self.qExtinct = None

        # end the timer
        end_time = self._timer.endTimer('CalcMethodBase.__init__')
        if self.verbose > 0 or self.debug:
            self.log('Finished Initializing calculation, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])

    interaction = 'DDA'
    """Discrete Dipole Interaction"""

    @property
    def wavelength_nm(self):
        """Frequencies as wavelengths in nm."""
        return HART2NM(self.freqs)

    def print_calc_information(self, output=None):
        """Print the information for this calculation."""
        if output is None:
            output = self.out

        print_header('Calculation Information', output)
        output(f'Title             : {self.title}')
        output(f'Interaction type  : {self.interaction}')
        approx = 'Quasi-static approximation'
        if self.kdir is not None:
            approx = (f'k = {self.kdir[0]:4.2f} {self.kdir[1]:4.2f} '
                      f'{self.kdir[2]:4.2f}')
        output(f'Wave vector mag.  : {approx}')
        output(f'Solver            : {self.solver}')
        output()


    ###########################################################
    # Matrix and Tensor Functions
    # FIXME: These are not written to be efficient right now
    # consider using scipy.sparse to make them memory efficient
    ###########################################################

    @check_memory
    @check_time(log='debug')
    def t0(self, **kwargs):
        r"""Calculate and return the zeroth-order interaction tensor.

        .. math::

            T^{(0)}_{ij} = \frac{1}{|r_{ij}|}

        """

        if self._t0 is None:

            dists = self.nanoparticle.distances
            self._t0 = np.divide(1, dists, out=np.zeros_like(dists),
                                 where=dists!=0, dtype=np.float32)

        return self._t0

    @check_memory
    @check_time(log='debug')
    def t1(self, **kwargs):
        r"""Calculate and return the first-order interaction tensor.

        .. math::

            T^{(1)}_{ij,\alpha} = -\frac{r_{ij,\alpha}}{|r_{ij}|^3}

        """

        if self._t1 is None:

            r_vec = self.nanoparticle.r_vec
            r_inv = self.t0()
            r3_inv = r_inv * r_inv * r_inv
            self._t1 = -1.0 * r3_inv[:,:,np.newaxis] * r_vec

        return self._t1

    @check_memory
    @check_time(log='debug')
    def t2(self, vector=None, **kwargs):
        r"""Calculate and return the second-order interaction tensor.

        .. math::

            T^{(2)}_{i\alpha,j\beta} = \frac{3 r_{ij,\alpha} r_{ij,\beta}}{|r_{ij}|^5} -
            \frac{\delta_{\alpha\beta}}{|r_{ij}|^3}

        """

        # get r_vec and r_inv
        if vector is None:
            r_vec = self.nanoparticle.r_vec
            dists = self.nanoparticle.distances
        else:
            r_vec = self.nanoparticle.r_vec + vector
            c_old = self.nanoparticle.coordinates
            c_new = self.nanoparticle.coordinates + vector
            dists = sp.spatial.distance.cdist(c_old, c_new)
        r_inv = ne.evaluate("1/dists")

        r3_inv = ne.evaluate("r_inv * r_inv * r_inv")
        r5_inv = ne.evaluate("r3_inv * r_inv * r_inv")
        del(r_inv)

        # calc off-diagonal terms
        natoms = self.nanoparticle.natoms
        r5_inv = r5_inv.reshape(1, natoms * natoms, 1)
        r_vec1 = r_vec.copy().reshape(1, natoms * natoms, 3)
        r_vec2 = np.swapaxes(r_vec1, 0, 2)
        t2 = ne.evaluate("3 * r5_inv * r_vec1 * r_vec2")
        t2 = t2.reshape(3, natoms, natoms, 3)
        t2 = np.swapaxes(t2, 0, 1)

        # calc diagonal terms
        t2[:,0,:,0] -= r3_inv
        t2[:,1,:,1] -= r3_inv
        t2[:,2,:,2] -= r3_inv

        return t2


    @check_memory
    @check_time
    def A_matrix(self, omega=None):
        r"""Calculate the A-matrix.

        .. math::

            A_{ij,\alpha\beta} = \begin{cases}
                \alpha_{\alpha\beta}^{-1}, & \text{if $i=j$} \\
                -T^{(2)}_{ij,\alpha\beta}, & \text{if $i\neq j$}
            \end{cases}

        :param omega: Incident frequency, default None
        :type omega: float, optional

        :returns: The A-matrix
        :rtype: numpy.ndarray (float or complex)

        """

        natoms = self.nanoparticle.natoms
        pol = self.nanoparticle.atomic_polarizabilities(omega)
        pol_inv = ne.evaluate("1/pol")

        # initialize the A-matrix and get off-diagonal terms for
        # origin cell
        A_matrix = np.zeros((natoms, 3, natoms, 3), dtype=self._dtype)
        A_matrix -= self.t2(omega=omega)

        # get diagonal elements (inverse polarizability)
        for i in range(natoms):
            A_matrix[i,:,i,:] = 0
            for a in range(3):
                A_matrix[i,a,i,a] = pol_inv[i]

        # calculate interactions from image cells
        if self.pbc is not None:

            # get lattice vectors and max unit cells
            mvec = self.pbc[0]
            mmax = max(1, int(self.r_max_pbc / np.linalg.norm(self.pbc[0])))
            if len(self.pbc) > 1:
                nvec = self.pbc[1]
                nmax = max(1, int(self.r_max_pbc / np.linalg.norm(self.pbc[1])))
            else:
                nvec = np.zeros((3))
                nmax = 0 

            # cycle over the first lattice vector
            for m in range(-mmax,mmax+1,1):

                # cycle over the second lattice vector
                for n in range(-nmax,nmax+1,1):

                    # skip if this is the origin unit cell
                    if m == 0 and n == 0:
                        continue

                    # get new vector
                    new_vec = m * mvec + n * nvec

                    # skip if our distance is greater than needed
                    dist = np.linalg.norm(new_vec)
                    if dist > self.r_max_pbc:
                        continue

                    # calculate the interaction and add to the A-matrix
                    t2 = self.t2(omega=omega, vector=new_vec)
                    A_matrix -= t2

        # reshape and return A matrix
        self._A_matrix = A_matrix.reshape(natoms * 3, natoms * 3)
        return self._A_matrix

    @check_memory(log='debug')
    @check_time(log='debug')
    def _get_Einc(self, dimension, natoms, omega):
        """Get incident field for this direction."""
        # E magnitude is 1 in each dimension
        if self.kdir is None:
            E = np.zeros((natoms, 3), dtype=np.float32)
            E[:, dimension] = 1 

        # E magnitude is perpendicular to kdir
        else:
            E = np.zeros((natoms, 3), dtype=np.complex64)

            # get a vector for this e-field
            Evec = np.zeros((3))
            Evec[dimension] = 1 

            # field is only perpendicular to k-vector
            perp_factor = np.sin(np.arccos(np.clip(np.dot(Evec, self.kdir),
                                                   -1, 1)))
            perp_factor = perp_factor**2

            # if the field is perpendicular, calculate it
            if perp_factor != 0:

                # calculate the k-vector
                kvec = 2 * np.pi / NM2BOHR(HART2NM(omega)) * self.kdir

                # calculate k dot r
                k_dor_r = np.dot(self.nanoparticle.coordinates, kvec)

                # the field is e^{ikr}
                E[:,dimension] = np.exp(1j * k_dor_r) * perp_factor

        # reshape and return
        E = E.reshape(natoms * 3)
        return E

    @check_memory
    @check_time(log='debug')
    def solve_one_direction(self, Amat, dimension=0, x0=None, omega=None):
        """Solve the induced dipoles for incident fields in one of the
        three Cartesian directions.

        :param Amat: A-matrix for this frequency
        :type Amat: :meth:`A_matrix`

        :param int dimension: Direction of incident electric field polarization
            for calculation, default is 0. 0 corresponds to "x", 1 corresponds to
            "y", and 2 corresponds to "z".

        :param x0: Starting guess for calculation, default None
        :type x0: numpy.ndarray, optional

        :param omega: Incident frequency, default None
        :type omega: float, optional

        :returns: Induced dipoles
        :rtype: numpy.ndarray (flattened 3 by natoms)

        """

        # get direction from dimension
        if dimension == 0:
            xyz = 'X'
        elif dimension == 1:
            xyz = 'Y'
        elif dimension == 2:
            xyz = 'Z'
        else:
            xyz = 'UNKNOWN'

        start_time = self._timer.startTimer(f'{xyz} direction')
        if self.verbose > 0 or self.debug:
            self.log(f'Solving {xyz} direction', time=start_time)

        natoms = self.nanoparticle.natoms

        # get incident fields
        E = self._get_Einc(dimension, natoms, omega)

        # solver for dipoles
        mu = self.solve(Amat, E, x0=x0)


        # reshape field and dipoles
        E = E.reshape(natoms, 3)
        mu = mu.reshape(natoms, 3)

        # print dipoles if needed
        self._print_atomic_dipoles(xyz, mu)
        if np.isrealobj(mu):
            self._calc_total_energy(dimension, mu, E)
        self._calc_polarizability(dimension, mu)

        # end the timer
        end_time = self._timer.endTimer(f'{xyz} direction')
        if self.verbose > 0 or self.debug:
            self.log(f'Finished {xyz} direction, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])

        return mu.flatten()


    @check_memory(log='debug')
    @check_time(log='debug')
    def solve(self, Amat, E, x0=None, solver=None):
        """Choose the appropriate solver based on the solver given.

        :param Amat: A-matrix for this frequency
        :type Amat: :meth:`A_matrix`

        :param E: Incident electric field for this calculation
        :type E: numpy.ndarray (3 by 3)

        :param x0: Starting guess for calculation, default None
        :type x0: numpy.ndarray, optional

        :param solver: Solver to use, default is None (solver
            set previously). Options for solver are in
            :attr:`dimpy.read_input_file.ReadInput.solvers`
        :type solver: str or None, optional

        :returns: Induced dipoles
        :rtype: numpy.ndarray (3 by natoms)

        """

        if solver is None: solver = self.solver

        # print iterations and residuals to logfile
        self._iteration = 0 
        self._itertime = time.perf_counter()

        def report(xk):
            self._iteration += 1
            if self._iteration%10 == 0:
                ellapsed_time = time.perf_counter() - self._itertime
                self._itertime = time.perf_counter()
                if self.verbose > 1 or self.debug:
                    self.log(f'Iter: {self._iteration//10:>4d}, Err: {xk:8.2e},'
                             f' {ellapsed_time:6.3f} s')
        self._old_solution_vector = None

        def report2(xk):
            self._iteration += 1
            self._new_solution_vector = xk.copy()
            if self._iteration > 1:
                ellapsed_time = time.perf_counter() - self._itertime
                self._itertime = time.perf_counter()
                old = self._old_solution_vector
                new = self._new_solution_vector
                denom = self.nanoparticle.natoms * 3
                error = (ne.evaluate("sum(abs(old - new))") / denom).real
                if self.verbose > 1 or self.debug:
                    self.log(f'Iter: {self._iteration:>4d}, Err: {error:8.2e},'
                             f' {ellapsed_time:6.3f} s')
            self._old_solution_vector = xk.copy()

        # get appropriate solver
        if solver == 'direct' or solver == 'solve':
            mu = sp.linalg.solve(Amat, E)

        elif solver == 'bicg':
            mu, info = sp.sparse.linalg.bicg(Amat, E, x0=x0,
                                             callback=report2, atol=1e-5)

        elif solver == 'bicgstab':
            mu, info = sp.sparse.linalg.bicgstab(Amat, E, x0=x0,
                                                 callback=report2, atol=1e-5)

        elif solver == 'cg':
            mu, info = sp.sparse.linalg.cg(Amat, E, x0=x0,
                                           callback=report2, atol=1e-5)

        elif solver == 'cgs':
            mu, info = sp.sparse.linalg.cgs(Amat, E, x0=x0,
                                            callback=report2, atol=1e-5)

        elif solver == 'gmres':
            mu, info = sp.sparse.linalg.gmres(Amat, E, x0=x0, atol=1e-5,
                                              callback=report, callback_type='legacy')

        elif solver == 'lgmres':
            mu, info = sp.sparse.linalg.lgmres(Amat, E, x0=x0,
                                               callback=report2, atol=1e-5)

        elif solver == 'minres':
            mu, info = sp.sparse.linalg.minres(Amat, E, x0=x0,
                                               callback=report2)

        elif solver == 'qmr':
            mu, info = sp.sparse.linalg.qmr(Amat, E, x0=x0,
                                            callback=report2, atol=1e-5)

        elif solver == 'gcrotmk':
            mu, info = sp.sparse.linalg.gcrotmk(Amat, E, x0=x0,
                                                callback=report2, atol=1e-5)

        else:
            raise DIMPyError("Unimplemented solver solver specified: "
                + solver)

        return mu

    #################################
    # End Matrix and Tensor functions
    #################################

    ###############################
    # Begin calculated properties #
    ###############################

    @check_memory(log='debug')
    @check_time(log='debug')
    def _calc_total_energy(self, dim, mu, E):
        if self.total_energy is None:
            self.total_energy = np.zeros((3))
        self.total_energy[dim] = -0.5 * (mu[:,dim] * E[:,dim]).sum()


    @check_memory(log='debug')
    @check_time(log='debug')
    def _calc_polarizability(self, dim, mu):
        if self._polarizability is None:
            self._polarizability = np.zeros((3,3), dtype=self._dtype)
        self._polarizability[dim,:] = mu.sum(axis=0)

    @property
    @check_memory(log='debug')
    @check_time(log='debug')
    def isotropic_polarizabilities(self):
        """Isotropic polarizabilities calculated at each frequency."""
        return np.trace(self.polarizabilities, axis1=1, axis2=2) / 3.0

    @check_memory(log='debug')
    @check_time(log='debug')
    def _calc_efficiencies(self, pol, omega):

        wavelength  = HART2NM(omega)
        k           = 2 * math.pi /wavelength
        iso         = np.trace(pol) / 3.0
        iso        *= BOHR2NM(1)**3 # convert pol from au^3 to nm^3
        cAbsorb     = k * iso.imag
        cScatter    = k**4 * np.abs(iso.real**2 + iso.imag**2) / (6 * math.pi)
        cExtinct    = cAbsorb + cScatter

        # calculate the radius of a sphere with equivalent volume
        effrad      = (( 3.0 / ( 4.0 * math.pi )) * self.nanoparticle.volume )**(1./3.)
        effrad_sq   = effrad * effrad
        qAbsorb     = cAbsorb / (math.pi * effrad_sq)
        qScatter    = cScatter / (math.pi * effrad_sq)
        qExtinct    = cExtinct / (math.pi * effrad_sq)

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

        if self.verbose > 0 or self.debug:
            print_efficiencies(qAbsorb, qScatter, qExtinct, cAbsorb, cScatter,
                               cExtinct, self.out)

    #############################
    # End calculated properties #
    #############################

    ########################
    # Begin main run routine
    ########################

    def run(self):
        """Run the calculation."""

        # print headers
        if self.verbose > 1 or self.debug:
            print_welcome(self.out)
        # FIXME
#        if self.verbose > 2 or self.debug:
#            print_input_file(opts, output=self.out)
        if self.verbose > 1 or self.debug:
            self.print_calc_information()
            self.nanoparticle.print_nanoparticle()

        # temporarily hold the atomic dipoles here for
        # restart when calculating multiple frequencies
        self.atomic_dipoles = None
        x0 = [None, None, None]

        # cycle over each frequency
        for self._iFreq in range(self.nFreqs):

            omega = self.freqs[self._iFreq]

            # reset polarizability
            self._polarizability = None

            # check if this is a static calculation
            if omega == 0 or omega is None:
                self._dtype = np.float32
            else:
                self._dtype = np.complex64

            # print frequency header
            if self.verbose > 0 or self.debug:
                self._print_freq_header(self.freqs[self._iFreq])

            # create the A matrix
            Amat = self.A_matrix(omega)

            if self.atomic_dipoles is None:
                self.atomic_dipoles = np.zeros((1,3,self.natoms,3),
                    dtype=self._dtype)
            else:
                self.atomic_dipoles = np.append(self.atomic_dipoles,
                    np.zeros((1,3,self.natoms,3), dtype=self._dtype), axis=0)
            # run each direction
            for ixyz in range(3):
                atomic_dipoles = self.solve_one_direction(Amat,
                    ixyz, x0=x0[ixyz], omega=omega)
                self.atomic_dipoles[self._iFreq,ixyz,:] = atomic_dipoles.reshape(
                                                            self.natoms,3)
                x0[ixyz] = atomic_dipoles

            # print energy terms
            if ( self.total_energy is not None and
                 (self.verbose > 0 or self.debug) ):
                print_energy(self.total_energy, self.out)

            # print polarizability tensor
            if self.polarizabilities is None:
                self.polarizabilities = np.array([self._polarizability])
            else:
                self.polarizabilities = np.append(self.polarizabilities,
                                                  [self._polarizability], axis=0)
            if self.verbose > 0 or self.debug:
                print_polarizability(self._polarizability, self.out)

            # calculate and print efficiencies
            if omega is not None and omega != 0:
                self._calc_efficiencies(self._polarizability, omega)

        # print memory statistics
        if self.verbose > 0 or self.debug:
            self._memory.printLogs(verbosity=0, out=self.out)

        # print timing statistics
        self._timer.endAllTimers()
        if self.verbose > 0 or self.debug:
            self._timer.dumpTimings(verbosity=0, out=self.out)


    ######################
    # End main run routine
    ###################### 

    # Printing Methods specific to the calculation
    @check_memory(log='debug')
    @check_time(log='debug')
    def _print_atomic_dipoles(self, xyz, mu):
        if self.verbose > 2 or self.debug:
            print_atomic_dipoles(xyz, self.nanoparticle.atoms, mu, self.out)

    def _print_freq_header(self, omega):

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

