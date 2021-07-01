import os

import argparse
import numpy as np
import scipy as sp
from scipy import spatial

from ..methods.base import CalcMethodBase
from ..tools.constants import HART2NM
from ..dimpy_error import DIMPyError
from ..tools.memory import Memory
from ..nanoparticle import Nanoparticle
from ..tools.printer import Output
from ..tools.timer import Timer
from .._version import __version__

class Analyzer(object):
    r"""Read and analyze the information from a DIMPy output file.

    :param filename_or_calc: The DIMPy input or output filename as 
        a string or a :class:`dimpy.methods.base.CalcMethodBase` object
    :type filename_or_calc: str or :class:`dimpy.methods.base.CalcMethodBase`

    :param log_filename: Log filename to print any information to,
        defaults to ``std.out``
    :type log_filename: str, optional

    :param debug: Whether to print debug statements, default False
    :type debug: bool, optional

    :cvar calculation: :class:`dimpy.methods.base.CalcMethodBase` object

    :cvar nanoparticle: :class:`dimpy.nanoparticle.nanoparticle.Nanoparticle` object

    """

    def __init__(self, filename_or_calc, log_filename=None, debug=False,
                  verbose=2):
        """Initialize the analyzer object given a filename or
        :class:`dimpy.methods.CalcMethodBase` object.
        """

        # set up some initializations here
        self.log_filename = log_filename
        self.debug = debug
        self.verbose = verbose

        if isinstance(filename_or_calc, CalcMethodBase):

            self.calculation  = filename_or_calc
            self.nanoparticle = self.calculation.nanoparticle

        elif isinstance(filename_or_calc, str):

            filename = filename_or_calc

            # check if this is an input or output file
            if os.path.splitext(filename)[1] == '.dimpy':
                output_filename = os.path.splitext(filename)[0] + '.out'
                if not os.path.isfile(output_filename):
                    raise DIMPyError(f'Corresponding output file '
                                     f'`{output_filename}` not found!')
            elif os.path.splitext(filename)[1] == '.out':
                output_filename = filename
            else:
                raise DIMPyError(f'File `{filename}` is not a recognized '
                                  'DIMPy input or output file!')
            self._collect_from_filename(output_filename)

        else:
            raise DIMPyError('Expected `CalcMethodBase` object or string!')


    def _collect_from_filename(self, output_filename):
        """Read the keys from the output file."""

        # Collect all data into memory for data retention
        with open(output_filename) as fl:
            f = tuple([line.rstrip() for line in fl])

        # Define lines that are accociated with various keys or properties.
        # Since these lines may appear more than once, we define three groups:
        # one where we want the first appearence, one where we want the last
        # appearance, and one where we want every appearence
        # The keys are the lines to find, and the first index of the value is the
        # associated propery, and the second is the number to add to the line
        # number to find the start of where to collect the property.
        #
        # Note that since we are defining the entire line, not just a part,
        # we can search these lines as part of a set which are designed for
        # searching.  If we defined a part of the line, we would have to scan each
        # line, being much slower.
    
        first = {
                #########
                # General
                #########
                '                             Calculation Information':
                                                            ['CALCULATION', 3],

                #############
                # Nanoparticle Input
                #############
    
                # End of the input block
                '                             Nanoparticle Information':
                                                           ['NANOPARTICLE', 3],
                ' Periodic Lattice Vector(s)':                  ['LATTICE', 2],
                ' Atom Parameter Value(s)':                  ['ATOMPARAMS', 2],
                '                                 G E O M E T R Y':
                                                               ['GEOMETRY', 6],
                # Timing stats
                '                                Timing Statistics':
                                                                 ['TIMING', 5],
        }
    
        last = { }
    
        each = {
                ################################
                # Frequency dependent properties
                ################################
    
                # The frequency header
                '                     *****  Performing calculation at:  *****':
                                                              ['FREQUENCY', 2],
                # The Polarizabilities
                '                              Polarizability Tensor':
                                                                    ['POL', 3],
                # Induced dipoles and charges
                '         Atom                Dipole                  Charge':
                                                                ['CPIMDIP', 0],
                # Induced dipoles
                '         Atom                Dipole':
                                                                 ['PIMDIP', 3],
                # Efficiencies
                '                         Efficiencies and Cross-Sections':
                                                                    ['EFF', 4],
        }
    
    
        search_lines = list(first) + list(last) + list(each)
        indices = {}
        for i, line in enumerate(f):
            if line in search_lines:
                # If the line is in the first dict, store then remove from search
                if line in first:
                    indices[first[line][0]] = i+first[line][1]
                    del first[line]
                    search_lines.remove(line)
                # If the line is in a last dict, store and overwrite
                elif line in last:
                    indices[last[line][0]] = i+last[line][1]
                # Otherwise, append this value to the index
                else:
                    try:
                        indices[each[line][0]].append(i+each[line][1])
                    except KeyError:
                        indices[each[line][0]] = []
                        indices[each[line][0]].append(i+each[line][1])

        # collect basic information about this calculation
        if 'CALCULATION' in indices:
            index = indices['CALCULATION']
            title = f[index].split()[2]
            interaction = f[index+1].split()[3]
            kdir = np.array(f[index+2].split()[6:], dtype=float)
            solver = f[index+3].split()[2]
        else:
            raise DIMPyError("'Calculation Information' not found in output")

        ###############################################
        # Information needed to generate nanoparticle #
        ###############################################

        # collect atom_params
        atom_params = {}
        if 'ATOMPARAMS' in indices:
            index = indices['ATOMPARAMS']
            i = 0
            while True:
                st = f[index+i].split()
                atom = st[0]
                key = st[1]
                value = st[2:]
                for j in range(len(value)):
                    try:
                        if len(value) == 1:
                            value = float(value[j])
                        else:
                            value[j] = float(value[j])
                    except:
                        if len(value) == 1:
                            value = value[j]
                try:
                    atom_params[atom][key] = value
                except KeyError:
                    atom_params[atom] = {key: value}
                i += 1
                if (f[index+i] == '' or
                    f[index+i] == '                                      ======'):
                    break

        # collect lattice vectors
        pbc = None
        if 'LATTICE' in indices:
            index = indices['LATTICE']
            pbc = [f[index].split()]
            if len(f[index+1].split()) == 3:
                pbc.append(f[index+1].split())
            pbc = np.array(pbc, dtype=np.float32)

        # collect geometry information
        if 'GEOMETRY' in indices:
            index = indices['GEOMETRY']
            atoms_string = ''
            i = 0
            while True:
                st = ' '.join(f[index+i].split()[1:])
                if i > 0:
                    st = '; ' + st
                atoms_string = atoms_string + st
                i += 1
                if f[index+i] == '':
                    break
        else:
            index = indices['NANOPARTICLE'] + 3
            atoms_string = f[index].split()[3]
            if st == 'None':
                raise DIMPyError('Cannot find coordinates for nanoparticle!')

        # generate nanoparticle
        nanoparticle = Nanoparticle(atoms_string, unit='bohr', pbc=pbc,
                                    atom_params=atom_params, debug=self.debug,
                                    verbose=self.verbose,
                                    output_filename=self.log_filename)
        nanoparticle.build()
        self.nanoparticle = nanoparticle
        self.log = nanoparticle.log
        self.out = nanoparticle.out
        self._memory = nanoparticle._memory
        self._timer = nanoparticle._timer
        self.natoms = nanoparticle.natoms

        ###############################
        # Method specific information #
        ###############################
 
        # Read frequencies
        if 'FREQUENCY' in indices:
            freqs = []
            for ifreq in indices['FREQUENCY']:
                freqs.append(f[ifreq].split()[1])
            freqs = np.array(freqs, dtype=float)

        # create CalcMethodBase object
        calc = CalcMethodBase(nanoparticle, freqs=freqs, title=title, kdir=kdir,
                          solver=solver, verbose=self.verbose,
                          debug=self.debug)

        # Read polarizabilities
        if 'POL' in indices:
            pols = []
            for ipol in indices['POL']:

                # Collect complex polarizability
                if 'Real' in f[ipol]:
                    pol = np.zeros((3,3), dtype=complex)
                    for i in range(3):
                        pol[i] = np.array(f[ipol+3+i].split()[1:], dtype=float)
                        pol[i] += np.array(f[ipol+10+i].split()[1:], dtype=float) * 1j


                # Collect real polarizability
                else:
                    pol = np.zeros((3,3), dtype=float)
                    for i in range(3):
                        pol[i] = np.array(f[ipol+2+i].split()[1:], dtype=float)

                pols.append(pol)
                calc.polarizabilities = np.array(pols)

        # Collect efficiencies and cross-sections
        if 'EFF' in indices:
            qAbs  = []
            qScat = []
            qExt  = []
            cAbs  = []
            cScat = []
            cExt  = []

            for ieff in indices['EFF']:
                temp = np.array(f[ieff].split(), dtype=float)
                qAbs.append(temp[0])
                qScat.append(temp[1])
                qExt.append(temp[2])
                cAbs.append(temp[3])
                cScat.append(temp[4])
                cExt.append(temp[5])

            calc.qAbsorb  = np.array(qAbs)
            calc.qScatter = np.array(qScat)
            calc.qExtinct = np.array(qExt)
            calc.cAbsorb  = np.array(cAbs)
            calc.cScatter = np.array(cScat)
            calc.cExtinct = np.array(cExt)

        # collect atomic dipoles
        if 'PIMDIP' in indices:

            atomic_dipoles = []

            for idip in indices['PIMDIP']:

                # Collect complex dipoles
                if '(R)' in f[idip]:
                    temp = np.zeros((nanoparticle.natoms, 3),
                                    dtype=complex)
                    for i in range(nanoparticle.natoms):
                         temp[i] = np.array(f[idip+i*2].split()[2:5],
                                            dtype=float)
                         temp[i] += np.array(f[idip+i*2+1].split()[:3],
                                             dtype=float) * 1j

                # Collect real dipoles
                else:
                    temp = np.zeros((nanoparticle.natoms, 3), dtype=float)
                    for i in range(nanoparticle.natoms):
                        temp[i] = np.array(f[idip+i].split()[2:5], dtype=float)

                atomic_dipoles.append(temp)

            calc.atomic_dipoles = np.array(atomic_dipoles)
            calc.atomic_dipoles = calc.atomic_dipoles.reshape(calc.nFreqs,
                3, calc.nanoparticle.natoms, 3)

        self.calculation = calc

    def plot_spectrum(self, spectrum='absorbance', backend='matplotlib'):
        """Plot the spectrum using matplotlib or plotly backend."""
        if backend == 'matplotlib':
            self._plot_with_matplotlib(spectrum=spectrum)
        else:
            raise DIMPyError (f'Backend "{backend}" not implemented!')

    def _plot_with_matplotlib(self, spectrum):

        from matplotlib import pyplot as plt

        calc = self.calculation

        x = HART2NM(calc.freqs)

        # get y-value
        if spectrum == 'absorbance':
            y = calc.cAbsorb
            ylabel = 'Absorbance'

        plt.plot(x,y,lw=2)
        plt.ylabel(ylabel)
        plt.xlabel('Wavelength (nm)')
        plt.xlim((x.min(),x.max()))
        plt.ylim((y.min(),y.max()+(y.max()-y.min())*0.05))
        
        plt.show()

    def gen_local_field(self, pos, wavelength=500, radius=3):
        """Generate fields at positions `pos`.
        pos can be either a single coordinate (size 3) or
        array of coordinates (size nx3).

        :returns: an n x 3 x 3 numpy.ndarray for fields
        """
        pos = np.array(pos, dtype=np.float32)
        if pos.ndim == 1:
            pos = pos.reshape(1,3)

        coords = self.nanoparticle.coordinates
        calc   = self.calculation

        R_vec = pos[:,np.newaxis,:] - coords[np.newaxis,:,:]
        dists = sp.spatial.distance.cdist(pos, coords)
        R_inv = np.divide(1, dists, out=np.zeros_like(dists),
                where=dists>radius, dtype=np.float32)
        R3_inv = R_inv * R_inv * R_inv
        R5_inv = R3_inv * R_inv * R_inv

        T2 = 3.0 * np.einsum('ij,ijk,ijl->ijkl', R5_inv, R_vec,
                             R_vec, dtype=np.float32, casting='same_kind')
        T2[:,:,0,0] -= R3_inv
        T2[:,:,1,1] -= R3_inv
        T2[:,:,2,2] -= R3_inv

        # get field
        nm_diff = np.abs(wavelength - calc.wavelength_nm)
        ifreq = np.argsort(nm_diff)[0]
        E = np.einsum('ijab,cjb->ica', T2, calc.atomic_dipoles[ifreq])

        return E


    def plot_fields(self, plane=('x', 0), field_dir='y', npoints=10000,
                    wavelength=500, emax=2, log_field=True, draw_atoms=True,
                    field_exp=1):
        r"""Plot the fields along a plane for a particular indicent field
        direction. Uses :mod:`matplotlib`.

        :param plane: the plane to plot, given as a tuple. The first value of
            the tuple is a str of the axis and the second value is the
            value of that axis. The default is ('x', 0) which corresponds to
            the x=0 plane
        :type plane: tuple, optional

        :param field_dir: the direction of the incident field, default 'y'
        :type field_dir: str, optional

        :param npoints: maximum number of points on the surface plot, default
            is 10000
        :type npoints: int, optional

        :param wavelength: for calculations with multiple excitaiton frequencies,
            choose the excitation that is closest to this wavelength (in nm),
            default is 500
        :type wavelength: float, optional

        :param emax: the maximum value of the local field to plot, default is 2
        :type emax: float, optional

        :param log_field: whether to plot the log of the local field, default is
            True
        :type log_field: bool, optional

        :param draw_atoms: whether to draw the atoms of the nanoparticle, default
            is True
        :type draw_atoms: bool, optional

        :param field_exp: plot :math:`|E|^\mathrm{field_exp}`, default is 1
        :type field_exp: int, optional

        """

        from matplotlib import pyplot as plt

        txyz = {'x': 0, 'y': 1, 'z': 2}
        vxyz = ['x', 'y', 'z']

        fig, ax = plt.subplots()

        # get dimensions to plot
        if plane[0].lower() == 'x':
            ix, iy, iz = (1, 2, 0)
        elif plane[0].lower() == 'y':
            ix, iy, iz = (0, 2, 1)
        elif plane[0].lower() == 'z':
            ix, iy, iz = (0, 1, 2)
        else:
            raise DIMPyError('plane must be a tuple, the first argument '
                             'of which is either "x", "y", or "z".')

        calc   = self.calculation
        coords = self.nanoparticle.coordinates

        xmin = coords.T[ix].min() - 10.0
        xmax = coords.T[ix].max() + 10.0
        ymin = coords.T[iy].min() - 10.0
        ymax = coords.T[iy].max() + 10.0
        xlen = xmax - xmin
        ylen = ymax - ymin
        step = np.sqrt(xlen * ylen / npoints)
        x    = np.arange(xmin, xmax+step, step)
        y    = np.arange(ymin, ymax+step, step)
        X, Y = np.meshgrid(x, y)

        grid = np.zeros((len(x), len(y), 3))
        grid[:,:,0] = X.T
        grid[:,:,1] = Y.T
        grid[:,:,2] = plane[1]
        grid = grid.reshape(len(x) * len(y), 3)

        R_vec = grid[:,np.newaxis,:] - coords[np.newaxis,:,:]
        dists = sp.spatial.distance.cdist(grid, coords)
        R_inv = np.divide(1, dists, out=np.zeros_like(dists),
                where=dists!=0, dtype=np.float32)
        R3_inv = R_inv * R_inv * R_inv
        R5_inv = R3_inv * R_inv * R_inv

        T2 = 3.0 * np.einsum('ij,ijk,ijl->ijkl', R5_inv, R_vec,
                             R_vec, dtype=np.float32, casting='same_kind')
        T2[:,:,0,0] -= R3_inv
        T2[:,:,1,1] -= R3_inv
        T2[:,:,2,2] -= R3_inv

        # get field
        nm_diff = np.abs(wavelength - calc.wavelength_nm)
        ifreq = np.argsort(nm_diff)[0]
        E = np.einsum('ijkl,jl->ik', T2, calc.atomic_dipoles[ifreq][txyz[field_dir]])

        E_mag = ( E[:,0].conjugate() * E[:,0] + E[:,1].conjugate() * E[:,1]
                + E[:,2].conjugate() * E[:,2] ).real + 1
        E_mag = np.sqrt(E_mag)
        E_mag = E_mag**field_exp
        if log_field:
            E_mag = np.log10(E_mag)
        E_mag = E_mag.reshape(len(x), len(y))

        # reshape X, Y to Angstrom
        X *= 0.529177
        Y *= 0.529177

        levels = np.linspace(0, emax, 101)
        contour = ax.contourf(X.T, Y.T, E_mag, levels=levels, vmax=emax,
                              cmap='plasma')
        e_string = '|E|'
        if field_exp != 1:
            e_string = e_string + f'^{field_exp}'
        if log_field:
            plt.colorbar(contour, label='$\\mathrm{log}('+e_string+')$')
        else:
            plt.colorbar(contour, label='$'+e_string+'$')

        # now we draw the atoms of the nanoparticle
        if draw_atoms:
            idx = np.argsort(coords.T[iz])
            for i in range(len(coords)):
                x0 = coords.T[ix][idx][i] * 0.529177
                y0 = coords.T[iy][idx][i] * 0.529177

                color = '#d9c86a'
                if calc.nanoparticle.atom[i][0] == 'Ag': color = '#c7cdd6'

                circle = plt.Circle((x0,y0), 1, fc=color, edgecolor='k', lw=1)
                ax.add_patch(circle)

        plt.xlabel(f'{vxyz[ix]}-axis')
        plt.ylabel(f'{vxyz[iy]}-axis')

        plt.xlim((xmin*0.529177, xmax*0.529177))
        plt.ylim((ymin*0.529177, ymax*0.529177))
        plt.show()

    def plot_field_density(self, field_dir='x', rmin=3, npoints=100000, wavelength=500,
                           **kwargs):

        from mayavi import mlab

        txyz = {'x': 0, 'y': 1, 'z': 2}
        coords = self.nanoparticle.coordinates
        calc   = self.calculation

        xmin = coords.T[0].min() - 10
        xmax = coords.T[0].max() + 10
        ymin = coords.T[1].min() - 10
        ymax = coords.T[1].max() + 10
        zmin = coords.T[2].min() - 10
        zmax = coords.T[2].max() + 10

        step = (((xmax-xmin) * (ymax-ymin) * (zmax-zmin)) / npoints)**(1/3)

        x = np.arange(xmin, xmax+step, step)
        y = np.arange(ymin, ymax+step, step)
        z = np.arange(zmin, zmax+step, step)

        X, Y, Z = np.meshgrid(x, y, z)

        grid = np.zeros((len(x), len(y), len(z), 3)) 
        grid[:,:,:,0] = X.T
        grid[:,:,:,1] = Y.T 
        grid[:,:,:,2] = Z.T
        grid = grid.reshape(len(x) * len(y) * len(z), 3)

        R_vec = grid[:,np.newaxis,:] - coords[np.newaxis,:,:]
        dists = sp.spatial.distance.cdist(grid, coords)
        R_inv = np.divide(1, dists, out=np.zeros_like(dists),
                where=dists>rmin, dtype=np.float32)
        R3_inv = R_inv * R_inv * R_inv
        R5_inv = R3_inv * R_inv * R_inv

        T2 = 3.0 * np.einsum('ij,ijk,ijl->ijkl', R5_inv, R_vec,
                             R_vec, dtype=np.float32, casting='same_kind')
        T2[:,:,0,0] -= R3_inv
        T2[:,:,1,1] -= R3_inv
        T2[:,:,2,2] -= R3_inv

        # get field
        nm_diff = np.abs(wavelength - calc.wavelength_nm)
        ifreq = np.argsort(nm_diff)[0]
        E = np.einsum('ijkl,jl->ik', T2, calc.atomic_dipoles[ifreq][txyz[field_dir]])

#        E[:,0] = (E[:,0].conjugate() * E[:,0]).real
#        E[:,1] = (E[:,1].conjugate() * E[:,1]).real
#        E[:,2] = (E[:,2].conjugate() * E[:,2]).real
#        E[:,txyz[field_dir]] += 1
#        E = E.real
#        E = E.reshape((len(x),len(y),len(z),3))

        E_mag = ( E[:,0].conjugate() * E[:,0] + E[:,1].conjugate() * E[:,1]
                + E[:,2].conjugate() * E[:,2] ).real + 1 
        E_mag = np.log10(E_mag)
        E_mag = E_mag.reshape(X.shape)

        figure = mlab.figure('Field Density Plot')

#        vec = mlab.quiver3d(X, Y, Z, E[:,:,:,0], E[:,:,:,1], E[:,:,:,2], **kwargs)

        pts = mlab.points3d(X, Y, Z, E_mag, **kwargs)

        mlab.axes()
        mlab.show()

def run_from_command_line():
    """Read the output from the command line and plot stuff
    spectrum of fields.

    for useage, see:

    .. code-block:: console

        python -m dimpy.analyzer --help

    """

    # Assume that argparse exists and create an argument parser
    parser = argparse.ArgumentParser(description="Analyze a DIMPy output file.",
                                     prog='DIMPy Analyzer')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    parser.add_argument('-d', '--density', help='Plot the electric field '
                        'density.', action="store_true")
    # read the input file name
    parser.add_argument('file', help='The DIMPy output file to use.')

    args = parser.parse_args()

    # get object
    analyzer = Analyzer(args.file)

    if args.density:
        analyzer.plot_field_density()


if __name__ == '__main__':
    run_from_command_line()
