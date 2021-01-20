from numpy import array, append, argsort, asarray
from datetime import datetime, timedelta
from .constants import BOHR2ANGSTROM
from os.path import isabs, join, dirname
import sys


def intround(val):
    '''Return an int from an arbitrary type, or None if not possible.'''
    try:  # Easy conversion (floats)
        return int(round(val))
    except TypeError:  # Complex or string.
        try:
            return int(round(float(val)))
        except TypeError:  # Complex number
            try:
                return int(round(abs(val)))
            except TypeError:  # Complex string
                try:
                    return int(round(abs(complex(val))))
                except TypeError:
                    return None


def collect_dim(self, f, indices):
    '''Collect all DIM properties.'''

    ########################
    # COLLECT DIM CORDINATES
    ########################

    if 'COORDS' in indices:
        s = indices['COORDS']
        e = next(i for i, x in enumerate(f[s:], s) if not x.strip())
        # Elements 2, 3 and 4 are the X, Y, and Z coordinates
        self.coordinates = array([c.split()[2:5] for c in f[s:e]],
                                 dtype=float)
        self.atoms = array([a.split()[1] for a in f[s:e]])
        self.elements = set(self.atoms)
        self.nelements = len(self.elements)
        self.natoms = len(self.atoms)
        self.coordinates = BOHR2ANGSTROM(self.coordinates)
    # Try to get this info from the coordinates
    elif 'XYZ' in self.key:
        if len(self.key['XYZ']) == 1:
            # Make sure the xyz file path absolute
            if isabs(self.key['XYZ'][0]):
                fname = self.key['XYZ'][0]
            else:
                fname = dirname(self.filename)
                fname = join(fname, self.key['XYZ'][0])
            with open(fname) as fl:
                temp = [l.strip() for l in fl][2:]
        else:
            temp = [l for l in self.key['XYZ'] if len(l.split()) > 1]
        self.coordinates = array([c.split()[1:4] for c in temp],
                                 dtype=float)
        self.atoms = array([a.split()[0] for a in temp])
        self.elements = set(self.atoms)
        self.nelements = len(self.elements)
        self.natoms = len(self.atoms)
        if 'BOHR' in self.key:
            self.coordinates = BOHR2ANGSTROM(self.coordinates)
    # Try to find the number of atoms in the header
    elif 'RUNTIME' in indices:
            ix = indices['RUNTIME'] + 13
            self.natoms = int(f[ix].split()[3])

    ###################################
    # DETERMINE CALCULATION FREQUENCIES
    ###################################

    # Define the search string
    if 'FREQUENCY' in indices:
        ar = indices['FREQUENCY']
        self.e_frequencies = array([], dtype=float)
        for ix in ar:
            # Collect the frequency
            ln = f[ix].split()
            self.e_frequencies = append(self.e_frequencies, float(ln[1]))
        # Store the number of frequencies
        self.npol = len(self.e_frequencies)
    else:
        self._raise_or_pass('Error locating DIM frequencies')

    #########################################
    # COLLECT DIM INDUCED CHARGES AND DIPOLES
    #########################################

    if 'CPIMDIP' in indices:
        start = indices['CPIMDIP']
    elif 'PIMDIP' in indices:
        start = indices['PIMDIP']
    else:
        start = []

    if start:
        self.dipoles = {}
        if 'CPIM' in self.subkey:
            self.charges = {}

        # Locate all locations of the induced charges and dipoles
        for st in start:

            # Start and stop limits
            s = st + 3
            e = next(i for i, x in enumerate(f[s:], s) if not x.strip())

            # Complex.
            if 'FD' in self.calctype:
                # Elements 2, 3, and 4 on every other line are the real dipoles
                r = array([x.split()[2:5] for x in f[s:e-1:2]], dtype=float)
                # Elements 0, 1, and 2 on every other line are the imag dipoles
                i = array([x.split()[0:3] for x in f[s+1:e:2]], dtype=float)
                dpls = r + i*1j
                # Charge not printed for PIM and DRF.
                if 'CPIM' in self.subkey:
                    # Element 5 on every other line is the real charge.
                    r = array([x.split()[5] for x in f[s:e-1:2]], dtype=float)
                    # Element 3 on every other line is the imag charge.
                    i = array([x.split()[3] for x in f[s+1:e:2]], dtype=float)
                    chrgs = r + i*1j

                # Assign to proper location based on description after header
                head = 'FD scattered'
                # Store the different direction sequentially for each frequency
                dir = {'X': 0, 'Y': 1, 'Z': 2}[f[st-3].split()[-3]]
                try:
                    self.dipoles[head][dir].append(dpls)
                    if 'CPIM' in self.subkey:
                        self.charges[head][dir].append(chrgs)
                except KeyError:
                    self.dipoles[head] = [[], [], []]
                    self.dipoles[head][dir].append(dpls)
                    if 'CPIM' in self.subkey:
                        self.charges[head] = [[], [], []]
                        self.charges[head][dir].append(chrgs)

            # Real
            else:
                dpls = array([x.split()[2:5] for x in f[s:e]], dtype=float)
                if 'CPIM' in self.subkey:
                    chrgs = array([x.split()[5] for x in f[s:e]], dtype=float)

                # Assign to proper location based on description after header
                head = 'static scattered'
                # Store the different direction sequentially for each frequency
                dir = {'X': 0, 'Y': 1, 'Z': 2}[f[st-3].split()[-3]]

                try:
                    self.dipoles[head][dir] = dpls
                    if 'CPIM' in self.subkey:
                        self.charges[head][dir] = chrgs
                except KeyError:
                    self.dipoles[head] = [None, None, None]
                    self.dipoles[head][dir] = dpls
                    if 'CPIM' in self.subkey:
                        self.charges[head] = [None, None, None]
                        self.charges[head][dir] = chrgs

        # Reorder so that it goes head:freq:dir:atom(:xyz) instead
        # of head:dir:freq:atom(:xyz)
        for head in self.dipoles:
            if head == 'static scattered':
                self.dipoles[head] = array(self.dipoles[head])
                if 'CPIM' in self.subkey:
                    self.charges[head] = array(self.charges[head])
            else:
                self.dipoles[head] = array(
                    self.dipoles[head]).swapaxes(0, 1)
                if 'CPIM' in self.subkey:
                    self.charges[head] = array(
                        self.charges[head]).swapaxes(0, 1)

    # If this table was not found but we were supposed to, alert.
    else:
        if 'ATMDIP' in self.subkey:
            self._raise_or_pass('Could not find DIM atomic dipoles.')
        else:
            pass

    ####################################
    # COLLECT DIM POLARIZABILITY TENSORS
    ####################################

    # The polarizability tensor for the DIM system is collected.
    # It can be real or complex, depending on the calculation.
    if 'POL' in indices:
        ar = indices['POL']
        for ix in ar:
            if 'FD' in self.calctype:
                # Collect complex isolated DIM tensor
                s = ix + 5
                e = ix + 8
                r = array([[x.split()[1:4] for x in f[s:e]]], dtype=float)
                s = ix + 11
                e = ix + 14
                i = array([[x.split()[1:4] for x in f[s:e]]], dtype=float)
                try:
                    self.polarizability = append(self.polarizability, r+i*1j, axis=0)
                except ValueError:
                    self.polarizability = r+i*1j
            else:
                # Collect real isolated DIM tensor
                s = ix + 4
                e = ix + 7
                r = array([[x.split()[1:4] for x in f[s:e]]], dtype=float)
                try:
                    self.polarizability = append(self.polarizability, r, axis=0)
                except ValueError:
                    self.polarizability = r

    # If this table was not found but we were supposed to, alert.
    else:
        if 'POLARIZABILITY' in self.calctype:
            self._raise_or_pass('Could not find DIM polarizabilities')
        else:
            pass

    #########################################
    # COLLECT EFFICIENCIES AND CROSS SECTIONS
    #########################################

    if 'EFF' in indices:
        ar = indices['EFF']
        for ix in ar:
            ln = f[ix].split()
            try:
                self.efficiencies = append(self.efficiencies,
                                           ln[0:3], axis=0)
                self.cross_sections = append(self.cross_sections,
                                             ln[3:6], axis=0)
            except ValueError:
                self.efficiencies = array(ln[0:3], dtype=float)
                self.cross_sections = array(ln[3:6], dtype=float)
        self.efficiencies = asarray(self.efficiencies, dtype=float)
        self.cross_sections = asarray(self.cross_sections, dtype=float)

    # Put the frequencies in ascending order
    indx = argsort(self.e_frequencies)
    # Sort the other properties accordingly
    if self.dipoles is not None:
        for head in self.dipoles:
            if head is not 'static scattered':
                self.dipoles[head] = self.dipoles[head][indx]
    if self.charges is not None:
        for head in self.charges:
            self.charges[head] = self.charges[head][indx]
    if self.polarizability is not None:
        self.polarizability = self.polarizability[indx]
    if self.efficiencies is not None:
        self.efficiencies = self.efficiencies[indx]
    if self.cross_sections is not None:
        self.cross_sections = self.cross_sections[indx]


def collect_timing(self, f, indices):
    '''Collect the timing info.'''

    # Collect the starting time
    if 'RUNTIME' in indices:
        ix = indices['RUNTIME']
        tp = ' '.join(f[ix+1].strip().split()[3:])  # The date
        tp += ' ' + f[ix+2].strip().split()[3]    # Add time to date
        self.start = datetime.strptime(tp, '%A, %b %d, %Y %X')
    else:
        self._raise_or_pass('Could not find DIM runtime conditions')

    # Collect the timings
    if 'TIMING' in indices:
        ix = indices['TIMING']
        self.real_time = timedelta(seconds=intround(f[ix].split()[0]))
        self.cpu_time = timedelta(seconds=intround(f[ix].split()[1]))
        self.routine_times = {}
        for line in f[ix+1:]:
            if not line.strip():
                continue
            ln = line.split()
            # If the routine is multiple words, join the words
            tp = ' '.join(ln[3:])
            # 0: Real time, 1: CPU time, 2: # Calls
            if tp not in self.routine_times:
                self.routine_times[tp] = (timedelta(seconds=intround(ln[0])),
                                          timedelta(seconds=intround(ln[1])),
                                          int(ln[2]))
            else:
                # Add the new time to the old time
                self.routine_times[tp] = list(self.routine_times[tp])
                self.routine_times[tp][0] += timedelta(seconds=intround(ln[0]))
                self.routine_times[tp][1] += timedelta(seconds=intround(ln[1]))
                self.routine_times[tp][2] += int(ln[2])
                self.routine_times[tp] = tuple(self.routine_times[tp])


def collect_technical(self, f, indices):
    '''Collect technical info, such as where the job was run and termination'''

    # Look above the timing for an error message
    if 'TIMING' in indices:
        ix = indices['TIMING']
        try:
            # Look up to 50 lines before the timing for an error
            ix = next(i for i in range(ix, ix-50, -1) if 'ERROR' in f[i])
        except StopIteration:
            # If no error was found then we terminated correctly
            self.termination = 'NORMAL TERMINATION'
        else:
            # An error was found, save it
            self.termination = f[ix].strip().replace('ERROR: ', '')

    # Find the number of processor
    if 'RUNTIME' in indices:
        ix = indices['RUNTIME']

        # Get the host name
        self.host = f[ix+3].split()[3]
        # Get the number of processors
        self.nprocs = int(f[ix+4].split()[3])
