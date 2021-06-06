import os

import numpy as np

from .calc_method import CalcMethod
from .dimpy_error import DIMPyError
from .nanoparticle import Nanoparticle
from .printer import Output

class Analyzer(CalcMethod):
    """\
    Reads and analyzes the information from a DIMPy output file.
    """

    def __init__(self, filename, log_filename=None):
        '''Initializes the analyzer object given a filename.
        '''


        # create log file
        self.log = Output(log_filename, logfile=True)

        # check if this is an input or output file
        if filename[-4:] == '.dim':
            self.input_filename = filename
            self.output_filename = filename[:-4] + '.out'
            if not os.path.isfile(self.output_filename):
                self.log('Corresponding output file not found!')
                return
        elif filename[-4:] == '.out':
            self.output_filename = filename
            self.input_filename = None
        else:
            self.log('This is not a recognized DIMPy input or output file!')
            return

        # get the various indices in the output file
        self._collect_output()


    def _collect_output(self, output_filename=None):
        '''Reads the keys from the output file.
        '''

        # Collect all data into memory for data retention
        if output_filename is None: output_filename = self.output_filename
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
                #############
                # Nanoparticle Input
                #############
    
                # End of the input block
                '                                Nanoparticle Input':
                                                           ['NANOPARTICLE', 3],
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
  
        # collect information about the nanoparticle input file
        if 'NANOPARTICLE' in indices:

            # collect the nanoparticle object
            input_filename = f[indices['NANOPARTICLE']].split()[3]
            self.input_filename = input_filename
            self.nanoparticle = Nanoparticle(self.input_filename)

        else:
            raise DIMPyError("Something went wrong. Perhaps this is not a "
                             "DIMPy output?")
 
        # Read frequencies
        if 'FREQUENCY' in indices:
            freqs = []
            for ifreq in indices['FREQUENCY']:
                freqs.append(f[ifreq].split()[1])
            self.freqs = np.array(freqs, dtype=float)
            self.nFreqs = len(self.freqs)

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
                self.polarizabilities = np.array(pols)

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

            self.qAbsorb  = np.array(qAbs)
            self.qScatter = np.array(qScat)
            self.qExtinct = np.array(qExt)
            self.cAbsorb  = np.array(cAbs)
            self.cScatter = np.array(cScat)
            self.cExtinct = np.array(cExt)

        # collect atomic dipoles
        if 'PIMDIP' in indices:

            atomic_dipoles = []

            for idip in indices['PIMDIP']:

                # Collect complex dipoles
                if '(R)' in f[idip]:
                    temp = np.zeros((self.nanoparticle.natoms, 3),
                                    dtype=complex)
                    for i in range(self.nanoparticle.natoms):
                         temp[i] = np.array(f[idip+i*2].split()[2:5],
                                            dtype=float)
                         temp[i] += np.array(f[idip+i*2+1].split()[:3],
                                             dtype=float) * 1j

                # Collect real dipoles
                else:
                    temp = np.zeros((self.nanoparticle.natoms, 3), dtype=float)
                    for i in range(self.nanoparticle.natoms):
                        temp[i] = np.array(f[idip+i].split()[2:5], dtype=float)

                atomic_dipoles.append(temp)

            self.atomic_dipoles = np.array(atomic_dipoles)
            self.atomic_dipoles = self.atomic_dipoles.reshape(self.nFreqs,
                3, self.nanoparticle.natoms, 3)
