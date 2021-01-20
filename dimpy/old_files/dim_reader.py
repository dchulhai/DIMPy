from input_reader import InputReader, abs_file_path, SUPPRESS
from .constants import EV2HART, NM2HART, HZ2HART, WAVENUM2HART, NOCONV, HART2NM
from os.path import isabs, dirname, join, abspath
from sys import exit
from .printer import error
#from math import pi
import numpy, math

__all__ = ['DIMReader', 'SUPPRESS']

unitmap = {'ev': EV2HART,
           'nm': NM2HART,
           'hz': HZ2HART,
           'cm-1': WAVENUM2HART,
           'hartree': NOCONV,
           'au': NOCONV}


class DIMReader(InputReader):
    def __init__(self, **kwargs):
        """Run the InputReader initializations"""
        super(DIMReader, self).__init__(**kwargs)

    def post_process(self, namespace):
        """Post-process the input to be ready for use by FORTRAN"""

        self.set_print_rules_for_output(namespace)
        self.perform_sanity_checks(namespace)
        self.set_appropriate_numbers_from_input(namespace)
        self.collect_coordinates_and_atoms(namespace)
        self.format_frequency_range(namespace)
        self.format_parameters_for_each_element(namespace)

        # Keep the input file in the namespace
        namespace.add('input_file', self.input_file)

    def set_print_rules_for_output(self, namespace):
        """Sets the print rules to use for the output file"""
        print_rules = set([])
        # The PRINTLEVEL macros set pre-defined printers
        if namespace.printlevel >= 0:
            print_rules.update(['pol', 'timing', 'input'])
        if namespace.printlevel >= 1:
            print_rules.update(['coords', 'energy'])
        if namespace.printlevel >= 2:
            print_rules.add('atmdip')
        if namespace.printlevel >= 3:
            print_rules.add('timingverbose')
            print_rules.discard('timing')

        # Make noprint a set so we can add to it
        if 'noprint' in namespace:
            noprint = set(namespace.noprint)
        else:
            noprint = set([])

        # Add requested keys
        if 'print' in namespace:
            for k in namespace.print:
                print_rules.add(k)
                # TIMING and TIMINGVERBOSE are mutually exclusive
                if k == 'timing':
                    noprint.update(['timingverbose', 'timingveryverbose'])
                if k == 'timingverbose':
                    noprint.update(['timing', 'timingveryverbose'])
                if k == 'timingveryverbose':
                    noprint.update(['timing', 'timingverbose'])

        # If TIMINGVERYVERBOSE is present, choose this over the others
        if 'timingveryverbose' in print_rules:
            noprint.discard('timingveryverbose')
        # If TIMINGVERBOSE is present, choose it over TIMING
        elif 'timingverbose' in print_rules:
            noprint.discard('timingverbose')

        # Remove keys not wanted
        for k in noprint:
            print_rules.discard(k)

        namespace.add('print_rules', print_rules)

    def perform_sanity_checks(self, namespace):
        """Ensure that some variables that cannot appear together don't"""
        if namespace.nopol and namespace.nochar:
            exit('Using both NOPOL and NOCHAR makes no sense')
        if namespace.nopol and namespace.nocross:
            exit('Using both NOPOL and NOCROSS makes no sense')
        if namespace.nocross and namespace.nochar:
            exit('Using both NOCROSS and NOCHAR makes no sense')
        if namespace.totalcharge != 0:
            exit('Nonzero TOTCHARGE not yet implemented')
        if namespace.cpim and namespace.algorithm > 0:
            exit('Iterative solver not implemented for CPIM')
        if namespace.direction and namespace.algorithm == 0:
            exit("Direct solver can't separate field directions")

    def set_appropriate_numbers_from_input(self, namespace):
        """Some values in input can be string or integer.  Fortran will only
        accept the integer version."""

        # Make sure the algorithm is a number.  If it is already a number,
        # an error will be raised and we don't do anything.
        try:
            namespace.algorithm = {'DIRECT': 0,
                                   'BRUTE': 1,
                                   'SINGLE': 2,
                                   'MULTI': 3}[namespace.algorithm.upper()]
        except AttributeError:
            pass

        # Default the algorithm if one was not specified
        if namespace.algorithm == -1:
            if namespace.cpim:
                namespace.algorithm = 0
            else:
                namespace.algorithm = 1

        # Set numbers to the preconditioner
        try:
            namespace.precon = {'NONE': 101.0,
                                'PERFECT': 0.0,
                                'JACOBI': 99.0}[namespace.precon.upper()]
        except AttributeError:
            pass

        # Make sure the grid spacing is a number.  If it is already a number, an
        # error will be raised and we don't do anything.
        try:
            namespace.grid = {'EXTRACOARSE': 3,
                              'COARSE': 5,
                              'MEDIUM': 7,
                              'FINE': 9,
                              'EXTRAFINE': 11}[namespace.grid.upper()]
        except AttributeError:
            pass
        dirs = [False, False, False]
        if namespace.direction:
            for k in namespace.direction:
                try:
                    i = {'X': 0,
                         'Y': 1,
                         'Z': 2}[k.upper()]
                    dirs[i] = True        
                except KeyError:
                    error('Direction must be x, y, and/or z')
                    exit ('Expected [X, Y, Z], got {0}'.format(k))

                    
        else:
            dirs = [True, True, True]
        
        namespace.add('dirs', dirs)

    def collect_coordinates_and_atoms(self, namespace):
        """Collect the coordinates from the input, either directly or from other file"""

        # Collect from a file
        if 'file' in namespace.xyz:
            # Make sure the xyz file path absolute
            if isabs(namespace.xyz.file.group(1)):
                fname = namespace.xyz.file.group(1)
            else:
                if isinstance(self.filename, str):
                    fname = dirname(self.filename)
                    fname = join(fname, namespace.xyz.file.group(1))
                else:
                    fname = abspath(namespace.xyz.file.group(1))
            try:
                with open(fname) as fl:
                    xyzfile = [line.strip() for line in fl]
            except IOError:
                exit('Error reading file {0}'.format(fname))
            else:
                try:
                    coords = [[float(x.split()[1]),
                               float(x.split()[2]),
                               float(x.split()[3])] for x in xyzfile[2:]]
                    names = [x.split()[0].capitalize() for x in xyzfile[2:]]
                except IndexError:
                    msg = 'Bad coordinate specification in file {0}.'
                    exit(msg.format(fname))
                try:
                    natoms = int(xyzfile[0])
                except ValueError:
                    msg = 'Unable to read number of atoms in file {0}.'
                    exit(msg.format(fname))
                if len(names) != natoms:
                    error('Number of atoms found does not match expected')
                    exit('Expected {0}, found {1}'.format(natoms, len(names)))

        # Otherwise, collect from the block defined in the input
        else:
            names = [m.group(1).capitalize() for m in namespace.xyz.coords]
            coords = [[float(m.group(2)), float(m.group(3)), float(m.group(4))]
                      for m in namespace.xyz.coords]
            natoms = len(names)
            if 'natoms' in namespace.xyz:
                namenatoms = int(namespace.xyz.natoms.group(1))
                if natoms != namenatoms:
                    error('Number of atoms found does not match expected')
                    exit('Expected {0}, found {1}'.format(natoms, namenatoms))
        from numpy import array, argsort

        # Add the coordinates to the namespace
        namespace.add('natoms', natoms)
        namespace.add('atoms', names)
        namespace.add('coords', [[namespace.distconv(x) for x in c] for c in coords])
        namespace.add('elements', set(names))

    def format_frequency_range(self, namespace):
        """Format the frequency range correctly"""

        freqs = []

        # Convert each given frequency
        if 'frequency' in namespace:
            conv = unitmap[namespace.frequency[0]]
            for freq in namespace.frequency[1:]:
                freqs.append(conv(freq))

        # Expand the frequency range and convert appropriately
        elif 'freqrange' in namespace:
            conv = unitmap[namespace.freqrange[0]]
            if namespace.freqrange[3] == 1:
                frequencies = [float(namespace.freqrange[1])]
            else:
                start, stop, num = namespace.freqrange[1:]
                step = (stop - start) / float(num - 1)
                frequencies = [y * step + start for y in range(num)]
                frequencies[-1] = stop
            for freq in frequencies:
                freqs.append(conv(freq))

        # No frequencies for static
        else:
            pass

        # Sort the frequencies in ascending order
        namespace.add('freqs', sorted(freqs))
        namespace.add('nfreq', len(freqs))



    def calculate_static_polarizability(self, elem, namespace, par, keyelem, keyrmin, keyrmax):
        """Add a special static polarizability term"""
        import numpy as np

        def find_CN(rmn,rmin,rmax):
            if rmn < rmin:
                return 1
            elif rmn > rmax:
                return 0
            elif rmin <= rmn <= rmax:
                return 0.5*(1 + numpy.cos(numpy.pi*((rmn - rmin)/(rmax - rmin))))

        if namespace.cpim:
                par.add('static_pol', [par.pol, par.cap])
        elif par.coorddepend==True:
            coord = numpy.array(namespace.coords, dtype='float64')
            # save the index of coordination dependent atom in dependindex
            n=-1
            dependindex=[]
            for atomname in namespace.atoms:
                n = n + 1
                if atomname == elem:
                    dependindex.append(n)
            #print("dependindex: {0}".format(dependindex))

            # save the index of metal atom in metalindex
            metal = ['Na', 'Mg', 'Al', 'Si', 'Ar', 'K',  'Ca', 'Sc','Ti', 'V',  'Cr',
                    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zi', 'Ga', 'Ge','As', 'Se', 'Br',
                    'Kr', 'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc','Ru', 'Rh', 'Pd',
                    'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La','Ce',
                    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb','Dy', 'Ho', 'Er', 'Tm',
                    'Yb', 'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr','Ra', 'Ac', 'Th', 'Pa', 'U']
            n=-1
            metalindex=[]
            for atomname in namespace.atoms:
                n = n + 1
                if atomname in metal:
                    metalindex.append(n)
            #print("metalindex: {0}".format(metalindex))

            # calculate the coordination number between coordination dependent atom and metal atom
            dependcn=[]
            for i in range(len(dependindex)):
                CN = 0
                for j in range(len(metalindex)):
                    rv = coord[dependindex[i]]-coord[metalindex[j]]
                    rv = numpy.sqrt(numpy.dot(rv, rv))
                    CN += find_CN(rv, par.rmin, par.rmax)
                dependcn.append(CN)


            # save the index of the atom binding to the coordination dependent atom in keyindex
            keycn=[]
            g=[]
            if len(keyelem) != 0:
                n=-1
                keyindex = []
                romin = []
                romax = []
                for atomname in namespace.atoms:
                    n = n + 1
                    for i in range(len(keyelem)):
                        if atomname == keyelem[i]:
                            keyindex.append(n)
                            romin.append(keyrmin[i])
                            romax.append(keyrmax[i])

                for i in range(len(dependindex)):
                    CN = 0
                    for j in range(len(keyindex)):
                        rv = coord[dependindex[i]] - coord[keyindex[j]]
                        rv = numpy.sqrt(numpy.dot(rv, rv))
                        CN += find_CN(rv, romin[j], romax[j])
                    keycn.append(CN)
                    f = 1.0 - CN/namespace.gfactor
                    g.append(f)

            else:
                for i in range(len(dependcn)):
                    keycn.append(0.0)
                    g.append(1.0)

            # total coordination number of coordination dependent atom
            cn=[]
            for i in range(len(dependcn)):
                cn.append(keycn[i]+dependcn[i])

            # calculate coordination dependent parameters
            r=[]
            for i in range(len(cn)):
                a = par.rsurf + (( par.rbulk - par.rsurf )*( min( par.cnmax,cn[i] )/par.cnmax ))
                r.append(a)

            p = [( 6/numpy.pi ) * x**3 for x in r]
            par.add('cd_static_pol', p)
            par.add('cd_rad', r)

            d = [(( x - par.rbulk )/( par.rsurf - par.rbulk )) for x in r]
            # Plasmon parameter
            conv = unitmap['ev']
            a=[]
            aa=[]
            b=[]
            for x in d:
                aa.append((( 1 - x )*conv(par.drude[0])) +( x*(conv(par.wpin))))
                b.append((( 1 - x )*conv(par.drude[1])) +( x*(conv(par.sc))))
            for i in range(len(aa)):
                a.append(aa[i]*g[i])
#
            par.add('cd_drude',list(zip(a,b)))
            #print("par.cd_drude {0}".format(par.cd_drude))
        else:
            rad = par.rad + (par.spillout / pow(namespace.natoms, 1 / 3))
            par.add('static_pol', (6 / math.pi) * rad ** 3)

       
    def format_parameters_for_each_element(self, namespace):
        """Make sure that the parameters for each element are formatted correctly"""

        keyelem = []
        keyrmin = []
        keyrmax = []
        for elem in namespace.elements:
            par = self.get_parameters_for_element(elem, namespace)
            if par.binding == True:
                keyelem.append(elem)
                keyrmin.append(par.romin)
                keyrmax.append(par.romax)

        for elem in namespace.elements:
            par = self.get_parameters_for_element(elem, namespace)
            self.check_only_valid_parameters_appear_for_element(elem, namespace, par)
            self.check_that_valid_parameters_are_defined(elem, namespace, par)
            self.collect_experimental_dielectrics(namespace, par)
            self.zero_parameters_that_are_not_needed(par)
            par.rad = namespace.distconv(par.rad)  # Make sure radius is in Bohr
            par.rmin = namespace.distconv(par.rmin)
            par.rmax = namespace.distconv(par.rmax)
            par.rsurf = namespace.distconv(par.rsurf)
            par.rbulk = namespace.distconv(par.rbulk)
            self.calculate_static_polarizability(elem, namespace, par, keyelem, keyrmin, keyrmax)
#
            self.collect_drude_term(par)
            self.collect_and_format_lorentzian_terms(par)



    def get_parameters_for_element(self, elem, namespace):
        """Get the parameters for the given element. If it doesn't exist, quit."""
        if elem.lower() not in namespace:
            exit('Missing type information for element ' + elem)
        else:
            par = getattr(namespace, elem.lower())
        return par
    
    def zero_parameters_that_are_not_needed(self, par):
        """Give zero value to parameters that weren't needed"""
        for p in ('pol', 'cap', 'om1', 'om2', 'gm1', 'gm2', 'size'):
            if p not in par:
                par.add(p, 0.0)

    def collect_experimental_dielectrics(self, namespace, par):
        """Collect the experimental dielectric, or return all zeros if none"""
        try:
            par.exp = self.expdie(par.exp, namespace.freqs)
        except AttributeError:
            par.exp = self.expdie('', namespace.freqs)

    def loadtxt(self, filename):
        """Loads text file as a list of floats., skipping comments
        Each column of data is a row in the returned data list."""
        with open(filename) as fl:
            f = [x.strip().split() for x in fl if x[0] != '#']
            # Now, reorganize from row x col ==> col x row
        return [[float(nums[n]) for nums in f] for n in range(len(f[0]))]

    def expdie(self, name, freqs):
        """Retrieves the experimental dielectric parameters for this type.
        If none were found, return complex zero's for each frequency"""
        from os.path import join, dirname, realpath
        from .spline import spline, interpolate

        die = []
        if name:
            try:
                expdata = self.loadtxt(name)
            except IOError:
                # If not found as given, look in the DIM library
                dn = dirname(realpath(__file__))
                try:
                    p = join(dn, 'dielectrics', name)
                    expdata = self.loadtxt(p)
                except IOError:
                    exit('Cannot find dielectric file for ' + name)

            # Spline the data
            realknots = spline(expdata[0], expdata[1])
            imagknots = spline(expdata[0], expdata[2])
            # For each frequency, interpolate
            for om in freqs:
                real = interpolate(expdata[0], expdata[1], realknots, HART2NM(om))
                imag = interpolate(expdata[0], expdata[2], imagknots, HART2NM(om))
                die.append(complex(real, imag))
        else:
            die = [complex(0.0) for i in range(len(freqs))]

        return die

    def check_only_valid_parameters_appear_for_element(self, elem, namespace, par):
        """Make sure that only valid parameters appear for the given element."""

        invalid = {
            'CPIM': ['lrtz', 'lrtz1', 'lrtz2', 'lrtz3', 'drude', 'exp'],
            'DIE': ['pol', 'cap', 'om1', 'om2', 'gm1', 'gm2', 'size'],
            'POL': ['drude', 'lrtz1', 'lrtz2', 'lrtz3', 'exp']}
        invalid['POL'] += invalid['DIE']

        msg = '{0} is not a valid parameter {1} ({2})'
        if namespace.cpim:
            for p in invalid['CPIM']:
                if p in par:
                    exit(msg.format(p, 'for CPIM', elem))
            if par.dielectric:
                exit('Dielectric parameters not valid for CPIM')
            else:
                par.polarizability = True
        else:
            # Default to dielectric if polarizability is not given
            par.dielectric = not par.polarizability
            t = 'dielectric' if par.dielectric else 'polarizability'
            k = 'DIE' if par.dielectric else 'POL'
            for p in invalid[k]:
                if p in par:
                    exit(msg.format(p, 'for a {0} for PIM'.format(t), elem))

    def check_that_valid_parameters_are_defined(self, elem, namespace, par):
        """Ensure that valid parameters for the elements are defined (if required).
        If they are not defined but a default is implemented, choose the default."""

        # These experimental dielectric values have defaults
        default_experimental_dielectrics_defined = set(
            ['Ag', 'Al', 'Au', 'Be', 'Co', 'Cr', 'Cu', 'Ir', 'K', 'Li', 'Mo', 'Na',
             'Nb', 'Ni', 'Os', 'Pd', 'Pt', 'Rh', 'Si', 'Ta', 'V', 'W', ])
        valid_parameters = {
            'CPIM': {'NOFREQ': ('pol', 'cap',),
                     'FREQ': ('pol', 'cap', 'om1', 'om2', 'gm1', 'gm2', 'size')},
            'PIM': {'NOFREQ': tuple(),
                    'FREQ': ('exp',) if par.dielectric else ('lrtz',)}}
        cpim_defaults = {
            'Ag': {'pol': 49.9843, 'cap': 2.7529, 'om1': 0.0747, 'om2': 0.0545,
                   'gm1': 0.0604, 'gm2': 0.0261, 'size': 2.7759, },
            'Au': {'pol': 39.5297, 'cap': 1.2159, }}

        typ = 'CPIM' if namespace.cpim else 'PIM'
        ftyp = 'FREQ' if namespace.nfreq else 'NOFREQ'

        # Loop over the valid parameters, assigning defaults if needed
        msg = 'No default {0} for {1} found for element {2}'
        for p in valid_parameters[typ][ftyp]:
            if p not in par:
                # Default CPIM parameters if not given
                if namespace.cpim:
                    try:
                        value = cpim_defaults[elem][p]
                    except KeyError:
                        exit(msg.format('value', p, elem))

                # Default PIM parameter if not given
                
                    # Experimental dielectric default
                    if (par.dielectric and
                        'drude' not in par and 'lrtz'  not in par and
                        'lrtz1' not in par and 'lrtz2' not in par and
                        'lrtz3' not in par):
                        if elem in default_experimental_dielectrics_defined:
                            value = elem
                        else:
                            exit(msg.format('dielectric', p, elem))

                    # Polarizability needs some lorentzians
                    elif not par.dielectric and 'lrtz' not in par:
                        exit(msg.format('polarizability', p, elem))

                    # No default, but parameter not required
                    else:
                        continue

                # Add the default to the parameters since it does not exist
                par.add(p, value)

    def collect_drude_term(self, par):
        """Puts the drude terms in the correct format, returning -1 if not given"""
        if 'drude' in par:
            conv = unitmap[par.drude[2]]
            #print(par.drude)
            par.drude = [conv(par.drude[0]), conv(par.drude[1])]
        else:
            par.add('drude', [-1.0, -1.0])
      
    def collect_and_format_lorentzian_terms(self, par):
        """Convert Lorentzian units and make them unified.  This is different
        for polarizability and dielectric parameters types."""

        def extract_data_for_lrtz(lrtz_type, counter, par, osc, om, res, pls, unit):
            for l in getattr(par, lrtz_type):
                c = unitmap[l[unit]]
                o = l[osc] if osc >= 0 else 1.0
                w = c(l[om]) if om >= 0 else 1.0
                try:
                    par.lrtz[counter] = [o, w, c(l[res]), c(l[pls])]
                except IndexError:
                    exit('The maximum number of lorentzians is 50')
                counter += 1
            par.remove(lrtz_type)
            return counter

        # First, convert lrtz to lrtz1
        if 'lrtz' in par:
            if 'q+=1lrtz1' not in par:
                par.add('lrtz1', [])
            else:
                par.lrtz1 = list(par.lrtz1)
            for l in par.lrtz:
                par.lrtz1.append(l)
            par.remove('lrtz')
            # Init lrtz to -1 for all values
        par.add('lrtz', [[-1.0, -1.0, -1.0, -1.0] for i in range(50)])

        counter = 0
        # Dielectric
        if par.dielectric:
            # lrtz1 uses res for res and pls
            if 'lrtz1' in par:
                counter = extract_data_for_lrtz('lrtz1', counter, par, 0, 1, 1, 2, 3)

            # lrtz2 uses all parameters
            if 'lrtz2' in par:
                counter = extract_data_for_lrtz('lrtz2', counter, par, 0, 1, 2, 3, 4)

            # lrtz3 has no osc
            if 'lrtz3' in par:
                extract_data_for_lrtz('lrtz3', counter, par, -1, 0, 1, 2, 3)

        # For polarizability
        else:
            # polarizability lorentzian has no om
            if 'lrtz1' in par:
                extract_data_for_lrtz('lrtz1', counter, par, 0, -1, 1, 2, 3)
