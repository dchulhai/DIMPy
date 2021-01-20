from ._version import __version__
from datetime import datetime
import sys
import os
import numpy as np

class Output(object):
    """Class for riting to a file or stdout. It is called like a function.
    """

    def __init__(self, filename=None, logfile=False):
        """Opens the file to be written."""
        self.filename = filename
        self.logfile = logfile

        if filename is not None:
            try:
                self.file = open(self.filename, 'w')
            except IOError:
                sys.exit(f'Error opening file {filename}')
        else:
            self.file = sys.stdout

    def __call__(self, *args, **kwargs):
        """Writes to file and adds a space to the front of the message.
        The space can be ommited with the nospace keyword.
        Takes the same keyword arguments as print except file."""
        nospace = kwargs.pop('nospace', False)

        # If this is the logfile, then add a timestamp to the front
        # of the output.
        if self.logfile:
            time = kwargs.pop('time', datetime.now())
            if nospace:
                print(time.strftime('%c'), *args, **kwargs)
            else:
                print('', time.strftime('%c'), *args, **kwargs)

        # This is a regular output file
        else: 
            if nospace:
                print(*args, file=self.file, **kwargs)
            else:
                print('', *args, file=self.file, **kwargs)

    def __del__(self):
        """Close the file if it is not already closed"""
        if self.file is sys.stdout:
            pass # do nothing
        else:
            self.file.close()

def print_welcome(output=print):
    """Prints the welcome on top of the output file."""

    import socket

    # title and authors
    title = ' D I S C R E T E   I N T E R A C T I O N   M O D E L   F O R'\
            '   P Y T H O N '
    dimpy_authors = 'Dhabih V. Chulhai'
    dim_authors = (' Lasse Jensen \n'
                   ' LinLin Jensen \n'
                   ' Justin E. Moore \n'
                   ' Seth M. Morton ')
    citations = ('cite this')

    # runtime conditions
    now = datetime.now()
    start_date = now.strftime('%A, %b %d, %Y')
    start_time = now.strftime('%H:%M:%S hrs')
    hostname = socket.gethostname()
    threads = os.environ['OMP_NUM_THREADS']

    # print the title of the program
    output('*'*79)
    output(title.center(79, '*'))
    output('*'*79)
    output()

    # print dimpy authors
    print_header('DIMPy Authors', output=output)
    for author in dimpy_authors.split('\n'):
        output(author.center(79))
    output()

    # print dim authors
    print_header('DIM Authors', output=output)
    for author in dim_authors.split('\n'):
        output(author.center(79))
    output()

    # print runtime conditions
    print_header('Runtime Conditions', output=output)
    output((f'Version        : {__version__}').ljust(40).center(79))
    output((f'Start date     : {start_date}').ljust(40).center(79))
    output((f'Start time     : {start_time}').ljust(40).center(79))
    output((f'Hostname       : {hostname}').ljust(40).center(79))
    output((f'# Threads      : {threads}').ljust(40).center(79))
    output()

    # print out a warning
    output('ALL UNITS ARE REPORTED IN ATOMIC'
           ' UNITS UNLESS OTHERWISE SPECIFIED'.center(79))
    output()
    

def print_header(header, output=print):
    length = len(header)
    output(('='*length).center(79))
    output(header.center(79))
    output(('='*length).center(79))
    output()

def print_atomic_dipoles(xyz, atoms, mu, output=print):
    """Prints the atomic dipoles to the `Output` object."""

    print_header(f'Induced dipoles for each DIM atom : {xyz} field direction',
                 output=output)

    output('        Atom                Dipole')
    output('                  X           Y           Z')
    output('-------------------------------------------------')

    for i in range(len(atoms)):
        indx = i+1
        atom = atoms[i]
        if np.isrealobj(mu):
            mux = mu[i,0]
            muy = mu[i,1]
            muz = mu[i,2]
            output(f'{indx:>7d} {atom:>4s} {mux:11.5f} {muy:11.5f} {muz:11.5f}')
        else:
            mux = mu[i,0].real
            muy = mu[i,1].real
            muz = mu[i,2].real
            output(f'{indx:>7d} {atom:>4s} {mux:11.5f} {muy:11.5f} {muz:11.5f} (R)')
            mux = mu[i,0].imag
            muy = mu[i,1].imag
            muz = mu[i,2].imag
            output(f'             {mux:11.5f} {muy:11.5f} {muz:11.5f} (I)')

    output('')

def print_energy(energy, output=print):
    ex, ey, ez = energy[:]
    print_header("Energy Terms For Each Field Direction", output)
    output('                       X               Y               Z')
    output('---------------------------------------------------------------------')
    output(f' Total    : {ex:17.7f} {ey:17.7f} {ez:17.7f}')
    output('---------------------------------------------------------------------')
    output()

def print_polarizability(pol, output=print):

    def print_pol(pol, output=print):
        output('                    X                   Y                   Z')
        output()
        XYZ = ['X', 'Y', 'Z']
        for i in range(3):
            px, py, pz = pol[i,:]
            xyz = XYZ[i]
            output(f'    {xyz} {px:19.3f} {py:19.3f} {pz:19.3f}')
        output()

    print_header('Polarizability Tensor', output)

    if np.isrealobj(pol):
        print_pol(pol, output)
    else:
        output('Real'.center(79))
        print_pol(pol.real, output)
        output('Imaginary'.center(79))
        print_pol(pol.imag, output)

    # isotropic polarizability
    iso_pol = np.trace(pol) / 3.0
    if np.isrealobj(iso_pol):
        output(f'Isotropic Polarizability = {iso_pol:18.2f}')
    else:
        output(f'Isotropic Polarizability = {iso_pol.real:18.2f} +'
               f' {iso_pol.imag:16.2f}i')

    # anisotropic polarizability
    def anisotropic(pol):
        ani_pol = np.sqrt( ( np.abs( pol[0,0] - pol[1,1] )**2
                           + np.abs( pol[0,0] - pol[2,2] )**2
                           + np.abs( pol[1,1] - pol[2,2] )**2
                           + 1.5 * ( np.abs( pol[0,1] + pol[1,0] )**2
                                   + np.abs( pol[1,2] + pol[2,1] )**2
                                   + np.abs( pol[2,0] + pol[0,2] )**2
                                   )
                            ) * 0.5 )
        return ani_pol

    ani_pol = anisotropic(pol.real) + anisotropic(pol.imag)
    output(f'Anisotropic Polarizability = {ani_pol:16.2f}')
    output()

def print_efficiencies(qAbs, qScat, qExt, cAbs, cScat, cExt, output=print):

    print_header('Efficiencies and Cross-Sections', output)
    output('        qAbs       qScat        qExt        cAbs '
           '      cScat        cExt')
    output(f'    {qAbs:11.4e} {qScat:11.4e} {qExt:11.4e} {cAbs:11.4e}'
           f' {cScat:11.4e} {cExt:11.4e}')
    output()
