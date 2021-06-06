#! /usr/bin/env python3

import os

import argparse
import input_reader

from .dimpy_error import DIMPyError
from .calc_method import DDA
from .modified_tensors import PIM
from .nanoparticle import Nanoparticle
from .printer import Output
from .read_input_file import read_input_file
from .timer import Timer
from ._version import __version__

class DIMPy(object):
    """\
    Stores the DIMPy data.
    """

    def __init__(self, input_filename=None, output_filename=None, log_filename=None):
        '''\
        Initializes the DIMPy class.

        :param input_filename: The name of the DIMPy input file (Optional).
        :param output_filename: The name of the DIMPy output file (Optional).
        :param log_filename: The name of the DIMPy log file (Optional).
        '''

        self.input_filename = input_filename
        self.output_filename = output_filename
        self.log_filename = log_filename

        # check if given input file exists
        if input_filename is not None and not os.path.isfile(input_filename):
            raise DIMPyError(f'Input file `{input_filename}` does not exist!')

        # start a timer, even if you're just reading output data
        self._timer = Timer()

        # create the output and log files
        self.out = Output(filename=output_filename)
        self.log = Output(filename=log_filename, logfile=True)

        # debug flag
        self.debug = True


    def read_input(self, input_filename=None):
        '''\
        Reads the input options from a DIMPy input file.

        :param input_filename: The name of the DIMPy input file (Optional).
        '''

        self.log('Reading input file', time=self._timer.startTimer('Prep Input'))

        if input_filename is None:
            input_filename = self.input_filename

        if input_filename is None:
            raise DIMPyError('Must specify an input filename!')
        elif not os.path.isfile(input_filename):
            raise DIMPyError(f'Input file `{input_filename}` does not exist!')

        self._input_options = read_input_file(input_filename)

        # check for pbc
        if self._input_options.pbc is not None:
            pbc = self._input_options.pbc
            pbc_coords = [0, 0, 0]
            for ix in range(3):
                if pbc.groups()[ix+1] is not None:
                    pbc_coords[ix] = float(pbc.groups()[ix+1]) * 1.88973 # to bohr
                else:
                    break
            self.pbc = pbc_coords
        else:
            self.pbc = None


def run_from_command_line():
    """\
    Reads the inputs from the command line and runs the calculation
    """

    # Assume that argparse exists and create an argument parser
    parser = argparse.ArgumentParser(description="Front-end for the DIMPy code.",
                                     prog='DIMPy')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    # read the input file name
    parser.add_argument('file', help='The input file to read.')

    # add optional arguments
    parser.add_argument('-o', '--out', help='Defines the file to output data '
                        'to.  If not given, the input name is used with a '
                        '.out extension.')
    parser.add_argument('-l', '--log', help='Defines the file to log data to. '
                        'If not given, the log file is printed to standard output.')

    args = parser.parse_args()

    # Determine output file name.
    # It has the same name as the input but with .out unless otherwise given
    if args.out is not None:
        output_filename = args.out
    else:
        output_filename = '.'.join([os.path.splitext(input_reader.abs_file_path(
                                   args.file))[0], 'out'])

    ################################
    # Perform the actual calculation
    ################################

    # read input file
    dimpy = DIMPy(input_filename=args.file, output_filename=output_filename)
    dimpy.read_input()

    # read in the nanoparticle data from the file
    nanoparticle = Nanoparticle(args.file, output_filename=output_filename,
                                pbc=dimpy.pbc)

    # set up a calculations method
    if dimpy._input_options.dda:
        dda = DDA(nanoparticle)
        dda.run()
    else:
        dim = PIM(nanoparticle)
        dim.run()

if __name__ == '__main__':
    try:
        run_from_command_line()
    except KeyboardInterrupt:
        exit(1)
