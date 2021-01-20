#! /usr/bin/env python3

from ._version import __version__
from os import path

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

        from .timer import Timer
        from .printer import Output
        from .nanoparticle import Nanoparticle

        self.input_filename = input_filename
        self.output_filename = output_filename
        self.log_filename = log_filename

        # check if given input file exists
        if input_filename is not None and not path.isfile(input_filename):
            raise DIMPyError(f'Input file `{input_filename}` does not exist!')

        # start a timer, even if you're just reading output data
        self._timer = Timer()

        # create the output and log files
        self.out = Output(filename=output_filename)
        self.log = Output(filename=log_filename, logfile=True)


    def read_input(self, input_filename=None):
        '''\
        Reads the input options from a DIMPy input file.

        :param input_filename: The name of the DIMPy input file (Optional).
        '''

        from .read_input_file import read_input_file

        self.log('Reading input file', time=self._timer.startTimer('Prep Input'))

        if input_filename is None:
            input_filename = self.input_filename

        if input_filename is None:
            raise DIMPyError('Must specify an input filename!')
        elif not path.isfile(input_filename):
            raise DIMPyError(f'Input file `{input_filename}` does not exist!')

        self._input_options = read_input_file(input_filename)
        self._timer.endTimer('Prep Input')


class DIMPyError(Exception):
    """Error class for DIMPy errors.

    Parameters
    ----------
    msg : :obj:`str`
        The message to give to the user.

    Examples
    --------

        >>> import dimpy
        >>> try:
        ...     filedata = dimpy.DIMPy(input_file='file.dim')
        ... except dimpy.DIMPyError as d:
        ...     sys.exit(str(d))

    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def run_from_command_line():
    """\
    Reads the inputs from the command line and runs the calculation
    """

    from os.path import splitext
    from input_reader import abs_file_path
    from argparse import ArgumentParser
    from dimpy import Nanoparticle, DDA

    # Assume that argparse exists and create an argument parser
    parser = ArgumentParser(description="Front-end for the DIMPy code.", prog='DIMPy')
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
        output_filename = '.'.join([splitext(abs_file_path(args.file))[0], 'out'])

    ################################
    # Perform the actual calculation
    ################################

    # read in the nanoparticle data from the file
    nanoparticle = Nanoparticle(args.file, output_filename=output_filename)

    # set up a calculations method
    dda = DDA(nanoparticle)

    dda.run()

if __name__ == '__main__':
    try:
        run_from_command_line()
    except KeyboardInterrupt:
        exit(1)
