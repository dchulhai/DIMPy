#! /usr/bin/env python3

import os

import argparse
import input_reader
import numpy as np

from .dimpy_error import DIMPyError
from .methods import DDAs, DDAr, DIMs, DIMr
from .nanoparticle import Nanoparticle
from .tools.printer import Output
from .input_file.read_input_file import ReadInput
from .tools.timer import Timer
from ._version import __version__

def run_from_command_line():
    """Read the inputs from the command line and runs the calculation.

    For useage, see:

    .. code-block:: console

        python -m dimpy --help

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

    calculation = run(args.file, output_filename=output_filename)

def run (filename, output_filename=None, run_calc=True):
    """Run a DIMPy input file.

    :param filename str: The name of the DIMPy input file. See
        :class:`dimpy.input_file.ReadInput` for input file format

    :param output_filename: The name of the output file to print
        calculation information, default None
    :type output_filename: str, optional

    :param run_calc: whether to run the calculation, default True
    :param run_calc: bool, optional

    :returns: :class:`dimpy.methods.CalcMethodBase` object

    **Example**::

    >>> import dimpy
    >>> calc = dimpy.run('filename.dimpy')

    """

    # read the DIMPy input file
    options = ReadInput(filename).read()

    # create the nanoparticle
    nano = options.nanoparticle
    nanoparticle = Nanoparticle(nano.atoms, output_filename=output_filename,
                                verbose=options.verbose, debug=options.debug,
                                pbc=nano.pbc, unit=nano.unit,
                                atom_params=nano.atom_params)
    nanoparticle.build()

    # get calculation method options
    method = options.method

    calc = []
    calc.append(method.interaction.upper())
    if method.kdir is not None:
        calc.append('RET')
    else:
        calc.append('STA')
    calc.append('PIM')

    # get calculation method based on calctype
    methods = {'DDAs': [['STA', 'DDA', 'PIM'], DDAs],
               'DDAr': [['RET', 'DDA', 'PIM'], DDAr],
               'DIMs': [['STA', 'DIM', 'PIM'], DIMs],
               'DIMr': [['RET', 'DIM', 'PIM'], DIMr],
              }
    method_base = None
    for calctype in methods:
        if all([x in methods[calctype][0] for x in calc]):
            method_base = methods[calctype][1]
    if method_base is None:
        raise DIMPyError('Calculation type not yet implemented!')

    # run the calculation
    calculation = method_base(nanoparticle, kdir=method.kdir,
                              freqs=method.freqs,
                              solver=method.solver, title=options.title)

    if run_calc:
        try:
            calculation.run()


        # print statistics if the program was interrupted
        except KeyboardInterrupt:
            calculation._memory.printLogs(verbosity=0, out=calculation.out)
            calculation._timer.endAllTimers()
            calculation._timer.dumpTimings(verbosity=0, out=calculation.out)
            raise KeyboardInterrupt

    # return calculation
    return calculation

if __name__ == '__main__':
    try:
        run_from_command_line()
    except KeyboardInterrupt:
        exit(1)
