#! /usr/bin/env python3

from sys import exit
from os.path import splitext
from input_reader import abs_file_path
from argparse import ArgumentParser, Action
from .run import run
from .parallel import Parallel
from .process import fix_binary
from ._version import __version__


def main():
    """\
    Front-end for the DIM code.
    """

    # Assume that argparse exists and create an argument parser
    parser = ArgumentParser(description="Front-end for the DIMPy code.", prog='DIMPy')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    parser.add_argument('file', help='The input file to read.')
    # Add the options
    parser.add_argument('-o', '--out', help='Defines the file to output data '
                        'to.  If not given, the input name is used with a '
                        '.out extension.')
    parser.add_argument('--fix_binary', action='store_true',
                        default=False,
                        help='Distutils changes the mode '
                        'of the DIM binary to non-executable.  The first '
                        'time you run dim it will try to change this '
                        'back to executable for you, but if it cannot you '
                        'will need run with this mode as root to fix.')
    parser.add_argument
    if Parallel.is_parallel():
        parser.add_argument('-n', '--nproc', **Parallel.nproc_opts())
    args = parser.parse_args()
    # Fix the binary if needed
    if args.fix_binary:
        fix_binary()

    # Determine output file name.
    # It has the same name as the input but with .out unless otherwise given
    if args.out is not None:
        outputname = args.out
    else:
        outputname = '.'.join([splitext(abs_file_path(args.file))[0], 'out'])

    # Perform the actual calculation
    if Parallel.is_parallel():
        run(args.file, outputname, args.nproc)
    else:
        run(args.file, outputname)


def show():
    """\
    Front-end for the DIM printing commands.
    """
    try:
        from dimclass import DIM
    except ImportError:
        exit("Cannot import the DIM tools... they may not have been compiled.")

    parser = ArgumentParser(description="Front-end for the DIM printing commands.",
                            prog='dim-show')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    modes = parser.add_subparsers(title='print modes',
                                  description='Choose the type of information '
                                              'to print to the screen',
                                  help='Select the type of information to print',
                                  dest='mode')
    coords = modes.add_parser('coords',
                              help='Print the atomic coordinates to '
                                   'standard output')
    coords.add_argument('-n', '--num', action='store_true', default=False,
                        help='Prints the number of the atom with the '
                             'coordinates')

    pol = modes.add_parser('pol',
                           help='Print the polarizabiliy tensors to '
                                'standard output')
    pol.add_argument('--ani', action='store_true', default=False,
                     help='Includes the anisotropic polarizabiliy with '
                          'the tensor')
    pol.add_argument('-u', '--unit', choices=('au', 'ev', 'nm'),
                     default='au',
                     help='For frequency-dependent calculations, '
                          'specifies the unit to print the frequencies in. '
                          "The default is '%(default)s'.")
    cross = modes.add_parser('cross',
                             help='Prints the absorption cross-sections to '
                                  'the standard output')
    cross.add_argument('--absunit', default='angstroms',
                       choices=('angstroms', 'bohr', 'nm', 'cm', 'm'),
                       help='The units of absorption cross-sections to print. '
                            "The default is '%(default)s'.")
    molar = modes.add_parser('molar',
                             help='Prints the molar absorptivitty to the '
                                  'standard output')
    absorbance = modes.add_parser('absorbance',
                                  help='Prints the unitless absorbance to the '
                                       'standard output')
    transmittance = modes.add_parser('transmittance',
                                     help='Prints the normalized '
                                          'transmittance to the '
                                          'standard output')
    # Add common arguments
    for m in (cross, molar, absorbance, transmittance):
        m.add_argument('-u', '--unit', choices=('au', 'ev', 'nm'),
                       default='au',
                       help='Specifies the unit '
                            'to print the frequencies in. '
                            "The default is '%(default)s'.")
        if m in (absorbance, transmittance):
            m.add_argument('--conc', type=float, default=1E-6,
                           help="The Beer's law concentration in molarity. "
                                "The default is %(default)s")
            m.add_argument('--path', type=float, default=1.0,
                           help="The Beer's law pathlength in centimeters. "
                                "The default is %(default)s")
    parser.add_argument('file', help='The DIM output file to read.')
    args = parser.parse_args()

    # Read the data
    data = DIM(args.file)
    data.collect()

    # Print out the information
    if args.mode == 'pol':
        data.printTensor(iso=True, ani=args.ani, unit=args.unit)
    elif args.mode == 'coords':
        if args.num:
            data.printCoords(mode='num')
        else:
            data.printCoords()
    elif args.mode == 'cross':
        data.printOptical(property='cross-section', unit=args.unit,
                          absunit=args.absunit)
    elif args.mode == 'molar':
        data.printOptical(property='absorptivitty', unit=args.unit)
    elif args.mode == 'absorbance':
        data.printOptical(property='absorbance', unit=args.unit,
                          concentration=args.conc, pathlength=args.path)
    elif args.mode == 'transmittance':
        data.printOptical(property='transmittance', unit=args.unit,
                          concentration=args.conc, pathlength=args.path)


def plot():
    """Easily plot DIM data"""
    from plot import plot_dim

    color_dict = {'blue': 'b', 'green': 'g', 'red': 'r', 'cyan': 'c',
                  'magenta': 'm', 'yellow': 'y', 'black': 'k', 'white': 'w'}

    class ColorAction(Action):
        def __call__(self, parser, namespace, value, option_string=None):
            setattr(namespace, self.dest, color_dict[value])

    point_dict = {'square': 's', 'circle': 'o', 'tri_up': '^',
                  'tri_down': 'v', 'tri_right': '>', 'tri_left': '<',
                  'plus': '+', 'cross': 'x', 'star': '*', 'diamond': 'D'}

    class PointAction(Action):
        def __call__(self, parser, namespace, value, option_string=None):
            setattr(namespace, self.dest, point_dict[value])

    line_dict = {'solid': '-', 'dashed': '--', 'dot_dash': '-.', 'dotted': '.'}

    class LineAction(Action):
        def __call__(self, parser, namespace, value, option_string=None):
            setattr(namespace, self.dest, line_dict[value])

    parser = ArgumentParser(description="Plot DIM output data.",
                            prog='dim-plot')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {0}'.format(__version__))
    modes = parser.add_subparsers(title='plot modes',
                                  description='Choose the type of information '
                                              'to plot',
                                  help='Select the type of information to plot',
                                  dest='mode')
    pol = modes.add_parser('pol',
                           help='Plot the isotropic polarizability')
    pol.add_argument('--real', action='store_true', default=False,
                     help='If plotting polarizabilities, this will plot '
                     'only the real component.  By default both real and '
                     'imaginary are plotted.')
    pol.add_argument('--imag', action='store_true', default=False,
                     help='If plotting polarizabilities, this will plot '
                     'only the imaginary component.  By default both real '
                     'and imaginary are plotted.')
    pol.add_argument('--ylimits2', type=float, nargs=2,
                     default=[None, None],
                     help='The plot limits on the Y-axis.  By default the'
                     'limits are dictated by the data. This is only used '
                     'for the imaginary polarizability when both the '
                     'real and imaginary polarizabilities are plotted.')
    pol.add_argument('--color2', default='g', choices=color_dict.keys(),
                     action=ColorAction,
                     help='Defines the color of the line and points '
                     'of the plot for the imaginary polarizability if '
                     'both real and imaginary are plotted. The default '
                     'is green')
    pol.add_argument('--linestyle2', default='-',
                     choices=line_dict, action=LineAction,
                     help='Specifies the line style of the plot for the '
                     'for the imaginary polarizability if both real '
                     'and imaginary are plotted.  The default '
                     'is solid.')
    pol.add_argument('--pointstyle2', default='s',
                     choices=point_dict.keys(), action=PointAction,
                     help='Specifies the point style of the plot '
                     'for the imaginary polarizability if both real '
                     'and imaginary are plotted. '
                     'The default is square.')
    cross = modes.add_parser('cross',
                             help='Plots the absorption cross-sections')
    cross.add_argument('--absunit', default='angstroms',
                       choices=('angstroms', 'bohr', 'nm', 'cm', 'm'),
                       help='The units of absorption cross-sections to plot. '
                            "The default is '%(default)s'.")
    modes.add_parser('molar', help='Plots the molar absorptivitty')
    absorbance = modes.add_parser('absorbance',
                                  help='Plots the unitless absorbance')
    transmittance = modes.add_parser('transmittance',
                                     help='Plots the normalized '
                                          'transmittance')
    for m in (absorbance, transmittance):
        m.add_argument('--conc', type=float, default=1E-6,
                       help="The Beer's law concentration in molarity, "
                            "The default is %(default)s")
        m.add_argument('--path', type=float, default=1.0,
                       help="The Beer's law pathlength in centimeters. "
                            "The default is %(default)s")
    parser.add_argument('--title', help='A title for the plot.  If your '
                        'title consists of multiple words you must quote it.')
    parser.add_argument('--points', action='store_true', default=False,
                        help='Plot the calculated datapoints over the '
                        'interpolated curve.')
    parser.add_argument('-u', '--unit', choices=('au', 'ev', 'nm'),
                        default='nm',
                        help='Chooses the unit of the frequencies (x-axis) '
                        'to print the frequencies in.  The choices are '
                        "'au', 'nm', or 'ev', and the default is 'nm'.")
    parser.add_argument('--xlimits', type=float, nargs=2, default=[None, None],
                        help='The plot limits on the X-axis.  By default the'
                        'limits are dictated by the data.')
    parser.add_argument('--ylimits', type=float, nargs=2, default=[None, None],
                        help='The plot limits on the Y-axis.  By default the'
                        'limits are dictated by the data.')
    parser.add_argument('--lw', '--linewidth', type=float, default=2.0,
                        help='The width of the lines that are plotted. '
                        'The default is %(default)s.')
    parser.add_argument('--ls', '--linestyle', default='-',
                        choices=line_dict, action=LineAction,
                        help='Specifies the line style of the plot. '
                        'The default is solid.')
    parser.add_argument('--ms', '--pointsize', type=float, default=8.0,
                        help='The size of the points that are plotted. '
                        'The default is %(default)s.')
    parser.add_argument('--pt', '--pointstyle', default='s',
                        choices=point_dict.keys(), action=PointAction,
                        help='Specifies the point style of the plot. '
                        'The default is square.')
    parser.add_argument('--co', '--color', default='b',
                        choices=color_dict.keys(), action=ColorAction,
                        help='Defines the color of the line and points '
                        'of the plot. The default is blue')
    parser.add_argument('file', help='The DIM output file to read.')
    args = parser.parse_args()

    plot_dim(args)


def gui():
    """Start the DIM GUI"""
    pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(1)
