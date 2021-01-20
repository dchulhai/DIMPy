from input_reader import ReaderError
from .timer import Timer, TimerError
from .parallel import Parallel
from .process import ProcessError
from .define_input_file import define_input_file
from .input_formatter import input_formatter
from .printer import log, Output, printInput
import sys
import os


def run(inputfilename, outputfilename, nprocs=None, nolog=False):
    """\
    Run the DIM code.

    The :func:`run` command is called directly by the command-line
    ``dim`` executable, so this function provides a way to script
    ``dim`` calculations.

    Parameters
    ----------
    inputfilename : :obj:`str`
        The DIM input file with a '.dim' or '.inp' extension
    outputfilename : :obj:`str`
        The output of the DIM run
    nprocs : :obj:`int`, optional
        Number of processors to use for the job
    nolog : :obj:`bool`, optional
        Suppresses log output to standard output

    """

    # Redirect log output if requested
    if nolog:
        savestdout = sys.stdout
        dn = open(os.devnull, 'w')
        sys.stdout = dn

    # Start the timer
    timer = Timer()
    start = timer.startTimer('Total Process', short='DIM')
    log('Reading input file', time=timer.startTimer('Prep Input'))

    # Define what the input block should look like
    reader = define_input_file()

    # Read in the file
    try:
        inputblock = reader.read_input(inputfilename)
    except ReaderError as e:
        sys.exit(str(e))
    # Open up the output file for printing
    outfile = Output(outputfilename)

    # Print the input block at head of file if requested
    if 'input' in inputblock.print_rules:
        printInput(outfile, inputblock.input_file)

    log('Converting input to FORTRAN-readable format')
    FORTRAN_input = input_formatter(inputblock, outputfilename, start)
    #sys.exit(FORTRAN_input.getvalue())
    timer.endTimer('Prep Input')

    # Set the number of processors
    procs = Parallel(nprocs, inputblock.algorithm == 0)

    # Return the class that will be used to call the FORTRAN routine
    log('Entering DIM FORTRAN routine')
    try:
        dimprocess = procs.get_process(timer, outfile, silent=nolog)

        # Execute the job
        #f = open('input.in', 'w')
        #print(FORTRAN_input.getvalue(), file=f)
        #f.close()
        try:
            dimprocess.run_job(FORTRAN_input.getvalue())
        except KeyboardInterrupt:
            log('Calculation terminated from a system SIGNAL')
            dimprocess.abrupt_quit(force=True)
            raise KeyboardInterrupt
        except (ProcessError, TimerError) as e:
            log(str(e))
            raise KeyboardInterrupt
        finally:
            # Report timing data on successful exit
            log('End of DIM calculation')
            # Close the "input file"
            FORTRAN_input.close()
            # Read the timing messages from FORTRAN
            dimprocess.timings()
            # Wrap up and print the timer
            try:
                timer.endTimer('Total Process')
            except TimerError as te:
                sys.exit(str(te))
            except IndexError:
                sys.exit('Error parsing timers')
            if 'timing' in inputblock.print_rules:
                timer.dumpTimings(0, out=outfile)
            elif 'timingverbose' in inputblock.print_rules:
                timer.dumpTimings(1, out=outfile)
            elif 'timingveryverbose' in inputblock.print_rules:
                timer.dumpTimings(2, out=outfile)
            # Recover stdout
            if nolog:
                sys.stdout = savestdout
                dn.close()

    except OSError:
        log ('Test terminated - No calculations ran')
