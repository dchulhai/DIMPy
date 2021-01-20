from subprocess import Popen, PIPE
from .printer import error
from os import chmod, setpgrp
from os import stat as osstat
from os.path import join, dirname, realpath
import sys
import stat

__all__ = ['Process', 'ProcessError', 'fix_binary']


def fix_binary():
    """Changes the permission on the DIM binary to be executable"""
    DIMEXE = join(dirname(realpath(__file__)), 'bin', 'DIM')
    st = osstat(DIMEXE)
    try:
        chmod(DIMEXE, st.st_mode | stat.S_IEXEC | stat.S_IXOTH)
    except OSError:
        msg = "Error updating the permissions of the DIM binary.\n"
        msg += "Execute 'dim --fix_binary' as root to fix.\n"
        msg += "You should only need to do this once."
        sys.exit(msg)


class ProcessError(Exception):
    """Exception for errors in the calculation"""

    def __init__(self, msg='Error in subprocess'):
        self.msg = msg

    def __str__(self):
        return self.msg


class Process(object):
    """
    A class to manage a child process and process it's output
    The double entendre is intentional.
    """

    def __init__(self, command, executable, timer, out, env=None, silent=False):
        """Open the pipe to the process"""

        def runit():
            if silent:
                if env is not None:
                    self._child = Popen(command, env=env, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                else:
                    self._child = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            else:
                if env is not None:
                    self._child = Popen(command, env=env, stdin=PIPE, stderr=PIPE)
                else:
                    self._child = Popen(command, stdin=PIPE, stderr=PIPE)

        self._name = executable
        self._complete = False
        self._command = command
        # Save the timer and out function internally
        self._timer = timer
        self._out = out
        # List for timing data straight from Fortran
        self._timingdata = []
        # The child process, all IO is piped.  When first installed, the binary may not
        # be executable
        try:
            runit()
        except (OSError, IOError):
            fix_binary()
            try:
                runit()
            except (OSError, IOError) as e:
                raise ProcessError('Unable to start ' + self._name + ' : '+str(e))

    def run_job(self, input_file):
        """Execute the job by giving it the input file"""
        # Execute the job
        self._out.close()
        dummy, stderr = self._child.communicate(input_file)
        self._complete = True
        # Extract the errors and/or timing info from the standard error stream
        err = []
        for line in stderr.split('\n'):
            if line[0:4] == 'TIME':
                self._timingdata.append(line[4:])
            else:
                err.append(line)
        # Reopen output file.
        self._out.open()
        # Check for errors
        self._error_message('\n'.join(err))

    def abrupt_quit(self, force=False):
        """Send termination signal to the program"""
        if force:
            try:
                self._child.terminate()
            except OSError:
                pass
            else:
                error('Termination signal sent to ' + self._name)
        # Reopen output file
        self._out.open()
        # End all but the last timer
        self._timer.endAllTimers(leave=1)

    def timings(self):
        """Parses timing data from FORTRAN output"""

        # Read in the file
        for line in self._timingdata:
            # Get a short label
            sh = None if not line[1:11].strip() else line[1:11].strip()
            # Get the real time and cpu time
            time = float(line[11:31].strip())
            cpu  = float(line[31:51].strip())
            # Get the routine nemae
            label = line[51:].rstrip()
            # Some lines are start of timers, some are end
            if line[0] == 'S':
                self._timer.startTimer(label, time=time, cpu=cpu, short=sh)
            else:
                self._timer.endTimer(label, time=time, cpu=cpu)

    def _error_message(self, err):
        """Prints out the error message of the child if it
        did not exit properly.
        """
        if err:
            error('Error message from {0}:'.format(self._name))
            error(err)
            self._out('ERROR:', err)
            raise ProcessError('\n ERROR: '+err)

    def __del__(self):
        """On termination of the program, make sure child is stopped.
        This is included in case of a keyboard interrupt or sigterm
        of the python process.
        """
        # If child was not started for some reason, do nothing
        # Also, nothing to do if the process has completed
        if getattr(self, '_child', True) or self._complete:
            return
        else:
            self.abrupt_quit(force=True)
