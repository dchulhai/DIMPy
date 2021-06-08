from collections import OrderedDict
from datetime import datetime
import functools
from time import time as currtime, process_time

__all__ = ['Timer', 'TimerError']

class TimerError(Exception):
    """Exception for errors in using the timer"""
    def __init__(self, msg='Error in using the timer'):
        self.msg = msg
    def __str__(self):
        return self.msg

class Timer(object):
    """
    Keeps track of the timing statistics for different parts of a code.

    This class uses a stack approach so that the last started timer must
    be the first ended timer.  This facillitates speed, removes the need
    to search for timer instances on each stop, and prevents bad programming.
    Routines are named internally based on the all timers in the stack.
    """

    def __init__(self):
        """Initiallize a Timer instance."""
        # Create a new ordered dictionary for logging times
        self._log = OrderedDict()
        # Initiallize the timer stack
        self._stack_long  = []
        self._stack_short = []
        # keep track of what timers were already logged
        self._logged = {}

    def startTimer(self, routine, time=None, cpu=None, short=None):
        """
        Starts a timer and adds it to the stack. 'group' is used to group
        the timings together.  A place for this timer is created in the log.
        Returns the time stamp.
        """
        # Use now if nothing was given
        if time is None:
            time = currtime()
        # Use the time for the cpu if not given
        if cpu is None:
            cpu = process_time()
        # If the short name is not given, make it the same as the long
        if short is None:
            short = routine
        # Add this timer to the stack
        self._stack_short.append(short)
        self._stack_long.append(routine)
        # Create a name based on all the timers in the stack.
        name = ' => '.join(self._stack_short)
        # Add to the log.
        try:
            # Assume that we've already seen a timer of this name
            self._log[name][1].append(time)
        # Account for times when this name does not exist yet
        except KeyError:
            self._log[name] = [routine, [time], [cpu]]
        else:
            self._log[name][2].append(cpu)

        return datetime.fromtimestamp(time)

    def endTimer(self, routine, time=None, cpu=None):
        """
        Ends a timer.  Checks that it is the last started timer in the
        stack, otherwise a ValueError is raised.  This is to ensure that
        the appropriate timer ordering is maintained.
        If everything checks out, the result is added to the log.
        Returns a tuple of the end time stamp and the elapsed time.
        """
        # Use now if nothing was given
        if time is None:
            time = currtime()
        # Use time for cpu if not given
        if cpu is None:
            cpu = process_time()
        # Create a name based on all the timers in the stack.
        name = ' => '.join(self._stack_short)
        # Make sure the last timer is the one we want to stop
        if routine != self._stack_long[-1]:
            string = '"{0}" is not the most recently started timer!'
            raise TimerError (string.format(routine))
        # Removes last timer
        self._stack_long.pop()
        self._stack_short.pop()
        # Determie the total time, and add this to the log
        elapsed = time - self._log[name][1].pop()
        self._log[name][1].append(elapsed)
        cpu_elapsed = cpu - self._log[name][2].pop()
        self._log[name][2].append(cpu_elapsed)

        return datetime.fromtimestamp(time), elapsed

    def endAllTimers(self, leave=0):
        """
        Ends all timers in the case of an emergency stop.
        leave indicates how many timers to keep running
        """
        if leave < 0:
            raise TimerError ('leave parameter must be positive')
        # End all timers in reverse order using current time
        while len(self._stack_long) > leave:
            self.endTimer(self._stack_long[-1])

    def dumpTimings(self, verbosity=0, out=print):
        """
        Prints the timing statistics in a nice table.
        Verbosity level can be 0, 1, or 2.
        0 - Each routine is printed with # calls and total time spent
        1 - Divided into timing groups, with # calls and total time per group
        2 - Same as 1, but for each call the time is listed as well.

        out is a function that can be used to redirect output.
        By default the standard print function is used.
        """
        if verbosity not in (0, 1, 2,):
            string = 'dumpTimings: Invalid verbosity level {0}'
            raise TimerError (string.format(verbosity))
 
        # Print heading
        line = 'Timing Statistics'
        dashes = '='*len(line)
        out()
        out()
        out(dashes.center(79))
        out(line.center(79))
        out(dashes.center(79))
        out()

        f = '  {0:>14.3f}  {1:>12.3f}  {2:>7d}  {3:<}'
        # Verbosity level 0
        if verbosity == 0:
            routines = OrderedDict()

            # Sum up all timings from routines with same name from all groups
            for fullpath, (routine, times, cpu) in self._log.items():
                r = routine[0].upper() + routine[1:]
                try:
                    routines[r][0] += sum(times)
                    routines[r][1] += sum(cpu)
                    routines[r][2] += len(times)
                except KeyError:
                    routines[r] = [sum(times), sum(cpu), len(times)]

            # Now print off the timings for the routines
            out('  Total Time (s)  CPU Time (s)  # Calls  Routine')
            out('  --------------------------------------------------------')
            for r, times in routines.items():
                out(f.format(times[0], times[1], times[2], r))

        elif verbosity == 1: 
            out('  Total Time (s)  CPU Time (s)  # Calls  Routine')
            out('  --------------------------------------------------------')
            # Loop over each timing group
            for fullpath, (routine, times, cpu) in self._log.items():
                # Print off the total time spent in that routine
                out(f.format(sum(times), sum(cpu), len(times), fullpath))

        elif verbosity == 2:
            out('  Total Time (s)  CPU Time (s)   Call #  Routine')
            out('  --------------------------------------------------------')
            # Loop over each timing group
            for fullpath, (routine, times, cpu) in self._log.items():
                # Print off each time this routine was called
                for i, (t, c) in enumerate(zip(times, cpu)):
                    out(f.format(t, c, i+1, fullpath))

        # Leave a blank at end.
        out()

def check_time(function=None, log='all'):
    '''Checks how much time a function uses.'''
    if not function:
        return functools.partial(check_time, log=log)
    @functools.wraps(function)
    def new_function(self, *args, **kwargs):

        # get a function name
        function_name = type(self).__name__+'.'+function.__name__

        # start the timer
        if ((log != 'debug') or (log == 'debug' and self.debug)):
            start_time = self._timer.startTimer(function_name)

        # check whether we should log the time of this function
        log_this = ((log == 'all') or ((log == 'once') and
            (function_name not in self._timer._logged))
            or self.debug)

        # log the timer
        if log_this:
            self.log('Starting '+function_name, time=start_time)
        try:
            self._timer._logged[function_name] += 1
        except KeyError:
            self._timer._logged[function_name] = 1

        result = function(self, *args, **kwargs)

        # end the timer
        if ((log != 'debug') or (log == 'debug' and self.debug)):
            end_time = self._timer.endTimer(function_name)

        # log the timer
        if log_this:
            self.log('Finished '+function_name
                +' in {0:.3f} seconds'.format(end_time[1]),
                time=end_time[0])

        return result
    return new_function
