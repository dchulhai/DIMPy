import functools
import resource

class Memory(object):
    """Keep tract of the memory statistics for different parts of
    a code.

    **Example**::

    >>> mem = Memory()
    >>> mem.startLog('function 1')
    >>> 
    >>> # run function 1
    >>>
    >>> mem.endLog('function 1')
    >>> mem.startLog('function 2')
    >>>
    >>> # run function 2
    >>>
    >>> mem.endLog('function 2')
    >>> mem.printLogs()
    >>>
    >>> # Output of memory useage for functions 1 and 2

    """

    def __init__(self):
        """Initialize the memory class."""
        self._logs = {}
        self._startMem = {}

    def startLog(self, function_name):
        """Start logging the memory for a function.

        :param str function_name: Name used to log memory useage

        """

        self._startMem[function_name] = self._getMem()

    def endLog(self, function_name):
        """End logging the memory for a function.

        :param str function_name: Name used to log memory useage
            and must match a function name given to
            :meth:`startLog`

        """

        self._endMem = self._getMem()
        useage = self._endMem - self._startMem[function_name]

        if function_name in self._logs:
            if useage > self._logs[function_name]:
                self._logs[function_name] = useage
        else:
            self._logs[function_name] = useage

    def printLogs(self, verbosity=0, out=print):
        """Print the memory statistics in a nice table.

        :param int verbosity: Level of information to print (currently
            not unused)

        :param out: print function to use, default is :func:`print`

        """

        # get total memory
        total_memory = self._getMem()

        # Print heading
        line = 'Memory Statistics'
        dashes = '='*len(line)
        out()
        out()
        out(dashes.center(79))
        out(line.center(79))
        out(dashes.center(79))
        out()

        # print memory statistics
        out('  Routine                                            Memory (MB) ')
        out('  ---------------------------------------------------------------')
        f = '  {0:<47s} {1:>15.2f}'
        for function in self._logs:
            out(f.format(function, self._logs[function]))
        out(f.format('Total Process', total_memory))

    def _getMem(self):
        """Get current memory useage in MB."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def check_memory(function=None, log='all'):
    """A function decorator to check how much memory a
    class function uses. Coded to work specifically with DIMPy.

    :param function: Function to decorate. Function must be a
        class function and the class function must have attributes
        `debug` (bool) and `_memory` (:class:`Memory` object)

    :param log: String to let :class:`Memory` know when to
        keep track of the memory useage. Can be 'all' (default) of 'debug'.
        In the case of 'debug', it will only log the memory if the
        class that the function belongs to has the attribute 'debug'
        set to `True`
    :type log: str, optional

    **Example**::

    >>> from dimpy.memory import Memory, check_memory
    >>> class Test(object):
    >>>     def __init__(self):
    >>>         self._memory = Memory()
    >>>         self.debug = False
    >>>
    >>>     @check_memory
    >>>     def some_function(self):
    >>>         # do stuff, memory always logged
    >>>
    >>>     @check_memory(log='debug')
    >>>     def other_function(self)
    >>>         # do stuff, memory logged only if self.debug=True
    
    >>> temp = Test()
    >>> temp.some_function()
    >>> temp.other_function()
    >>> temp._memory.printLogs()

    """
    if not function:
        return functools.partial(check_memory, log=log)
    @functools.wraps(function)
    def new_function(self, *args, **kwargs):

        # get a function name
        function_name = type(self).__name__+'.'+function.__name__

        # get initial memory usage
        if ((log != 'debug') or (log == 'debug' and self.debug)):
            self._memory.startLog(function_name)

        # run the function
        result = function(self, *args, **kwargs)

        # get final memory usage
        if ((log != 'debug') or (log == 'debug' and self.debug)):
            self._memory.endLog(function_name)

        return result
    return new_function
