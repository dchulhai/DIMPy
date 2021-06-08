import functools
import resource

class Memory(object):
    """
    Keeps tract of the memory statistics for different parts of
    a code.
    """

    def __init__(self):

        self._logs = {}

        self._startMem = {}

    def startLog(self, function_name):
        self._startMem[function_name] = self._getMem()

    def endLog(self, function_name):
        self._endMem = self._getMem()
        useage = self._endMem - self._startMem[function_name]

        if function_name in self._logs:
            if useage > self._logs[function_name]:
                self._logs[function_name] = useage
        else:
            self._logs[function_name] = useage

    def printLogs(self, verbosity=0, out=print):
        """
        Prints the memory statistics in a nice table.
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
        out('  Routine                            Memory (MB)')
        out('  ----------------------------------------------')
        f = '  {0:<30s} {1:>15.2f}'
        for function in self._logs:
            out(f.format(function, self._logs[function]))
        out(f.format('Total Process', total_memory))

    def _getMem(self):
        '''
        Get current memory useage in MB.
        '''
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

#def check_memory(function):
#    '''Checks how much memory a function uses.'''
#    @functools.wraps(function)
#    def new_function(self, *args, **kwargs):
#        function_name = type(self).__name__+'.'+function.__name__
#        self._memory.startLog(function_name)
#        result = function(self, *args, **kwargs)
#        self._memory.endLog(function_name)
#        return result
#    return new_function

def check_memory(function=None, log='all'):
    '''Checks how much memory a function uses.'''
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
