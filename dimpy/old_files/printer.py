from datetime import datetime
import sys

class Output(object):
    """Class for writing to an output file.  It is called like a function.
    or like a file object"""

    def __init__(self, filename):
        """Opens the file to be written."""
        self.name = filename
        try:
            self.f = open(self.name, 'w')
        except TypeError:
            self.f = self.name
        except IOError:
            sys.exit('Error opening file '+self.name)

    def __call__(self, *args, **kwargs):
        """Writes to file and adds a space to the front of the message.
        The space can be ommited with the nospace keyword.
        Takes the same keyword arguments as print except file."""
        nospace = kwargs.pop('nospace', False)
        if 'file' in kwargs:
            raise TypeError('Cannot give "file" keyword to Output class')
        elif nospace:
            print(*args, file=self.f, **kwargs)
        else:
            print('', *args, file=self.f, **kwargs)

    def close(self):
        """Close the file."""
        if getattr(self, 'f', False) and self.f != self.name:
            self.f.close()

    def open(self):
        """Reopens the file"""
        try:
            self.f = open(self.name, 'a')
        except TypeError:
            self.f = self.name

    def __del__(self):
        """Close the file if it is not already closed"""
        self.close()

def error(*args, **kwargs):
    """
    Wrapper for print(*args, file=sys.stderr, **kwargs)
    """
    if 'file' in kwargs:
        raise TypeError('Cannot give "file" keyword to error')
    print(*args, file=sys.stderr, **kwargs)

def log(*args, **kwargs):
    """
    Wrapper for print that adds a timestamp to the front of the output.
    The space can be ommited with the nospace keyword.
    time is a keyword argument that indicates the time to display.
    It must be a datetime object or omitted.
    """
    time = kwargs.pop('time', datetime.now())
    nospace = kwargs.pop('nospace', False)
    if nospace:
        print(time.strftime('%c'), *args, **kwargs)
    else:
        print('', time.strftime('%c'), *args, **kwargs)
    return time

def printInput(out, inputfile):
    """
    Echos the input deck.
    """
    for line in inputfile:
        # Skip blank lines
        if line.strip():
            out(line, nospace=True)
    out()
    out()
    out()
