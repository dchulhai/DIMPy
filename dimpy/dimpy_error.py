class DIMPyError(Exception):
    """Error class for DIMPy errors.

    :param msg: The message to give to the user, default is
        'Error in using DIMPy'
    :type msg: str, optional

    **Example**::

    >>> import sys
    >>> from dimpy import DIMPyError
    >>> def function():
    >>>     # do something
    >>>     if some_error:
    >>>         raise DIMPyError('Error message')

    >>> try:
    >>>     function()
    >>> except DIMPyError as e:
    >>>     sys.exit(str(e))

    """
    def __init__(self, msg='Error in using DIMPy'):
        self.msg = msg 

    def __str__(self):
        return self.msg


