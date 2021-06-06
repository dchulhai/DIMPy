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


