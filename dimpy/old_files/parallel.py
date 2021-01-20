from os.path import join, dirname
from .process import Process, ProcessError
from glob import glob
import os

__all__ = ['Parallel']

class Parallel(object):
    """A class to hold the parallel environment-related code"""

    # These are the raw parameters from CMAKE
    mpi             = 'OFF'.upper() in ('ON', '1', 'TRUE')
    openmp          = 'ON'.upper() in ('ON', '1', 'TRUE')
    defaultnproc    = '24'
    mpiexec         = ''
    mpiexec_np_flag = ''

    @classmethod
    def is_parallel(cls):
        return cls.mpi or cls.openmp

    # Options for the nprocs command line argument (class method)
    @classmethod
    def nproc_opts(cls):
        if cls.mpi:
            return { 'type' : int,
                     'required' : True,
                     'help' :
                            'The number of processors to use for MPI. '
                            'If -1 is given, it is assumed you are running '
                            'through a PBS script and the number of nodes is '
                            'set by an environment variable' }
        elif cls.openmp:
            return { 'type' : int,
                     'help' :
                            'The number of processors to use for OpenMP. '
                            'If not given and the $OMP_NUM_THREADS environment'
                            ' variable is not defined, then the program '
                            'defaults to '+cls.defaultnproc+'.' }
        else:
            return {}

    def __init__(self, cmd_line_nprocs, threadedMPI=False, custom_mpi=['mpirun']):
        """\
        Determine the number of processors to run, either from a command-line
        option or from the compile-time environment variables.
        """

        # Make a copy of the environment variables that we may edit if needed
        self.env = os.environ.copy()

        # A user defined custom mpi caller
        self.custom_mpi = custom_mpi

        # A value of -1 means let the environment determine nprocs
        self.auto_nprocs = cmd_line_nprocs is not None and cmd_line_nprocs < 0

        # Determine the number of processors to call
        if self.mpi and self.auto_nprocs:
            try:
                self.nprocs = self.env['PBS_NODEFILE']
            except KeyError:
                raise ProcessError ('PBS_NODEFILE variable not set')
        elif self.mpi:
            if not isinstance(cmd_line_nprocs, int):
                raise ProcessError ('Number of processors must be set for MPI')
            else:
                self.nprocs = cmd_line_nprocs
        elif self.openmp and cmd_line_nprocs is not None:
            self.nprocs = cmd_line_nprocs
        elif self.openmp and 'OMP_NUM_THREADS' in self.env:
            self.nprocs = self.env['OMP_NUM_THREADS']
        elif self.openmp:
            self.nprocs = int(self.defaultnproc)
        else:
            self.nprocs = 1

        # Set up the environment for OpenMP
        if self.mpi and not threadedMPI:
            # Set the OpenMP thread number to 1 so BLAS doesn't parallelize
            # We only want this if not doing the direct solver
            self.env.update({'OMP_NUM_THREADS':str(1)})
        elif self.openmp:
            self.env.update({'OMP_NUM_THREADS':str(self.nprocs)})

    def get_process(self, timer, outfile, silent=False):
        """\
        Return the process correctly for the current parallel environment.
        """

        try:
            dimexe = glob(join(dirname(__file__), 'bin', 'DIM*'))[0]
        except IndexError:
            msg = 'Cannot locate DIM executable.  '
            raise OSError (msg+'Have you compiled the FORTRAN source?')
        # Call the FORTRAN routine in the correct way depending on the
        # parallelization method.  
        if self.mpi and self.auto_nprocs:
            # Use MPI with PBS; we don't specify number of flags manually
            return Process(self.custom_mpi+[dimexe], dimexe, timer, outfile,
                            env=self.env, silent=silent)
        elif self.mpi:
            # Use the MPI runtime executer
            executable = [self.mpiexec, self.mpiexec_np_flag, str(self.nprocs), dimexe]
            return Process(executable, dimexe, timer, outfile, env=self.env, silent=silent)
        elif self.openmp:
            # Call OpenMP
            return Process([dimexe], dimexe, timer, outfile, env=self.env, silent=silent)
        else:
            # Just call if serial
            return Process([dimexe], dimexe, timer, outfile, silent=silent)

