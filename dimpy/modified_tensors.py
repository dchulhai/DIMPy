import numpy as np
import scipy as sp

from .calc_method import CalcMethod


class ModifiedTensors(CalcMethod):


    @property
    def t0(self):
        '''The modified zeroth-order interaction tensor.'''
        if self._t0 is None:

            self.log('Generating modified T0 tensor',
                     time=self._timer.startTimer('T0 Tensor'))

            ##################
            # CODE GOES HERE #
            ##################

            dists = self.nanoparticle.distances
            self._t0 = np.divide(1, dists, out=np.zeros_like(dists),
                                 where=dists!=0, dtype=np.float32)

            ############
            # END CODE #
            ############

            end_time = self._timer.endTimer('T0 Tensor')
            self.log('Finished generating modified T0 tensor, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])


        return self._t0


    @property
    def t1(self):
        '''The modified first-order interaction tensor.'''
        if self._t1 is None:

            self.log('Generating modified T1 tensor',
                     time=self._timer.startTimer('T1 Tensor'))

            ##################
            # CODE GOES HERE #
            ##################

            r_vec = self.nanoparticle.r_vec
            r_inv = self.t0
            r3_inv = r_inv * r_inv * r_inv
            self._t1 = -1.0 * r3_inv[:,:,np.newaxis] * r_vec

            ############
            # END CODE #
            ############

            end_time = self._timer.endTimer('T1 Tensor')
            self.log('Finished generating modified T1 tensor, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])

        return self._t1


    @property
    def t2(self):
        '''The modified second-order interaction tensor.'''
        if self._t2 is None:

            self.log('Generating modified T2 tensor',
                     time=self._timer.startTimer('T2 Tensor'))

            ##################
            # CODE GOES HERE #
            ##################

            r_vec = self.nanoparticle.r_vec
            r_inv = self.t0
            r3_inv = r_inv * r_inv * r_inv
            r5_inv = r3_inv * r_inv * r_inv

            self._t2 = 3 * np.einsum('ij,ijk,ijl->ijkl', r5_inv, r_vec,
                                     r_vec, dtype=np.float32, casting='same_kind')
            self._t2[:,:,0,0] -= r3_inv
            self._t2[:,:,1,1] -= r3_inv
            self._t2[:,:,2,2] -= r3_inv

            ############
            # END CODE #
            ############

            end_time = self._timer.endTimer('T2 Tensor')
            self.log('Finished generating modified T2 tensor, '
                     '{0:.3f} seconds'.format(end_time[1]),
                     time=end_time[0])

        return self._t2

PIM = ModifiedTensors
