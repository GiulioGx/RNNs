import abc

import theano as T
import theano.tensor as TT

__author__ = 'giulio'


class CombiningRule(object):
    __metaclass__ = abc.ABCMeta

    def combine(self, vector_list, n):
        values, _ = T.scan(self.step, sequences=[TT.unbroadcast(TT.as_tensor_variable(vector_list),1)],
                           outputs_info=[TT.unbroadcast(TT.zeros_like(vector_list[0]), 1), None],
                           non_sequences=[],
                           name='net_output',
                           mode=T.Mode(linker='cvm'),
                           n_steps=n)

        grads_combinantions = values[0]
        separate_norms = values[1]

        normalized_combinantion = self.normalize_step(grads_combinantions[-1], separate_norms)

        return normalized_combinantion, separate_norms

    @abc.abstractmethod
    def step(self, v, acc):
        """define the combining step """

    @abc.abstractmethod
    def normalize_step(self, grads_combinantion, norms):
        """final step, used for an eventual normalization"""
