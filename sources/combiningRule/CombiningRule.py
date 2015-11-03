import abc
import theano as T
import theano.tensor as TT
from theanoUtils import norm, is_not_real

__author__ = 'giulio'


class CombiningRule(object):
    __metaclass__ = abc.ABCMeta

    def combine(self, vector_list, n):
        coefficients = self.get_linear_coefficients(vector_list, n)

        values, _ = T.scan(CombiningRule.step, sequences=[TT.unbroadcast(TT.as_tensor_variable(vector_list), 1),
                                                          coefficients],
                           outputs_info=[TT.unbroadcast(TT.zeros_like(vector_list[0]), 1), None],
                           non_sequences=[],
                           name='net_output',
                           mode=T.Mode(linker='cvm'),
                           n_steps=n)

        grads_combinantions = values[0]
        separate_norms = values[1]

        normalized_combinantion = self.normalize_step(grads_combinantions[-1], separate_norms)

        return normalized_combinantion, separate_norms

    @staticmethod
    def step(v, alpha, acc):
        norm_v = norm(v)
        return TT.switch(TT.or_(is_not_real(norm_v), norm_v <= 0), acc, (v * alpha) / norm(v) + acc), norm_v

    @abc.abstractmethod
    def get_linear_coefficients(self, vector_list, n):
        """returns a list of coefficients to be used to combine the first n vectors in 'vector_list' """

    @abc.abstractmethod
    def normalize_step(self, grads_combinantion, norms):
        """final step, used for an eventual normalization"""
