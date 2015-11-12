from combiningRule.LinearCombination import LinearCombinationRule
import theano.tensor as TT
import theano as T

from infos.InfoElement import SimpleDescription
from theanoUtils import norm

__author__ = 'giulio'


class SimpleSum(LinearCombinationRule):

    def get_linear_coefficients(self, vector_list, n):

        values, _ = T.scan(norm, sequences=[TT.unbroadcast(TT.as_tensor_variable(vector_list), 1)],
                           outputs_info=[None],
                           non_sequences=[],
                           name='simple_sum_coeffs_scan',
                           n_steps=n)

        return values

    def normalize_step(self, grads_combinantion, norms):
        return grads_combinantion

    @property
    def infos(self):
        return SimpleDescription('simple_sum_combination')
