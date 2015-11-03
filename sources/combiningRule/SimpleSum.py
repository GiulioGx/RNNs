from combiningRule.CombiningRule import CombiningRule
import theano.tensor as TT
import theano as T

from theanoUtils import norm

__author__ = 'giulio'


class SimpleSum(CombiningRule):

    def get_linear_coefficients(self, vector_list, n):

        values, _ = T.scan(norm, sequences=[TT.unbroadcast(TT.as_tensor_variable(vector_list), 1)],
                           outputs_info=[None],
                           non_sequences=[],
                           name='net_output',
                           mode=T.Mode(linker='cvm'),
                           n_steps=n)

        return values

    def normalize_step(self, grads_combinantion, norms):
        return grads_combinantion
