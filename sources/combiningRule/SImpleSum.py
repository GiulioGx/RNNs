from combiningRule.CombiningRule import CombiningRule
from theanoUtils import norm
import theano as T
import theano.tensor as TT

__author__ = 'giulio'


class SimpleSum(CombiningRule):
    def combine(self, vector_list, n):
        def step(v, acc):
            return v + acc, norm(v)

        values, _ = T.scan(step, sequences=[TT.as_tensor_variable(vector_list)],
                           outputs_info=[TT.zeros_like(vector_list[0]), None],
                           non_sequences=[],
                           name='net_output',
                           mode=T.Mode(linker='cvm'),
                           n_steps=n)

        grads_combinantions = values[0]
        separate_norms = values[1]

        return grads_combinantions[-1], separate_norms
