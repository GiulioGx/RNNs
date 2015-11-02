from combiningRule.CombiningRule import CombiningRule
import theano.tensor as TT
import theano as T
from theanoUtils import norm, is_not_real

__author__ = 'giulio'


class NormalizedSum(CombiningRule):

    def normalize_step(self, grads_combination, norms):
        return grads_combination/norm(grads_combination)

    def get_linear_coefficients(self, vector_list, n):
        
        def f(v):
            norm_v = norm(v)
            return TT.switch(TT.or_(is_not_real(norm_v), norm_v <= 0), 0, 1/norm_v)

        values, _ = T.scan(f, sequences=[TT.unbroadcast(TT.as_tensor_variable(vector_list), 1)],
                           outputs_info=[None],
                           non_sequences=[],
                           name='net_output',
                           mode=T.Mode(linker='cvm'),
                           n_steps=n)

        return values




