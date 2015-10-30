from combiningRule.CombiningRule import CombiningRule
from theanoUtils import norm, is_not_real
import theano.tensor as TT

__author__ = 'giulio'


class NormalizedSum(CombiningRule):

    def __init__(self):
        self.__thr = 0.001

    def normalize_step(self, grads_combination, norms):
        return grads_combination/norm(grads_combination)

    def step(self, v, acc):
        norm_v = norm(v)
        return TT.switch(TT.or_(is_not_real(norm_v), norm_v <= 0), acc, v/norm_v + acc), norm_v

