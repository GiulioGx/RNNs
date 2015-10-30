from combiningRule.CombiningRule import CombiningRule
from theanoUtils import norm

__author__ = 'giulio'


class SimpleSum(CombiningRule):
    def normalize_step(self, grads_combinantion, norms):
        return grads_combinantion

    def step(self, v, acc):
        return v + acc, norm(v)
