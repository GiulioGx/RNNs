from Configs import Configs
from combiningRule.CombiningRule import CombiningRule
from theanoUtils import norm, is_not_real
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams


__author__ = 'giulio'


class StochasticCombination(CombiningRule):

    def __init__(self, seed=Configs.seed):
        self.__srng = RandomStreams(seed=seed)

    def normalize_step(self, grads_combinantion, norms):
        return grads_combinantion/norm(grads_combinantion)

    def step(self, v, acc):
        rand_scalar = self.__srng.uniform((1, 1))
        norm_v = norm(v)
        return TT.switch(TT.or_(is_not_real(norm_v), norm_v <= 0), acc, v/norm_v*rand_scalar + acc), norm(v)