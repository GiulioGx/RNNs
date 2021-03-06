from theano import tensor as TT

from infos.InfoElement import SimpleDescription
from infos.InfoList import InfoList
from penalty.Penalty import Penalty
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from theanoUtils import norm2

__author__ = 'giulio'


class AntiGradientWithPenalty(DescentDirectionRule):

    def __init__(self, penalty: Penalty, penalty_lambda=0.001):
        self.__penalty = penalty
        self.__penalty_lambda = penalty_lambda

    @property
    def infos(self):
        return InfoList(SimpleDescription('antigradient'), self.__penalty.infos)

    @property
    def penalty(self):
        return self.__penalty

    @property
    def penalty_lambda(self):
        return self.__penalty_lambda

    def compile(self, net_symbols, obj_symbols):
        return AntiGradientWithPenalty.Symbols(self, net_symbols, obj_symbols)

    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols):
            # add penalty term
            self.__penalty_symbols = rule.penalty.compute_update(net_symbols.current_params, net_symbols)
            penalty_grad = self.__penalty_symbols.penalty_grad
            penalty_grad_norm = penalty_grad.norm(2)

            self.__direction = obj_symbols.failsafe_grad * (-1)  # FIXME - operator

            W_rec_dir = - obj_symbols.failsafe_grad.W_rec
            W_rec_dir = TT.switch(penalty_grad_norm > 0, W_rec_dir - rule.penalty_lambda * penalty_grad,
                                  W_rec_dir)

            # self.__direction.setW_rec(W_rec_dir) #FIXME penalty not working

            self.__infos = self.__penalty_symbols.infos

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            return self.__penalty_symbols.format_infos(infos)

