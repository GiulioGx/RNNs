import abc

from ObjectiveFunction import ObjectiveFunction
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoList import InfoList
from infos.SimpleInfoProducer import SimpleInfoProducer
from infos.SymbolicInfoProducer import SymbolicInfoProducer
from penalty import Penalty
import theano.tensor as TT

from theanoUtils import is_inf_or_nan

__author__ = 'giulio'


class DirectionWithPenalty(DescentDirectionRule):
    """A decorator which add a penalty to the direction"""

    @property
    def infos(self):
        return InfoList(self.__descent_dir.infos, self.penalty.infos)

    def __init__(self, direction_rule: DescentDirectionRule, penalty: Penalty, penalty_lambda=0.1):
        self.__penalty = penalty
        self.__penalty_lambda = penalty_lambda
        self.__descent_dir = direction_rule

    @property
    def penalty(self):
        return self.__penalty

    @property
    def penalty_lambda(self):
        return self.__penalty_lambda

    @property
    def descent_dir(self):
        return self.__descent_dir

    def compile(self, net_symbols, obj_symbols: ObjectiveFunction.Symbols):
        return DirectionWithPenalty.Symbols(self, net_symbols, obj_symbols)

    class Symbols(SymbolicInfoProducer):
        def __init__(self, rule, net_symbols, obj_symbols: ObjectiveFunction.Symbols):
            self.__dir_symbols = rule.descent_dir.compile(net_symbols, obj_symbols)
            dir_infos = self.__dir_symbols.infos

            # add penalty
            self.__penalty_symbols = rule.penalty.compile(net_symbols.current_params, net_symbols)
            penalty_infos = self.__penalty_symbols.infos
            penalty_grad = self.__penalty_symbols.penalty_grad

            direction = self.__dir_symbols.direction
            add_term = TT.switch(is_inf_or_nan(penalty_grad.norm(2)), TT.alloc(0.),
                                 rule.penalty_lambda * (-penalty_grad) / (penalty_grad.norm(2)))
            direction.setW_rec(direction.W_rec + add_term)  # FIXME
            self.__direction = direction
            self.__infos = dir_infos + penalty_infos

        @property
        def direction(self):
            return self.__direction

        def format_infos(self, infos_symbols):
            dir_info, infos_symbols = self.__dir_symbols.format_infos(infos_symbols)
            penalty_info, infos_symbols = self.__penalty_symbols.format_infos(infos_symbols)
            info = InfoList(dir_info, penalty_info)
            return info, infos_symbols

        @property
        def infos(self):
            return self.__infos
