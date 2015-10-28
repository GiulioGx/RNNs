import numpy
from theano import tensor as TT

from Configs import Configs
from infos.InfoElement import PrintableInfoElement
from learningRule.LearningRule import LearningStepRule, LearningStepSymbols

__author__ = 'giulio'


class ConstantNormalizedStep(LearningStepRule):
    class Symbols(LearningStepSymbols):
        def __init__(self, rule, dir_symbols):
            self.__learning_rate = rule.lr_value / dir_symbols.direction.norm()

        @property
        def learning_rate(self):
            return self.__learning_rate

        @property
        def infos(self):
            return [self.__learning_rate]

        def format_infos(self, infos_symbols):
            lr_info = PrintableInfoElement('lr', ':02.2e', infos_symbols[0].item())
            return lr_info, infos_symbols[lr_info.length:len(infos_symbols)]

    def __init__(self, lr_value=0.001):
        self.__lr_value = TT.alloc(numpy.array(lr_value, dtype=Configs.floatType))

    @property
    def lr_value(self):
        return self.__lr_value

    def compile(self, net, obj_fnc, dir_symbols):
        return ConstantNormalizedStep.Symbols(self, dir_symbols)
