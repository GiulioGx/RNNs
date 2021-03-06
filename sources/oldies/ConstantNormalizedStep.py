import numpy
from theano import tensor as TT

from Configs import Configs
from infos.InfoElement import PrintableInfoElement
from learningRule.LearningStepRule import LearningStepRule

__author__ = 'giulio'


class ConstantNormalizedStep(LearningStepRule):
    class Symbols(LearningStepRule.Symbols):
        def __init__(self, rule, dir_symbols):
            self.__learning_rate = TT.alloc(numpy.array(rule.lr_value, dtype=Configs.floatType)) / dir_symbols.direction.norm()

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
        self.__lr_value = lr_value

    @property
    def lr_value(self):
        return self.__lr_value

    @property
    def infos(self):
        return PrintableInfoElement('constant_normalized_step', ':02.2e', self.__lr_value)

    def compile(self, net, obj_fnc, dir_symbols):
        return ConstantNormalizedStep.Symbols(self, dir_symbols)
