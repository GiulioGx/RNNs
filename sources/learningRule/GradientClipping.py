import numpy
from theano import tensor as TT
from Configs import Configs
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import PrintableInfoElement
from infos.InfoList import InfoList
from learningRule.LearningRule import LearningStepRule

__author__ = 'giulio'


class GradientClipping(LearningStepRule):
    class Symbols(LearningStepRule.Symbols):
        def __init__(self, rule, dir_symbols: DescentDirectionRule.Symbols):
            norm = dir_symbols.direction.norm()
            TT.switch(norm > rule.clip_thr, rule.clip_thr / norm, rule.lr_value)

            self.__learning_rate = TT.alloc(numpy.array(rule.lr_value, dtype=Configs.floatType))

        @property
        def learning_rate(self):
            return self.__learning_rate

        @property
        def infos(self):
            return [self.__learning_rate]

        def format_infos(self, infos_symbols):
            lr_info = PrintableInfoElement('lr', ':02.2e', infos_symbols[0].item())
            return lr_info, infos_symbols[lr_info.length:len(infos_symbols)]

    def __init__(self, lr_value=0.01, clip_thr=1.):
        self.__lr_value = lr_value
        self.__clip_thr = clip_thr

    @property
    def lr_value(self):
        return self.__lr_value

    @property
    def clip_thr(self):
        return self.__clip_thr

    @property
    def infos(self):
        step_info = PrintableInfoElement('constant_step', ':02.2e', self.__lr_value)
        thr_info = PrintableInfoElement('clipping_thr', ':02.2e', self.__clip_thr)
        info = InfoList(step_info, thr_info)
        return info

    def compile(self, net, obj_fnc, dir_symbols):
        return ClippingGradient.Symbols(self, dir_symbols)
