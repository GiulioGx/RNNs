import numpy
from theano import tensor as TT

from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo
from learningRule.LearningStepRule import LearningStepRule

__author__ = 'giulio'


class ConstantStep(LearningStepRule):
    def __init__(self, lr_value=0.001, normalized_wrt_dir_norm: bool = False):
        self.__lr_value = lr_value
        self.__normalized_wrt_dir_norm = normalized_wrt_dir_norm

    def compute_lr(self, net, obj_fnc: ObjectiveFunction, direction):

        if self.__normalized_wrt_dir_norm:
            lr = TT.alloc(numpy.array(self.__lr_value, dtype=Configs.floatType)) / direction.norm()
        else:
            lr = TT.alloc(numpy.array(self.__lr_value, dtype=Configs.floatType))

        return lr, LearningStepRule.Infos(lr)

    @property
    def updates(self):
        return []

    @property
    def infos(self):
        return InfoGroup('constant step', InfoList(PrintableInfoElement('constant_step', ':02.2e', self.__lr_value),
                                                   PrintableInfoElement('normalized', '',
                                                                        self.__normalized_wrt_dir_norm)))
