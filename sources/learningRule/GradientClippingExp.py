from theano import tensor as TT
from theano.ifelse import ifelse

from ObjectiveFunction import ObjectiveFunction
from infos.InfoElement import PrintableInfoElement
from infos.InfoList import InfoList
from learningRule.LearningStepRule import LearningStepRule
from theanoUtils import is_inf_or_nan

__author__ = 'giulio'


class GradientClippingExp(LearningStepRule):
    def __init__(self, lr_value=0.01, clip_thr_l2=1., clip_thr_l1=0.1):
        self.__lr_value = lr_value
        self.__clip_thr_l1 = clip_thr_l1
        self.__clip_thr_l2 = clip_thr_l2

    def compute_lr(self, net, obj_fnc: ObjectiveFunction, direction):

        lr = self.__lr_value
        n1 = direction.norm(1)
        n2 = direction.norm(2)
        scale = ifelse(TT.or_(n1 < self.__clip_thr_l1, is_inf_or_nan(n1)),
                                        1., TT.cast((self.__clip_thr_l1 / n1), dtype="float32"))

        scale = ifelse(TT.or_(n2/scale < self.__clip_thr_l2, is_inf_or_nan(n2)),
                                        scale, scale * TT.cast((self.__clip_thr_l2 / n2), dtype="float32"))

        computed_learning_rate = lr * scale

        return computed_learning_rate, LearningStepRule.Infos(computed_learning_rate)

    @property
    def updates(self):
        return []

    @property
    def infos(self):
        step_info = PrintableInfoElement('constant_step', ':02.2e', self.__lr_value)
        thr_l1_info = PrintableInfoElement('clipping_thr_l1', ':02.2e', self.__clip_thr_l1)
        thr_l2_info = PrintableInfoElement('clipping_thr_l2', ':02.2e', self.__clip_thr_l2)
        info = InfoList(step_info, thr_l1_info, thr_l2_info)
        return info
