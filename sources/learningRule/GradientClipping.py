from theano import tensor as TT
from theano.ifelse import ifelse

from ObjectiveFunction import ObjectiveFunction
from infos.InfoElement import PrintableInfoElement
from infos.InfoList import InfoList
from learningRule.LearningStepRule import LearningStepRule
from theanoUtils import is_inf_or_nan

__author__ = 'giulio'


class GradientClipping(LearningStepRule):
    def __init__(self, lr_value=0.01, clip_thr=1., clip_wrt_max_comp:bool = False, normalize_wrt_dimension: bool = False):
        self.__lr_value = lr_value
        self.__clip_thr = clip_thr
        self.__normalize_wrt_dimension = normalize_wrt_dimension
        self.__clip_wrt_max_comp = clip_wrt_max_comp

    def compute_lr(self, net, obj_fnc: ObjectiveFunction, direction):

        if self.__clip_wrt_max_comp:
            norm = direction.norm(1)
        else:
            norm = direction.norm(2)

        lr = self.__lr_value
        if self.__normalize_wrt_dimension:
            lr = lr * direction.shape.prod()
        computed_learning_rate = ifelse(TT.or_(norm < self.__clip_thr, is_inf_or_nan(norm)), lr, (self.__clip_thr / norm) * lr)
        #computed_learning_rate = (self.__clip_thr / norm) * lr

        return computed_learning_rate, LearningStepRule.Infos(computed_learning_rate)

    @property
    def updates(self):
        return []

    @property
    def infos(self):
        step_info = PrintableInfoElement('constant_step', ':02.2e', self.__lr_value)
        thr_info = PrintableInfoElement('clipping_thr', ':02.2e', self.__clip_thr)
        info = InfoList(step_info, thr_info)
        return info
