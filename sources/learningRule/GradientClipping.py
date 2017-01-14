from theano import tensor as TT
from theano.ifelse import ifelse

from ObjectiveFunction import ObjectiveFunction
from infos.InfoElement import PrintableInfoElement
from infos.InfoList import InfoList
from learningRule.LearningStepRule import LearningStepRule
from theanoUtils import is_inf_or_nan

__author__ = 'giulio'


class GradientClipping(LearningStepRule):
    def __init__(self, lr_value=0.01, clip_thr=1., clip_style: str = 'l1'):
        self.__lr_value = lr_value
        self.__clip_thr = clip_thr
        self.__clip_style = clip_style.lower()

    def compute_lr(self, net, obj_fnc: ObjectiveFunction, direction):

        if self.__clip_style == "l2":
            norm = direction.norm(2)
        elif self.__clip_style == "l1":
            norm = direction.norm(1)  # * direction.shape[0] * direction.shape[1]
        else:
            raise AttributeError(
                "not supported clip_style '{}', available styles are 'l1' and 'l2'".format(self.__clip_style))

        lr = self.__lr_value
        computed_learning_rate = ifelse(TT.or_(norm < self.__clip_thr, is_inf_or_nan(norm)),
                                        TT.cast(lr, dtype='float32'), TT.cast((self.__clip_thr / norm) * lr, dtype="float32"))

        return computed_learning_rate, LearningStepRule.Infos(computed_learning_rate)

    @property
    def updates(self):
        return []

    @property
    def infos(self):
        step_info = PrintableInfoElement('constant_step', ':02.2e', self.__lr_value)
        thr_info = PrintableInfoElement('clipping_thr', ':02.2e', self.__clip_thr)
        clip_style = PrintableInfoElement('clip_style', '', self.__clip_style)
        info = InfoList(step_info, thr_info, clip_style)
        return info
