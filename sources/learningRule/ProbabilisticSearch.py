from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfo import NullSymbolicInfos
from learningRule.LearningStepRule import LearningStepRule
import theano.tensor as TT
import theano as T


class ProbabilisticSearch(LearningStepRule):
    def __init__(self, init_lr: float = 0.001, prob_check: float = 1., prob_augment: float = 0.5,
                 beta_augment: float = 0.1, beta_lessen: float = 0.5, seed: int = Configs.seed):
        self.__srng = RandomStreams(seed=seed)
        self.__prob_check = prob_check
        self.__prob_augment = prob_augment
        self.__beta_augment = beta_augment  # TODO add assertions
        self.__beta_lessen = beta_lessen
        self.__lr = T.shared(init_lr, name='lr')
        self.__updates = []

    @property
    def updates(self) -> list:
        return self.__updates

    def __random_decision(self, p):
        r = self.__srng.choice(size=(1,), a=[1, 0], replace=True, p=[p, 1. - p],
                                  dtype=Configs.floatType)[0]
        return r > 0

    def __augment_lr(self):
        lr = ifelse(self.__random_decision(self.__prob_augment), self.__lr * self.__beta_augment, self.__lr)
        return lr

    def __divide_lr(self):
        return self.__lr * self.__beta_lessen

    def __compute_loss_after_update(self, net, obj_fnc: ObjectiveFunction, direction):
        old_loss = obj_fnc.current_loss
        x_kp1 = net.symbols.current_params + direction * self.__lr
        y_kp1 = net.symbols.net_output(x_kp1, net.symbols.u)[0]
        new_loss = obj_fnc.value(y_kp1, net.symbols.t)

        lr = ifelse(new_loss > old_loss, self.__divide_lr(), self.__augment_lr())
        return lr

    def compute_lr(self, net, obj_fnc: ObjectiveFunction, direction):
        lr = ifelse(self.__random_decision(self.__prob_check), self.__compute_loss_after_update(net, obj_fnc, direction), self.__lr)
        self.__updates = [(self.__lr, lr)]
        return lr, LearningStepRule.Infos(lr)

    @property
    def infos(self):  # TODO add other fields
        return InfoGroup('probabilistic lr rule',
                         InfoList(PrintableInfoElement('prob_check', ':2.2f', self.__prob_check),
                                  PrintableInfoElement('prob_augment', ':2.2f', self.__prob_augment)))
