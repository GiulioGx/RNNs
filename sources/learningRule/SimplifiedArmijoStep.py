import theano as T
from infos.Info import Info
from infos.SymbolicInfo import SymbolicInfo
from theano import tensor as TT
from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from learningRule.LearningStepRule import LearningStepRule
import numpy as np
__author__ = 'giulio'


class SimplifiedArmijoStep(LearningStepRule):

    @property
    def updates(self) -> list:
        return self.__updates

    def __init__(self, beta=0.5, init_step=1., max_steps=100):
        #self.__init_step = TT.cast(TT.alloc(init_step), dtype=Configs.floatType)
        self.__beta = beta
        self.__max_steps = max_steps

        beta_info = PrintableInfoElement('beta', ':2.2f', beta)
        init_step_info = PrintableInfoElement('init_step', ':2.2f', init_step)
        max_steps_info = PrintableInfoElement('max_steps', ':2.2f', max_steps)
        self.__infos = InfoGroup('armijo', InfoList(beta_info, init_step_info, max_steps_info))

        self.__lr = T.shared(np.array(init_step, dtype=Configs.floatType), name='lr')
        self.__updates = []
    #
    #
    # @property
    # def init_step(self):
    #     return self.__init_step

    @property
    def max_steps(self):
        return self.__max_steps

    @property
    def beta(self):
        return self.__beta

    @property
    def infos(self):
        return self.__infos

    def compute_lr(self, net, obj_fnc: ObjectiveFunction, direction):
        net_symbols = net.symbols
        f0 = obj_fnc.value(net_symbols.y, net_symbols.t, net_symbols.mask)
        print("MMMMMMMMMM.....N'Se po' fa na line search di tipo GOLDDDDDDSTEIN?")

        dir_tensor = direction.as_tensor()
        curr_params_tensor = net_symbols.current_params.as_tensor()

        def armijo_step(step, beta, x0, f0, direction, u, t, mask):
            x_new = x0 + (direction * step)
            y, _, _ = net.net_output(net.from_tensor(x_new), u, net_symbols.h_m1)  # FIXME nn funziona con la penalty
            f1 = obj_fnc.value(y, t, mask)
            condition = f0 > f1

            # condition = x_new.sum().sum() >0
            return TT.cast(step * beta, dtype=Configs.floatType), T.scan_module.until(condition)

        values, _ = T.scan(armijo_step, outputs_info=[self.__lr],
                           non_sequences=[self.__beta, curr_params_tensor, f0,
                                          dir_tensor, net_symbols.u, net_symbols.t, net_symbols.mask],
                           n_steps=self.__max_steps,
                           name='armijo_scan')
        n_steps = values.size

        computed_learning_rate = TT.cast(values[-1] * self.__beta, dtype=Configs.floatType)

        self.__updates = [(self.__lr, computed_learning_rate)]
        return computed_learning_rate, ArmijoInfo(computed_learning_rate, n_steps)

class ArmijoInfo(SymbolicInfo):
    @property
    def symbols(self):
        return self.__symbols

    def fill_symbols(self, symbols_replacements: list) -> Info:
        lr_info = PrintableInfoElement('lr', ':02.2e', symbols_replacements[0].item())
        steps_info = PrintableInfoElement('steps', '', symbols_replacements[1].item())

        return InfoList(lr_info, steps_info)

    def __init__(self, lr, n_steps):
        self.__symbols = [lr, n_steps]
