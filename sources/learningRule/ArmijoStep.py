import numpy
import theano as T
from theano import tensor as TT

from Configs import Configs
from infos.InfoList import InfoList
from infos.InfoElement import PrintableInfoElement
from learningRule.LearningRule import LearningStepRule, LearningStepSymbols

__author__ = 'giulio'


class ArmijoStep(LearningStepRule):
    class Symbols(LearningStepSymbols):
        def __init__(self, rule, net, obj_fnc, dir_symbols):
            net_symbols = net.symbols
            obj_symbols = obj_fnc.compile(net, net_symbols.current_params, net_symbols.u,
                                          net_symbols.t)

            f_0 = obj_symbols.objective_value
            grad_dir_dot_product = obj_symbols.grad.dot(dir_symbols.direction)

            def armijo_step(step, beta, alpha, u, t):  # FIXME se capissi perchè togliendoli nn va...
                new_param = net_symbols.current_params + (dir_symbols.direction * step)
                new_obj_symbols = obj_fnc.compile(net, new_param, net_symbols.u, net_symbols.t)
                f_1 = new_obj_symbols.objective_value

                condition = f_0 - f_1 >= -alpha * step * grad_dir_dot_product  # sufficient decrease condition

                return step * beta, [], T.scan_module.until(
                    condition)

            values, updates = T.scan(armijo_step, outputs_info=rule.init_step,
                                     non_sequences=[rule.beta, rule.alpha,
                                                    net_symbols.u, net_symbols.t],
                                     n_steps=rule.max_steps)

            n_steps = values.size

            self.__learning_rate = values[-1] / rule.beta
            self.__infos = [self.__learning_rate, n_steps]

        @property
        def learning_rate(self):
            return self.__learning_rate

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos_symbols):
            lr_info = PrintableInfoElement('lr', ':02.2e', infos_symbols[0].item())
            n_step_info = PrintableInfoElement('n_step', ':02d', infos_symbols[1].item())
            infos = InfoList(lr_info, n_step_info)
            return infos, infos_symbols[2:len(infos_symbols)]

    def __init__(self, alpha=0.1, beta=0.5, init_step=1, max_steps=10):
        self.__init_step = TT.alloc(numpy.array(init_step, dtype=Configs.floatType))
        self.__beta = TT.alloc(numpy.array(beta, dtype=Configs.floatType))
        self.__alpha = TT.alloc(numpy.array(alpha, dtype=Configs.floatType))
        self.__max_steps = TT.alloc(numpy.array(max_steps, dtype=int))

    @property
    def init_step(self):
        return self.__init_step

    @property
    def alpha(self):
        return self.__alpha

    @property
    def max_steps(self):
        return self.__max_steps

    @property
    def beta(self):
        return self.__beta

    def compile(self, net, obj_fnc, dir_symbols):
        return ArmijoStep.Symbols(self, net, obj_fnc, dir_symbols)