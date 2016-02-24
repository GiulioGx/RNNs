import theano as T
from theano import tensor as TT
from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from learningRule.LearningStepRule import LearningStepRule

__author__ = 'giulio'


class ArmijoStep(LearningStepRule):
    def __init__(self, alpha=0.1, beta=0.5, init_step=1., max_steps=10):
        self.__init_step = TT.cast(TT.alloc(init_step), dtype=Configs.floatType)
        self.__beta = TT.cast(TT.alloc(beta), dtype=Configs.floatType)
        self.__alpha = TT.cast(TT.alloc(alpha), dtype=Configs.floatType)
        self.__max_steps = max_steps

        alpha = PrintableInfoElement('alpha', ':2.2f', alpha)
        beta = PrintableInfoElement('beta', ':2.2f', beta)
        init_step = PrintableInfoElement('init_step', ':2.2f', init_step)
        max_steps = PrintableInfoElement('max_steps', ':2.2f', max_steps)
        self.__infos = InfoGroup('armijo', InfoList(alpha, beta, init_step, max_steps))

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

    @property
    def infos(self):
        return self.__infos

    def compile(self, net, obj_fnc: ObjectiveFunction, dir_symbols: DescentDirectionRule.Symbols):
        return ArmijoStep.Symbols(self, net, obj_fnc, dir_symbols)

    class Symbols(LearningStepRule.Symbols):
        def __init__(self, rule, net, obj_fnc: ObjectiveFunction, dir_symbols: DescentDirectionRule.Symbols):
            net_symbols = net.symbols
            obj_symbols = obj_fnc.compile(net, net_symbols.current_params, net_symbols.u,
                                          net_symbols.t)

            dir_tensor = dir_symbols.direction.as_tensor()
            curr_params_tensor = net_symbols.current_params.as_tensor()

            grad_dir_dot_product = dir_symbols.direction.dot(obj_symbols.grad)

            def armijo_step(step, beta, alpha, x0, f0, direction, grad_dot, u, t):
                x_new = x0 + (direction * step)
                y, _ = net.net_output(net.from_tensor(x_new), u)  # FIXME nn funziona con la penalty
                f1 = obj_fnc.loss(y, t)
                condition = f0 - f1 >= -alpha * step * grad_dot
                return step * beta, T.scan_module.until(condition)

            values, _ = T.scan(armijo_step, outputs_info=[rule.init_step],
                               non_sequences=[rule.beta, rule.alpha, curr_params_tensor, obj_symbols.objective_value,
                                              dir_tensor, grad_dir_dot_product, net_symbols.u, net_symbols.t],
                               n_steps=rule.max_steps,
                               name='armijo_scan')

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
            info = InfoList(lr_info, n_step_info)
            return info, infos_symbols[info.length:len(infos_symbols)]
