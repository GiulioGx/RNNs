import abc
from builtins import property
from Configs import Configs
import theano as T
import theano.tensor as TT
import numpy
from DescentDirectionRule import DescentDirectionSymbols
from InfoProducer import InfoProducer
from theanoUtils import norm

__author__ = 'giulio'


class LearningStepRule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, net, obj_fnc, dir_symbols: DescentDirectionSymbols):
        """return the compiled version"""


class LearningStepSymbols(InfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def learning_rate(self):
        """return a theano expression for the learning rate """


class ConstantStep(LearningStepRule):
    class Symbols(LearningStepSymbols):
        def __init__(self, rule):
            self.__learning_rate = rule.lr_value

        @property
        def learning_rate(self):
            return self.__learning_rate

        @property
        def infos(self):
            return [self.__learning_rate]

        def format_infos(self, infos):
            return 'lr: {:02.4f}'.format(infos[0].item()), infos[1:len(infos)]

    def __init__(self, lr_value=0.001):
        self.__lr_value = TT.alloc(numpy.array(lr_value, dtype=Configs.floatType))

    @property
    def lr_value(self):
        return self.__lr_value

    def compile(self, net, obj_fnc, dir_symbols):
        return ConstantStep.Symbols(self)


class WRecNormalizedStep(LearningStepRule):
    class Symbols(LearningStepSymbols):
        def __init__(self, rule, net_symbols):
            self.__learning_rate = rule.lr_value / norm(net_symbols.W_rec)

        @property
        def learning_rate(self):
            return self.__learning_rate

        @property
        def infos(self):
            return [self.__learning_rate]

        def format_infos(self, infos):
            return 'lr: {:02.4f}'.format(infos[0].item()), infos[1:len(infos)]

    def __init__(self, lr_value=0.001):
        self.__lr_value = TT.alloc(numpy.array(lr_value, dtype=Configs.floatType))

    @property
    def lr_value(self):
        return self.__lr_value

    def compile(self, net, obj_fnc, dir_symbols):
        net_symbols = net.symbol_closet
        return WRecNormalizedStep.Symbols(self, net_symbols)


class ConstantNormalizedStep(LearningStepRule):
    class Symbols(LearningStepSymbols):
        def __init__(self, rule, net_symbols):
            grad_norm = norm(net_symbols.W_rec_dir, net_symbols.W_in_dir, net_symbols.W_out_dir, net_symbols.b_out_dir,
                             net_symbols.b_rec_dir)
            self.__learning_rate = rule.lr_value / grad_norm(net_symbols.W_rec)

        @property
        def learning_rate(self):
            return self.__learning_rate

        @property
        def infos(self):
            return [self.__learning_rate]

        def format_infos(self, infos):
            return 'lr: {:02.4f}'.format(infos[0].item()), infos[1:len(infos)]

    def __init__(self, lr_value=0.001):
        self.__lr_value = TT.alloc(numpy.array(lr_value, dtype=Configs.floatType))

    @property
    def lr_value(self):
        return self.__lr_value

    def compile(self, net, obj_fnc, dir_symbols):
        net_symbols = net.symbol_closet
        return ConstantNormalizedStep.Symbols(self, net_symbols)


class ArmijoStep(LearningStepRule):
    class Symbols(LearningStepSymbols):
        def __init__(self, rule, net, obj_fnc, dir_symbols):
            net_symbols = net.symbols
            obj_symbols = obj_fnc.compile(net, net_symbols.W_rec, net_symbols.W_in, net_symbols.W_out,
                                          net_symbols.b_rec, net_symbols.b_out, net_symbols.u,
                                          net_symbols.t)

            f_0 = obj_symbols.objective_value

            gradient = TT.concatenate(
                [obj_symbols.gW_rec.flatten(), obj_symbols.gW_in.flatten(), obj_symbols.gW_out.flatten(),
                 obj_symbols.gb_rec.flatten(), obj_symbols.gb_out.flatten()]).flatten()

            direction = TT.concatenate(
                [dir_symbols.W_rec_dir.flatten(), dir_symbols.W_in_dir.flatten(), dir_symbols.W_out_dir.flatten(),
                 dir_symbols.b_rec_dir.flatten(),
                 dir_symbols.b_out_dir.flatten()]).flatten()

            grad_dir_dot_product = TT.dot(gradient, direction)

            def armijo_step(step, beta, alpha, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir, f_0, u, t,
                            grad_dir_dot_product):
                W_rec_k = net_symbols.W_rec + step * W_rec_dir
                W_in_k = net_symbols.W_in + step * W_in_dir
                W_out_k = net_symbols.W_out + step * W_out_dir
                b_rec_k = net_symbols.b_rec + step * b_rec_dir
                b_out_k = net_symbols.b_out + step * b_out_dir

                obj_symbols = obj_fnc.compile(net, W_rec_k, W_in_k, W_out_k, b_rec_k, b_out_k, u, t)

                f_1 = obj_symbols.objective_value

                condition = f_0 - f_1 >= -alpha * step * grad_dir_dot_product  # sufficient decrease condition

                return step * beta, [], T.scan_module.until(
                    condition)

            values, updates = T.scan(armijo_step, outputs_info=rule.init_step,
                                     non_sequences=[rule.beta, rule.alpha, dir_symbols.W_rec_dir,
                                                    dir_symbols.W_in_dir, dir_symbols.W_out_dir,
                                                    dir_symbols.b_rec_dir, dir_symbols.b_out_dir,
                                                    f_0,
                                                    net_symbols.u, net_symbols.t, grad_dir_dot_product],
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

        def format_infos(self, infos):
            return 'lr: {:02.4f}, n_steps: {:02d}'.format(infos[0].item(), infos[1].item()), infos[2:len(infos)]

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
