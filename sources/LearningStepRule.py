import abc
from Configs import Configs
import theano as T
import theano.tensor as TT
import numpy
from Rule import Rule
from theanoUtils import norm

__author__ = 'giulio'


class LearningStepRule(Rule):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_lr(self, symbol_closet, obj_symbols, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir):
        """return a theano expression for the learning rate and the number of steps used to compute it"""
        return


class ConstantStep(LearningStepRule):
    def __init__(self, lr_value=0.001):
        self.__lr_value = TT.alloc(numpy.array(lr_value, dtype=Configs.floatType))

    def get_lr(self, symbol_closet, obj_symbols, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir):
        return [self.__lr_value]

    def format_infos(self, infos):
        return '', infos


class W_rec_step():
    def __init__(self, lr_value=0.001):
        self.__lr_value = TT.alloc(numpy.array(lr_value, dtype=Configs.floatType))

    def get_lr(self, symbol_closet, obj_symbols, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir):
        W_rec_norm = norm(W_rec_dir)

        return [self.__lr_value / W_rec_norm]

    def format_infos(self, infos):
        return '', infos


class ConstantNormalizedStep(LearningStepRule):
    def __init__(self, lr_value=0.001):
        self.__lr_value = TT.alloc(numpy.array(lr_value, dtype=Configs.floatType))

    def get_lr(self, symbol_closet, obj_symbols, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir):
        grad_norm = norm(W_rec_dir,W_in_dir,W_out_dir,b_out_dir,b_rec_dir)

        return [self.__lr_value / grad_norm]

    def format_infos(self, infos):
        return '', infos


class ArmijoStep(LearningStepRule):
    def __init__(self, alpha=0.1, beta=0.5, init_step=1, max_steps=10):
        self.__step = TT.alloc(numpy.array(init_step, dtype=Configs.floatType))
        self.__beta = TT.alloc(numpy.array(beta, dtype=Configs.floatType))
        self.__alpha = TT.alloc(numpy.array(alpha, dtype=Configs.floatType))
        self.__n_steps = TT.alloc(numpy.array(max_steps, dtype=int))

    def get_lr(self, symbol_closet, obj_symbols, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir):
        max_steps = 10

        gradient = TT.concatenate(
            [obj_symbols.gW_rec.flatten(), obj_symbols.gW_in.flatten(), obj_symbols.gW_out.flatten(),
             obj_symbols.gb_rec.flatten(), obj_symbols.gb_out.flatten()]).flatten()

        direction = TT.concatenate(
            [W_rec_dir.flatten(), W_in_dir.flatten(), W_out_dir.flatten(),
             b_rec_dir.flatten(),
             b_out_dir.flatten()]).flatten()

        grad_dir_dot_product = TT.dot(gradient, direction)

        def armijo_step(step, beta, alpha, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir, f_0, u, t,
                        grad_dir_dot_product):
            W_rec_k = symbol_closet.W_rec + step * W_rec_dir
            W_in_k = symbol_closet.W_in + step * W_in_dir
            W_out_k = symbol_closet.W_out + step * W_out_dir
            b_rec_k = symbol_closet.b_rec + step * b_rec_dir
            b_out_k = symbol_closet.b_out + step * b_out_dir

            f_1 = obj_symbols.loss(W_rec_k, W_in_k, W_out_k, b_rec_k, b_out_k, u, t)

            condition = f_0 - f_1 >= -alpha * step * grad_dir_dot_product  # sufficient decrease condition

            return step * beta, [], T.scan_module.until(
                condition)

        values, updates = T.scan(armijo_step, outputs_info=self.__step,
                                 non_sequences=[self.__beta, self.__alpha, W_rec_dir,
                                                W_in_dir, W_out_dir,
                                                b_rec_dir, b_out_dir,
                                                symbol_closet.loss_shared,
                                                symbol_closet.u, symbol_closet.t, grad_dir_dot_product],
                                 n_steps=max_steps)

        lr = values[-1] / self.__beta
        n_steps = values.size

        return lr, n_steps

    def format_infos(self, infos):
        return 'n_steps: {:02d}'.format(infos[0].item()), infos[1:len(infos)]
