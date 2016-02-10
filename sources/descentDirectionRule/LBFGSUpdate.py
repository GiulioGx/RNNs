from theano.ifelse import ifelse

from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from combiningRule.OnesCombination import OnesCombination
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.Info import Info
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo
from theanoUtils import vec_dot
import theano as T
import theano.tensor as TT
import numpy


class LBFGSDirection(DescentDirectionRule):
    def __init__(self, n_pairs: int, initial_h_scaling: float = 1):
        self.__updates = []
        self.__n_pairs = n_pairs
        self.__initial_h_scaling = initial_h_scaling

    @property
    def infos(self):
        return SimpleDescription('LBFGS direction')  # TODO

    @staticmethod
    def __update_storage(original_storage, new_value):
        # schift the pairs queue (LIFO) # 0 is the newest entry
        updated_storage = TT.set_subtensor(original_storage[1:], original_storage[0:-1])
        # add the new pair
        updated_storage = TT.set_subtensor(updated_storage[0], new_value)
        return updated_storage

    @staticmethod
    def __rho(sk, yk):
        return 1. / vec_dot(yk, sk)

    def __initialize(self, n_variables):
        self.__counter = T.shared(numpy.int32(0), name='lbfgs_iteration_counter')
        self.__stored_grad = T.shared(name='stored_gradient_differences',
                                      value=numpy.zeros(shape=(self.__n_pairs, n_variables, 1),
                                                        dtype=Configs.floatType))
        self.__stored_x = T.shared(name='stored_x__differences',
                                   value=numpy.zeros(shape=(self.__n_pairs, n_variables, 1), dtype=Configs.floatType))
        self.__prev_grad = T.shared(name='prev_grad',
                                    value=numpy.zeros(shape=(n_variables, 1), dtype=Configs.floatType),
                                    broadcastable=(False, True))
        self.__prev_x = T.shared(name='prev_x', value=numpy.zeros(shape=(n_variables, 1), dtype=Configs.floatType),
                                 broadcastable=(False, True))

        self.__n_pairs_collected = T.shared(numpy.int8(0))  # it is used as a boolean variable
        self.__first_step_completed = T.shared(numpy.int8(0))  # it is used as a boolean variable

    def __compute_h_dot(self, gradient):  # TODO
        """computes the dot between the hessian approximation and the gradient using the two-loop strategy"""

        def first_loop(sk, yk, q_acc):
            alpha_k = LBFGSDirection.__rho(sk, yk) * vec_dot(sk, q_acc)
            q = q_acc - alpha_k * yk

            return q, alpha_k

        values, _ = T.scan(first_loop, sequences=[self.__stored_x, self.__stored_grad],
                           outputs_info=[TT.unbroadcast(gradient, 1), None],
                           name='lbfgs_fist_loop_scan')

        q_first_loop = values[0][-1]
        alpha = values[1]

        q_first_loop_dot_h0 = self.__compute_h_k0_dot(q_first_loop)

        def second_loop(sk, yk, alpha_k, q_acc):
            beta_k = LBFGSDirection.__rho(sk, yk) * vec_dot(yk, q_acc)
            q = q_acc + sk * (alpha_k - beta_k)
            return q

        values, _ = T.scan(second_loop, sequences=[self.__stored_x, self.__stored_grad, alpha],
                           outputs_info=[q_first_loop_dot_h0],
                           go_backwards=True,
                           name='lbfgs_second_loop_scan')

        q_second_loop = values[-1]
        q_second_loop = TT.addbroadcast(q_second_loop, 1)

        return q_second_loop

    @property
    def __gamma_k(self):

        #s_km1 = self.__stored_x[1]
        #y_km1 = self.__stored_grad[1]
        #return vec_dot(s_km1, y_km1) / vec_dot(y_km1, y_km1)
        return TT.constant(1., dtype=Configs.floatType)

    def __compute_h_k0_dot(self, q):
        """compute the dot product between H_k0 and a vector q"""
        gamma_k = self.__gamma_k
        return q * gamma_k

    def direction(self, net, obj_fnc: ObjectiveFunction):
        """returns the LBFGS direction"""

        gradients_combination, combining_strategy_symbolic_info = obj_fnc.grad.temporal_combination(
            OnesCombination(normalize_components=False))
        antigradient = - gradients_combination

        self.__initialize(net.n_variables)

        g = antigradient.as_tensor()
        direction = ifelse(self.__n_pairs_collected > 0, self.__compute_h_dot(g), g)
        # xkp1 = antigradient * lr + net.symbols.current_params.as_tensor()

        xk = net.symbols.current_params.as_tensor()
        sk = xk - self.__prev_x
        yk = - g - self.__prev_grad

        # storing new pairs
        stored_grad_update = ifelse(self.__first_step_completed > 0,
                                    LBFGSDirection.__update_storage(self.__stored_grad, yk),
                                    self.__stored_grad)
        stored_x_update = ifelse(self.__first_step_completed > 0, LBFGSDirection.__update_storage(self.__stored_x, sk),
                                 self.__stored_x)

        prev_grad_update = - g
        prev_x_update = xk

        counter_update = ifelse(TT.and_(self.__counter >= self.__n_pairs, self.__first_step_completed > 0),
                                TT.constant(0, dtype='int32'),
                                self.__counter + 1)
        first_step_update = TT.constant(1, dtype='int8')
        n_pairs_collected_update = ifelse(TT.or_(self.__n_pairs_collected > 0, self.__counter >= self.__n_pairs), 1, 0)

        self.__updates = [(self.__counter, counter_update), (self.__stored_grad, stored_grad_update),
                          (self.__stored_x, stored_x_update), (self.__prev_grad, prev_grad_update),
                          (self.__prev_x, prev_x_update), (self.__first_step_completed, first_step_update),
                          (self.__n_pairs_collected, n_pairs_collected_update)]

        direction_vars = net.from_tensor(direction)

        return direction_vars, LBFGSDirection.Infos(direction_vars, gradients_combination, sk, yk,
                                                    self.__first_step_completed, self.__gamma_k)  # TODO infos

    @property
    def updates(self):
        return self.__updates

    class Infos(SymbolicInfo):
        def __init__(self, direction, gradient, sk, yk, collected, gamma_k):
            sy_dot = vec_dot(sk, yk)
            grad_dot = direction.cos(gradient)
            dir_norm = direction.norm(2)
            self.__symbols = [dir_norm, grad_dot, sy_dot, collected, sk.norm(2), yk.norm(2), gamma_k]

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacements: list) -> Info:
            dir_norm_info = PrintableInfoElement('dir_norm', ':07.3f', symbols_replacements[0].item())
            dot_info = PrintableInfoElement('grad_dot', ':1.2f', symbols_replacements[1].item())
            sy_dot_info = PrintableInfoElement('sy_dot', ':02.2e', symbols_replacements[2].item())
            collected = PrintableInfoElement('collected', ':1.2f', symbols_replacements[3].item())
            sk_norm = PrintableInfoElement('sk_norm', ':02.2e', symbols_replacements[4].item())
            yk_norm = PrintableInfoElement('yk_norm', ':02.2e', symbols_replacements[5].item())
            gamma_k_info = PrintableInfoElement('gamma_k', ':02.2e', symbols_replacements[6].item())
            info = InfoList(dir_norm_info, dot_info, sy_dot_info, collected, sk_norm, yk_norm, gamma_k_info)
            return info
