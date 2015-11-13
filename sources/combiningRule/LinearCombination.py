import abc
import theano as T
import theano.tensor as TT

from combiningRule.CombiningRule import CombiningRule
from infos.Info import NullInfo
from theanoUtils import norm, is_not_real

__author__ = 'giulio'


class LinearCombinationRule(CombiningRule):
    __metaclass__ = abc.ABCMeta

    def compile(self, vector_list, n):
        return LinearCombinationRule.Symbols(self, vector_list, n)

    class Symbols(CombiningRule.Symbols):

        @property
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos

        @property
        def combination(self):
            return self.__grads_combinantions

        def __init__(self, rule, vector_list, n):
            coefficients = rule.get_linear_coefficients(vector_list, n)

            values, _ = T.scan(LinearCombinationRule.step,
                               sequences=[TT.unbroadcast(TT.as_tensor_variable(vector_list), 1),
                                          coefficients],
                               outputs_info=[TT.unbroadcast(TT.zeros_like(vector_list[0]), 1), None],
                               non_sequences=[],
                               name='linear_combination_scan',
                               n_steps=n)

            grads_combinantions = values[0]
            separate_norms = values[1]

            normalized_combination = rule.normalize_step(grads_combinantions[-1], separate_norms)

            self.__grads_combinantions = normalized_combination

    @staticmethod
    def step(v, alpha, acc):
        norm_v = v.norm(2)  # FIXME frobenius
        return TT.switch(TT.or_(TT.or_(is_not_real(norm_v), norm_v <= 0), is_not_real(alpha)), acc,
                         (v * alpha) / norm_v + acc), norm_v

    @abc.abstractmethod
    def get_linear_coefficients(self, vector_list, n):
        """returns a list of coefficients to be used to combine the first n vectors in 'vector_list' """

    @abc.abstractmethod
    def normalize_step(self, grads_combinantion, norms):
        """final step, used for an eventual normalization"""
