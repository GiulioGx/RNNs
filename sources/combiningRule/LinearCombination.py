import abc
import theano as T
import theano.tensor as TT

from Configs import Configs
from combiningRule.CombiningRule import CombiningRule
from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement, NonPrintableInfoElement
from infos.InfoList import InfoList
from theanoUtils import norm, is_inf_or_nan, is_not_trustworthy

__author__ = 'giulio'


class LinearCombination(CombiningRule):
    __metaclass__ = abc.ABCMeta

    def compile(self, H):
        return LinearCombination.Symbols(self, H)

    def __init__(self, normalize_components: bool = True):
        self.__normalize_components = normalize_components

    @property
    def infos(self):
        return InfoList(PrintableInfoElement('normalized_comp', '', self.__normalize_components))

    @property
    def normalize_components(self):
        return self.__normalize_components

    class Symbols(CombiningRule.Symbols):
        @property
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos

        @property
        def combination(self):
            return self.__grads_combinantions

        def __init__(self, rule, H):
            self.__infos = []
            coefficients = rule.get_linear_coefficients(H)

            # scan implementation
            # if rule.normalize_components:
            #     self.__g = lambda v: v.norm(2)  # FIXME frobenius
            # else:
            #     self.__g = lambda v: TT.cast(TT.alloc(1.), dtype=Configs.floatType)
            #
            # values, _ = T.scan(self.__step,
            #                    sequences=[TT.unbroadcast(TT.as_tensor_variable(vector_list), 1),
            #                               coefficients],
            #                    outputs_info=[TT.unbroadcast(TT.zeros_like(vector_list[0]), 1), None],
            #                    non_sequences=[],
            #                    name='linear_combination_scan',
            #                    n_steps=n)
            #
            # grads_combinantions = values[0]
            # separate_norms = values[1]
            # self.__grads_combinantions = grads_combinantions[-1]

            # matrix implementation
            G = H
            if rule.normalize_components:
                norm_G = H.norm(2, axis=1).reshape((H.shape[0], 1))
                # G = G / TT.switch(norm_G > 0, norm_G, 1)
                G = H / TT.switch(is_not_trustworthy(norm_G), 1, norm_G)

            self.__grads_combinantions = TT.dot(G.T, coefficients)

        def __step(self, v, alpha, acc):
            norm_fac_v = self.__g(v)
            return TT.switch(TT.or_(TT.or_(is_inf_or_nan(norm_fac_v), norm_fac_v <= 0), is_inf_or_nan(alpha)), acc,
                             (v * alpha) / norm_fac_v + acc), norm_fac_v

    @abc.abstractmethod
    def get_linear_coefficients(self, H):
        """returns a list of coefficients to be used to combine the first n vectors in 'vector_list' """
