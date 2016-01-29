import abc
import theano.tensor as TT

from combiningRule.CombiningRule import CombiningRule
from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement
from infos.InfoList import InfoList
from theanoUtils import is_not_trustworthy

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
            return self.__infos

        def format_infos(self, infos):
            return PrintableInfoElement('db_G_norm', '', infos[0]), infos[1:]

        @property
        def combination(self):
            return self.__grads_combinantions

        def __init__(self, rule, H):
            self.__infos = []
            coefficients = rule.get_linear_coefficients(H)

            # matrix implementation
            G = H
            if rule.normalize_components:
                norm_G = H.norm(2, axis=1).reshape((H.shape[0], 1))
                G = H / TT.switch(is_not_trustworthy(norm_G), 1, norm_G)

            debug_norm = G.norm(2, axis=1).mean()
            self.__infos = [debug_norm]

            self.__grads_combinantions = TT.dot(G.T, coefficients)

    @abc.abstractmethod
    def get_linear_coefficients(self, H):
        """returns a list of coefficients to be used to combine the first n vectors in 'vector_list' """
