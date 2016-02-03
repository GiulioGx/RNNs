import abc
import theano.tensor as TT

from combiningRule.CombiningRule import CombiningRule
from infos.Info import NullInfo, Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo
from theanoUtils import is_not_trustworthy

__author__ = 'giulio'


class LinearCombination(CombiningRule):
    __metaclass__ = abc.ABCMeta

    def __init__(self, normalize_components: bool = True):
        self.__normalize_components = normalize_components

    def combine(self, H):
        coefficients = self.get_linear_coefficients(H)

        # matrix implementation
        G = H
        if self.__normalize_components:
            norm_G = H.norm(2, axis=1).reshape((H.shape[0], 1))
            G = H / TT.switch(is_not_trustworthy(norm_G), 1, norm_G)

        debug_norm = G.norm(2, axis=1).mean()
        combination = TT.dot(G.T, coefficients)

        return combination, LinearCombination.Infos(debug_norm)

    @property
    def infos(self):
        return InfoList(PrintableInfoElement('normalized_comp', '', self.__normalize_components))

    @abc.abstractmethod
    def get_linear_coefficients(self, H):
        """returns a list of coefficients to be used to combine the first n vectors in 'vector_list' """

    class Infos(SymbolicInfo):
        def __init__(self, db_g_norm):
            self.__symbols = [db_g_norm]

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacedments: list) -> Info:
            return PrintableInfoElement('db_G_norm', '', symbols_replacedments[0])
