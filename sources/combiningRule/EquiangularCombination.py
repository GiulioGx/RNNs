import theano.tensor as TT
import theano.tensor.nlinalg as li
import theano.tensor.slinalg as sli

from combiningRule.CombiningRule import CombiningRule
from infos import Info
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo
from theanoUtils import is_not_trustworthy

__author__ = 'giulio'


class EquiangularCombination(CombiningRule):
    @property
    def infos(self):
        return SimpleDescription('equiangular_combination')

    def combine(self, H):
        # normalize rows
        norm_G = H.norm(2, axis=1).reshape((H.shape[0], 1))
        G = H / TT.switch(is_not_trustworthy(norm_G), 1, norm_G)
        u = TT.ones((G.shape[0], 1))

        # solve problem
        r = li.qr(G.T, mode='r')  # QR factorization
        # _, r = li.qr(G.T, mode='complete')

        x = sli.solve(r.T, u)
        b = sli.solve(r, x)
        equi_cos = 1. / TT.sqrt(TT.sum(b))
        lambda_ = (-equi_cos ** 2) * b
        combination = - TT.dot(G.T, lambda_) / equi_cos

        return combination, EquiangularCombination.Infos(equi_cos, G.shape)

    class Infos(SymbolicInfo):
        def __init__(self, equi_cos, shape):
            self.__symbols = [equi_cos, shape]

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacedments: list) -> Info:
            shape = PrintableInfoElement('shape', '', symbols_replacedments[1])
            cos = PrintableInfoElement('equi_cos', ':1.3f', symbols_replacedments[0].item())
            info = InfoList(cos, shape)
            return info
