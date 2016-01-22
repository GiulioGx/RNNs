
import theano.tensor as TT
import theano.tensor.nlinalg as li
import theano.tensor.slinalg as sli

from Configs import Configs
from combiningRule.CombiningRule import CombiningRule
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoList import InfoList

__author__ = 'giulio'


class EquiangularCombination(CombiningRule):

    @property
    def infos(self):
        return SimpleDescription('equiangular_combination')

    def compile(self, H):
        return EquiangularCombination.Symbols(H)

    class Symbols(CombiningRule.Symbols):

        @property
        def infos(self):
            return self.__info

        def format_infos(self, infos_symbols):
            shape = PrintableInfoElement('shape', '', infos_symbols[1])
            cos = PrintableInfoElement('equi_cos', ':1.3f', infos_symbols[0].item())
            info = InfoList(cos, shape)
            return info, infos_symbols[info.length:len(infos_symbols)]

        @property
        def combination(self):
            return self.__combination

        @property
        def equi_cos(self):
            return self.__equi_cos

        def __init__(self, H):

            # normalize rows
            G = H / H.norm(2, axis=1).reshape((H.shape[0], 1))
            u = TT.ones((G.shape[0], 1))

            # solve problem
            r = li.qr(G.T, mode='r')  # QR factorization
            # _, r = li.qr(G.T, mode='complete')

            x = sli.solve(r.T, u)
            b = sli.solve(r, x)
            self.__equi_cos = 1./TT.sqrt(TT.sum(b))
            lambda_ = (-self.__equi_cos**2) * b
            d = - TT.dot(G.T, lambda_)/self.__equi_cos

            self.__combination = d
            self.__info = [self.__equi_cos, G.shape]
