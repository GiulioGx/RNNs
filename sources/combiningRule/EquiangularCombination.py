
import theano.tensor as TT
import theano.tensor.nlinalg as li
import theano.tensor.slinalg as sli
from combiningRule.CombiningRule import CombiningRule
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoList import InfoList

__author__ = 'giulio'


class EquiangularCombination(CombiningRule):

    @property
    def infos(self):
        return SimpleDescription('equiangular_combination')

    def compile(self, vector_list, n):
        return EquiangularCombination.Symbols(vector_list, n)

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

        def __init__(self, vector_list, n):

            # build G matrix
            H = TT.as_tensor_variable(vector_list[0:n])
            G = TT.reshape(H, (H.shape[0], H.shape[1]))

            # normalize rows
            G = G / G.norm(2, axis=1).reshape((G.shape[0], 1))
            u = TT.ones((G.shape[0], 1))

            # solve problem
            r = li.qr(G.T, mode='r')
            # _, r = li.qr(G.T)

            x = sli.solve(r.T, u)
            b = sli.solve(r, x)
            c = 1./TT.sqrt(TT.sum(b))
            lambda_ = (-c**2) * b
            d = - TT.dot(G.T, lambda_)/c

            self.__combination = d
            self.__info = [c, G.shape]
