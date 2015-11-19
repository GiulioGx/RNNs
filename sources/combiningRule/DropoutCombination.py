from theano.tensor.shared_randomstreams import RandomStreams
from Configs import Configs
from combiningRule.CombiningRule import CombiningRule
from combiningRule.LinearCombination import LinearCombinationRule
import theano.tensor as TT
import theano as T
from infos.Info import NullInfo
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from theanoUtils import norm

__author__ = 'giulio'


class DropoutCombination(CombiningRule):
    def __init__(self, drop_rate=0.1, seed=Configs.seed):
        self.__srng = RandomStreams(seed=seed)
        self.__drop_rate = drop_rate

    @property
    def infos(self):
        info = InfoGroup('drop_out_combination', InfoList(PrintableInfoElement('rate', ':1.2f', self.__drop_rate)))
        return info

    @property
    def drop_rate(self):
        return self.__drop_rate

    def generate_coeff(self, size):
        return self.__srng.choice(size=size, a=[0, 1], replace=True, p=[self.drop_rate, 1 - self.__drop_rate], dtype=Configs.floatType)

    def compile(self, vector_list, n):
        return DropoutCombination.Symbols(self, vector_list, n)

    class Symbols(CombiningRule.Symbols):
        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            return NullInfo(), infos

        @property
        def combination(self):
            return self.__grads_combinantions

        def __init__(self, rule, vector_list, n):
            vector_tensor = TT.unbroadcast(TT.as_tensor_variable(vector_list), 1)
            size = (n, 1)
            coeffs = TT.addbroadcast(rule.generate_coeff(size), 1)

            values, _ = T.scan(lambda v, c, acc: v * c + acc,
                               sequences=[vector_list, coeffs],
                               outputs_info=[TT.unbroadcast(TT.zeros_like(vector_list[0]), 1)],
                               non_sequences=[],
                               name='dropout_scan',
                               n_steps=n)

            self.__infos = []
            self.__grads_combinantions = values[-1]

            # size = (n, 1)
            # coeffs = TT.addbroadcast(rule.generate_coeff(size), 1)
