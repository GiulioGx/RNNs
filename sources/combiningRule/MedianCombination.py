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


class MedianCombination(CombiningRule):
    def __init__(self, seed=Configs.seed):
        self.__srng = RandomStreams(seed=seed)

    @property
    def infos(self):
        info = SimpleDescription('median_combination')
        return info

    def compile(self, vector_list, n):
        return MedianCombination.Symbols(self, vector_list, n)

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
            v = TT.sort(vector_tensor, axis=0)

            index = TT.cast((vector_tensor.shape[0]/2), 'int64')

            self.__infos = []
            self.__grads_combinantions = v[index]
