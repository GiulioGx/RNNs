from theano.ifelse import ifelse

from infos.Info import Info
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo
import theano.tensor as TT
from training.Rule import Rule


class NanCheckRule(Rule):
    @property
    def updates(self) -> list:
        return []

    @property
    def infos(self):
        return SimpleDescription('Nan and Inf Check')

    def check(self, *tensors_to_check):
        return NanCheckInfo(*tensors_to_check)


class NanCheckInfo(SymbolicInfo):
    def __init__(self, *tensors_to_check):
        self.__symbols = []
        self.__codes = dict()
        self.__codes['1'] = 'nan'
        self.__codes['2'] = 'inf'

        for t in tensors_to_check:
            norm = t.norm(2)
            self.__symbols.append(ifelse(TT.isnan(norm), 1, ifelse(TT.isinf(norm), 2, 0)))

    @property
    def symbols(self):
        return self.__symbols

    def fill_symbols(self, symbols_replacements: list) -> Info:
        info_elements = []
        for i in range(len(self.__symbols)):
            s = symbols_replacements[i]
            if str(s) in self.__codes.keys():
                info_elements.append(PrintableInfoElement('#{}'.format(i), '', self.__codes[str(s)]))
        return InfoList(*info_elements)
