from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import SimpleDescription
from oldies.AntiGradientWithPenalty import AntiGradientWithPenalty
from penalty.NullPenalty import NullPenalty

__author__ = 'giulio'


class AntiGradient(DescentDirectionRule):

    @property
    def infos(self):
        return SimpleDescription('antigradient')

    def __init__(self):
        self.__antigrad_with_pen = AntiGradientWithPenalty(NullPenalty())

    def compile(self, net_symbols, obj_symbols):
        return self.__antigrad_with_pen.compile(net_symbols, obj_symbols)