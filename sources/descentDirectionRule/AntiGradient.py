from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from descentDirectionRule.AntiGradientWithPenalty import AntiGradientWithPenalty
from penalty.NullPenalty import NullPenalty

__author__ = 'giulio'


class AntiGradient(DescentDirectionRule):
    def __init__(self):
        self.__antigrad_with_pen = AntiGradientWithPenalty(NullPenalty())

    def compile(self, symbol_closet, obj_symbols):
        return self.__antigrad_with_pen.compile(symbol_closet, obj_symbols)