from infos.Info import Info
from infos.InfoList import InfoList
from infos.PrefixedInfoElement import PrefixedInfoElement

__author__ = 'giulio'


class InfoGroup(Info):
    @property
    def elements(self):
        l = []
        for e in self.__info_list.elements:
            l.append(PrefixedInfoElement(e, self.__category_descr))
        return l

    def __init__(self, category_descr, info_list: InfoList):
        self.__category_descr = category_descr
        self.__info_list = info_list

    @property
    def descr(self):
        s = self.__category_descr + '=[' + self.__info_list.descr + ']'
        return s
