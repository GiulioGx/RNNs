from infos.Info import Info

__author__ = 'giulio'


class InfoList(Info):

    def __init__(self, *info_list: Info):
        self.__info_list = info_list

    @property
    def elements(self):
        l = []
        for e in self.__info_list:
            l += e.elements
        return l

    @property
    def descr(self):
        s = ''
        t = ''
        for e in self.__info_list:
            d = e.descr
            if d != '':
                s = s + t + e.descr
                t = ', '
        return s