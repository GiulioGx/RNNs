import numpy
from infos.Info import Info
import os

__author__ = 'giulio'


class Statistics(object): #TODO maybe use numpt.memmap
    def __init__(self, max_it, check_freq, train_info: Info, net_info:Info):
        self.__dictionary = {}
        self.__check_freq = check_freq
        self.__current_it = 0
        self.__actual_length = 0
        self.__elapsed_time = 0
        self.__m = numpy.ceil(max_it / check_freq) - 1

        self.__train_info = train_info
        self.__net_info = net_info

    def update(self, info: Info, it: int, elapsed_time):

        self.__current_it = it
        self.__actual_length += 1
        self.__elapsed_time = elapsed_time

        for e in info.elements:
            if e.name not in self.__dictionary:
                self.__dictionary[e.name] = []

            self.__dictionary[e.name].append(e.value)
        self.__dictionary['elapsed_time'] = elapsed_time
        self.__dictionary['length'] = self.__actual_length

    @property  # XXX REMOVE or private??
    def dictionary(self):
        return self.__dictionary

    def save(self, filename):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        net_info_dict = self.__net_info.dictionary
        stat_info_dict = self.__dictionary

        stat_info_dict.update(self.__train_info.dictionary)
        stat_info_dict.update(net_info_dict)
        numpy.savez(filename+'.npz', **stat_info_dict)
