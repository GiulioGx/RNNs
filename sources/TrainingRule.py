import theano as T
from combiningRule.CombiningRule import CombiningRule

from descentDirectionRule import DescentDirectionRule
from infos.InfoList import InfoList
from learningRule import LearningRule
from ObjectiveFunction import ObjectiveFunction

__author__ = 'giulio'


class TrainingRule(object):

    def __init__(self, desc_dir_rule: DescentDirectionRule, lr_rule: LearningRule, combining_rule: CombiningRule):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule
        self.__combining_rule = combining_rule

    def compile(self, net_symbols, obj_symbols):
        return TrainingRule.TrainCompiled(self, net_symbols, obj_symbols)

    @property
    def desc_dir_rule(self):
        return self.__desc_dir_rule

    @property
    def lr_rule(self):
        return self.__lr_rule

    @property
    def combining_rule(self):
        return self.__combining_rule

    class TrainCompiled(object):
        def __init__(self, rule, net, obj_fnc: ObjectiveFunction):
            net_symbols = net.symbols
            self.__obj_symbols = obj_fnc.compile(net, net_symbols.current_params, rule.combining_rule, net_symbols.u,
                                                 net_symbols.t)
            self.__dir_symbols = rule.desc_dir_rule.compile(net_symbols, self.__obj_symbols)
            dir_infos = self.__dir_symbols.infos

            self.__lr_symbols = rule.lr_rule.compile(net, obj_fnc, self.__dir_symbols)
            lr_infos = self.__lr_symbols.infos
            lr = self.__lr_symbols.learning_rate

            output_list = self.__obj_symbols.infos + lr_infos + dir_infos

            new_params = net_symbols.current_params + (self.__dir_symbols.direction * lr)
            self.__step = T.function([net_symbols.u, net_symbols.t], output_list,
                                     allow_input_downcast='true',
                                     on_unused_input='warn',
                                     updates=net_symbols.current_params.update_dictionary(new_params))

        def step(self, inputs, outputs):
            infos_symbols = self.__step(inputs, outputs)
            infos = self.__format_infos(infos_symbols)
            return infos

        def __format_infos(self, infos_symbols):
            obj_info, infos_symbols = self.__obj_symbols.format_infos(infos_symbols)
            lr_info, infos_symbols = self.__lr_symbols.format_infos(infos_symbols)
            dir_info, infos_symbols = self.__dir_symbols.format_infos(infos_symbols)
            info = InfoList(obj_info, lr_info, dir_info)
            return info
