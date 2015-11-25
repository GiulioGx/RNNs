import theano as T

from averaging.AveragingRule import AveragingRule
from averaging.NullAveraging import NullAveraging
from descentDirectionRule import DescentDirectionRule
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SimpleInfoProducer import SimpleInfoProducer
from learningRule import LearningRule
from ObjectiveFunction import ObjectiveFunction

__author__ = 'giulio'


class TrainingRule(SimpleInfoProducer):
    @property
    def infos(self):
        return self.__infos

    def __str__(self):
        return str(self.__infos)

    def __init__(self, desc_dir_rule: DescentDirectionRule, lr_rule: LearningRule,
                 avg_rule: AveragingRule = NullAveraging()):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule
        self.__avg_rule = avg_rule
        self.__infos = InfoGroup('train_rule', InfoList(InfoGroup('desc_dir', InfoList(self.__desc_dir_rule.infos)),
                                                        InfoGroup('lr_rate', InfoList(self.__lr_rule.infos)),
                                                        InfoGroup('avg_rule', InfoList(self.__avg_rule.infos))))

    def compile(self, net_symbols, obj_symbols):
        return TrainingRule.TrainCompiled(self, net_symbols, obj_symbols)

    @property
    def desc_dir_rule(self):
        return self.__desc_dir_rule

    @property
    def lr_rule(self):
        return self.__lr_rule

    @property
    def avg_rule(self):
        return self.__avg_rule

    class TrainCompiled(object):
        def __init__(self, rule, net, obj_fnc: ObjectiveFunction):
            net_symbols = net.symbols
            self.__obj_symbols = obj_fnc.compile(net, net_symbols.current_params, net_symbols.u,
                                                 net_symbols.t)
            self.__dir_symbols = rule.desc_dir_rule.compile(net_symbols, self.__obj_symbols)
            dir_infos = self.__dir_symbols.infos

            self.__lr_symbols = rule.lr_rule.compile(net, obj_fnc, self.__dir_symbols)
            lr_infos = self.__lr_symbols.infos
            lr = self.__lr_symbols.learning_rate

            new_params = net_symbols.current_params + (self.__dir_symbols.direction * lr)

            # averaging
            self.__avg_symbols = rule.avg_rule.compile(net, new_params)
            new_params = self.__avg_symbols.averaged_params
            avg_updates = self.__avg_symbols.update_list

            avg_info = self.__avg_symbols.infos

            train_updates = net_symbols.current_params.update_list(new_params)
            train_updates += avg_updates

            output_list = self.__obj_symbols.infos + lr_infos + dir_infos + avg_info

            self.__step = T.function([net_symbols.u, net_symbols.t], output_list,
                                     allow_input_downcast='true',
                                     on_unused_input='warn',
                                     updates=train_updates,
                                     name='train_step')

        def step(self, inputs, outputs):
            infos_symbols = self.__step(inputs, outputs)
            infos = self.__format_infos(infos_symbols)
            return infos

        def __format_infos(self, infos_symbols):
            obj_info, infos_symbols = self.__obj_symbols.format_infos(infos_symbols)
            lr_info, infos_symbols = self.__lr_symbols.format_infos(infos_symbols)
            dir_info, infos_symbols = self.__dir_symbols.format_infos(infos_symbols)
            avg_info, infos_symbols = self.__avg_symbols.format_infos(infos_symbols)
            info = InfoList(obj_info, lr_info, dir_info, avg_info)
            return info
