import theano as T

from infos.InfoElement import PrintableInfoElement
from updateRule.UpdateRule import UpdateRule
from updateRule.SimpleUpdate import SimpleUdpate
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
                 update_rule: UpdateRule = SimpleUdpate()):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule
        self.__update_rule = update_rule
        self.__infos = InfoGroup('train_rule', InfoList(InfoGroup('desc_dir', InfoList(self.__desc_dir_rule.infos)),
                                                        InfoGroup('lr_rate', InfoList(self.__lr_rule.infos)),
                                                        InfoGroup('update_rule', InfoList(self.__update_rule.infos))))

    def compile(self, net_symbols, obj_symbols):
        return TrainingRule.TrainCompiled(self, net_symbols, obj_symbols)

    @property
    def desc_dir_rule(self):
        return self.__desc_dir_rule

    @property
    def lr_rule(self):
        return self.__lr_rule

    @property
    def update_rule(self):
        return self.__update_rule

    class TrainCompiled(object):
        def __init__(self, rule, net, obj_fnc: ObjectiveFunction):

            self.__separate = True
            net_symbols = net.symbols
            self.__obj_symbols = obj_fnc.compile(net, net_symbols.current_params, net_symbols.u,
                                                 net_symbols.t)
            self.__dir_symbols = rule.desc_dir_rule.compile(net_symbols, self.__obj_symbols)
            dir_infos = self.__dir_symbols.infos

            if not self.__separate:
                self.__lr_symbols = rule.lr_rule.compile(net, obj_fnc, self.__dir_symbols)
                lr_infos = self.__lr_symbols.infos

                self.__update_symbols = rule.update_rule.compile(net, net_symbols, self.__lr_symbols, self.__dir_symbols)
                update_info = self.__update_symbols.infos
                updates = self.__update_symbols.update_list
                output_list = self.__obj_symbols.infos + lr_infos + dir_infos + update_info

            else:
                step, lr_infos = self.__dir_symbols.direction.step_as_direction(rule.lr_rule)
                update_vars = net_symbols.current_params + step
                diff = (update_vars-net_symbols.current_params).norm(2)
                output_list = self.__obj_symbols.infos + dir_infos + [diff] + lr_infos
                updates = net_symbols.current_params.update_list(update_vars)

            self.__step = T.function([net_symbols.u, net_symbols.t], output_list,
                                     allow_input_downcast='true',
                                     on_unused_input='warn',
                                     updates=updates,
                                     name='train_step')

        def step(self, inputs, outputs):
            infos_symbols = self.__step(inputs, outputs)
            infos = self.__format_infos(infos_symbols)
            return infos

        def __format_infos(self, infos_symbols):
            if not self.__separate:  # XXX
                obj_info, infos_symbols = self.__obj_symbols.format_infos(infos_symbols)
                lr_info, infos_symbols = self.__lr_symbols.format_infos(infos_symbols)
                dir_info, infos_symbols = self.__dir_symbols.format_infos(infos_symbols)
                avg_info, infos_symbols = self.__update_symbols.format_infos(infos_symbols)
                info = InfoList(obj_info, lr_info, dir_info, avg_info)
            else:
                obj_info, infos_symbols = self.__obj_symbols.format_infos(infos_symbols)
                dir_info, infos_symbols = self.__dir_symbols.format_infos(infos_symbols)
                diff_info = PrintableInfoElement('#diff', '', infos_symbols[0])
                lr_info, infos_symbols = self.__dir_symbols.direction.format_lr_info(infos_symbols[1:])
                info = InfoList(obj_info, dir_info, diff_info, lr_info)
            return info
