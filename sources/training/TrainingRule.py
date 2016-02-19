import theano as T

from ObjectiveFunction import ObjectiveFunction
from descentDirectionRule import DescentDirectionRule
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from infos.SymbolicInfo import NullSymbolicInfos
from learningRule import LearningRule
from lossFunctions import LossFunction
from lossFunctions.SquaredError import SquaredError
from training.Rule import Rule
from updateRule.SimpleUpdate import SimpleUdpate
from updateRule.UpdateRule import UpdateRule

__author__ = 'giulio'


class TrainingRule(SimpleInfoProducer):
    @property
    def infos(self):
        return self.__infos

    def __str__(self):
        return str(self.__infos)

    def __init__(self, desc_dir_rule: DescentDirectionRule, lr_rule: LearningRule,
                 update_rule: UpdateRule = SimpleUdpate(), loss_fnc: LossFunction = SquaredError()):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule
        self.__update_rule = update_rule
        self.__loss_fnc = loss_fnc
        self.__infos = InfoGroup('train_rule', InfoList(InfoGroup('desc_dir', InfoList(self.__desc_dir_rule.infos)),
                                                        InfoGroup('lr_rate', InfoList(self.__lr_rule.infos)),
                                                        InfoGroup('update_rule', InfoList(self.__update_rule.infos))))

        self.__updates_poller = TrainingRule.UpdatesPoller(lr_rule, desc_dir_rule, update_rule)

    def compile(self, net_symbols):
        return TrainingRule.TrainCompiled(self, net_symbols)

    @property
    def desc_dir_rule(self):
        return self.__desc_dir_rule

    @property
    def lr_rule(self):
        return self.__lr_rule

    @property
    def update_rule(self):
        return self.__update_rule

    @property
    def updates_poller(self):
        return self.__updates_poller

    @property
    def loss_fnc(self):
        return self.__loss_fnc

    class UpdatesPoller(object):

        def __init__(self, *rules: Rule):
            self.__observed_rules = rules

        @property
        def updates(self):
            updates = []
            for rule in self.__observed_rules:
                rule_updates = rule.updates
                assert (isinstance(rule_updates, list))
                updates += rule_updates
            return updates

    class TrainCompiled(object):
        def __init__(self, rule, net):

            self.__separate = False
            self.__symbolic_infos_list = []
            net_symbols = net.symbols
            self.__obj_fnc = ObjectiveFunction(rule.loss_fnc, net, net_symbols.current_params, net_symbols.u,
                                               net_symbols.t)
            obj_fnc_symbolic_info = self.__obj_fnc.infos
            self.__symbolic_infos_list.append(obj_fnc_symbolic_info)
            direction, dir_symbolic_dir_infos = rule.desc_dir_rule.direction(net, self.__obj_fnc)
            self.__symbolic_infos_list.append(dir_symbolic_dir_infos)

            if not self.__separate:
                lr, lr_symbolic_infos = rule.lr_rule.compute_lr(net, self.__obj_fnc, direction)
                update_vars, update_symbolic_info = rule.update_rule.compute_update(net, lr, direction)
                self.__symbolic_infos_list.append(update_symbolic_info)

            else:
                step, lr_symbolic_infos = direction.step_as_direction(rule.lr_rule)
                update_vars = net_symbols.current_params + step
                update_symbolic_info = NullSymbolicInfos()

            self.__symbolic_infos_list.append(lr_symbolic_infos)
            self.__symbolic_infos_list.append(update_symbolic_info)

            network_updates = net_symbols.current_params.update_list(update_vars)
            output_symbol_list = []
            for s in self.__symbolic_infos_list:
                output_symbol_list += s.symbols

            rule_updates = rule.updates_poller.updates

            params = dict(allow_input_downcast='true', on_unused_input='warn', updates=network_updates + rule_updates,
                          name='train_step')
            input_symbol_list = [net_symbols.u, net_symbols.t, rule.loss_fnc.mask]

            self.__step_with_info = T.function(input_symbol_list, output_symbol_list, **params)
            self.__step_without_info = T.function(input_symbol_list, [], **params)

        def step(self, inputs, outputs, mask, report_info: bool = True):
            if report_info:
                infos_symbols = self.__step_with_info(inputs, outputs, mask)
                infos = self.__format_infos(infos_symbols)
                return infos
            else:
                self.__step_without_info(inputs, outputs, mask)
                return

        def __format_infos(self, filled_symbols):

            start = 0
            info_list = []
            for symbolic_info in self.__symbolic_infos_list:
                info_list.append(symbolic_info.fill_symbols(filled_symbols[start:]))
                start += len(symbolic_info.symbols)

            return InfoList(*info_list)

        @property
        def mask(self):
            return self.__obj_fnc.loss_mask
