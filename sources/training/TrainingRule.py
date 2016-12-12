import theano as T
import theano.tensor as TT
from ObjectiveFunction import ObjectiveFunction
from descentDirectionRule import DescentDirectionRule
from infos import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from infos.SymbolicInfo import NullSymbolicInfos, SymbolicInfo
from learningRule import LearningStepRule
from lossFunctions import LossFunction
from lossFunctions.SquaredError import SquaredError
from training.NanCheckRule import NanCheckRule
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

    def __init__(self, desc_dir_rule: DescentDirectionRule, lr_rule: LearningStepRule,
                 update_rule: UpdateRule = SimpleUdpate(), loss_fnc: LossFunction = SquaredError(),
                 nan_check: bool = False):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule
        self.__update_rule = update_rule
        self.__loss_fnc = loss_fnc
        self.__nan_check = nan_check
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

    @property
    def nan_check(self):
        return self.__nan_check

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
            self.__symbolic_error_list = []
            net_symbols = net.symbols
            self.__obj_fnc = ObjectiveFunction(rule.loss_fnc, net, net_symbols.current_params, net_symbols.u,
                                               net_symbols.t, net_symbols.mask)
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
                update_vars, update_symbolic_info = rule.update_rule.compute_update(net, 1., direction)
                # update_vars = net_symbols.current_params + step
                update_symbolic_info = NullSymbolicInfos()

            self.__symbolic_infos_list.append(lr_symbolic_infos)
            self.__symbolic_infos_list.append(update_symbolic_info)

            if rule.nan_check:
                nan_check = NanCheckRule()
                to_check = [dict(value=lr, name='lr'), dict(value=direction, name='direction'),
                            dict(value=self.__obj_fnc.current_loss, name='loss'),
                            dict(value=net.symbols.y_shared, name='y')]
                self.__symbolic_error_list.append(nan_check.check(*to_check))  # XXX separate

            add_penalty = False
            lambda_lasso = 0.01
            if add_penalty:
                penalty_grad, penalty_cost = net_symbols.regularization_penalty(net_symbols.u, net_symbols.t, net_symbols.mask,
                                                                  net_symbols.current_params, type="W_rec")
                penalty_grad_norm = penalty_grad.norm(2)
                penalty_symbolic_info = PenaltyInfos(penalty_cost, penalty_grad_norm)
                self.__symbolic_infos_list.append(penalty_symbolic_info)

                c = TT.switch(penalty_grad_norm < 1, 1,  1/penalty_grad_norm)
                update_vars -= penalty_grad * lambda_lasso * c

            network_updates = net_symbols.current_params.update_list(update_vars)

            output_info_symbol_list = TrainingRule.TrainCompiled.get_symbols_list(self.__symbolic_infos_list)
            output_error_symbol_list = TrainingRule.TrainCompiled.get_symbols_list(self.__symbolic_error_list)

            rule_updates = rule.updates_poller.updates

            params = dict(allow_input_downcast='true', on_unused_input='warn', updates=network_updates + rule_updates,
                          name='train_step')
            input_symbol_list = [net_symbols.u, net_symbols.t, net_symbols.mask]

            self.__step_with_info = T.function(input_symbol_list, output_info_symbol_list + output_error_symbol_list,
                                               **params)
            self.__step_without_info = T.function(input_symbol_list, output_error_symbol_list, **params)

        @staticmethod
        def get_symbols_list(symbolic_infos_list):
            output_symbol_list = []
            for s in symbolic_infos_list:
                output_symbol_list += s.symbols
            return output_symbol_list

        def step(self, inputs, outputs, mask, report_info: bool = True):

            start_index = 0
            output_list = []
            if report_info:
                filled_symbols = self.__step_with_info(inputs, outputs, mask)
                infos, start_index = TrainingRule.TrainCompiled.__format_infos(self.__symbolic_infos_list,
                                                                               filled_symbols)
                output_list.append(infos)
            else:
                filled_symbols = self.__step_without_info(inputs, outputs, mask)
            errors, _ = TrainingRule.TrainCompiled.__format_infos(self.__symbolic_error_list, filled_symbols,
                                                                  start_index=start_index)
            output_list.append(errors)
            return output_list if len(output_list) > 1 else output_list[0]

        @staticmethod
        def __format_infos(symbolic_infos_list, filled_symbols, start_index: int = 0):
            start = start_index
            info_list = []
            for symbolic_info in symbolic_infos_list:
                info_list.append(symbolic_info.fill_symbols(filled_symbols[start:]))
                start += len(symbolic_info.symbols)

            return InfoList(*info_list), start

        @property
        def mask(self):
            return self.__obj_fnc.loss_mask


class PenaltyInfos(SymbolicInfo): ##TODO move elsewhere
    def __init__(self, penalty_cost, penalty_grad_norm):
        self.__symbols = [penalty_cost, penalty_grad_norm]

    @property
    def symbols(self):
        return self.__symbols

    def fill_symbols(self, symbols_replacements: list) -> Info:
        penalty_cost_info = PrintableInfoElement('cost', ':1.2f', symbols_replacements[0].item())
        penalty_grad_info = PrintableInfoElement('grad', ':1.2f', symbols_replacements[1].item())

        info = InfoGroup("penalty", InfoList(penalty_cost_info, penalty_grad_info))
        return info
