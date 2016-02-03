import theano as T

from infos.SymbolicInfo import NullSymbolicInfos
from lossFunctions import LossFunction
from lossFunctions.SquaredError import SquaredError
from updateRule.UpdateRule import UpdateRule
from updateRule.SimpleUpdate import SimpleUdpate
from descentDirectionRule import DescentDirectionRule
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
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
                 update_rule: UpdateRule = SimpleUdpate(), loss_fnc: LossFunction = SquaredError()):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule
        self.__update_rule = update_rule
        self.__loss_fnc = loss_fnc
        self.__infos = InfoGroup('train_rule', InfoList(InfoGroup('desc_dir', InfoList(self.__desc_dir_rule.infos)),
                                                        InfoGroup('lr_rate', InfoList(self.__lr_rule.infos)),
                                                        InfoGroup('update_rule', InfoList(self.__update_rule.infos))))

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
    def loss_fnc(self):
        return self.__loss_fnc

    class TrainCompiled(object):
        def __init__(self, rule, net):

            self.__separate = True
            net_symbols = net.symbols
            obj_fnc = ObjectiveFunction(rule.loss_fnc, net, net_symbols.current_params, net_symbols.u,
                                        net_symbols.t)
            self.__obj_fnc_symbolic_info = obj_fnc.infos
            direction, self.__dir_symbolic_infos = rule.desc_dir_rule.direction(net_symbols, obj_fnc)

            if not self.__separate:
                lr, self.__lr_symbolic_infos = rule.lr_rule.compute_lr(net, obj_fnc, direction)
                updates, self.__update_symbolic_info = rule.update_rule.compute_update(net, lr, direction)

            else:
                step, self.__lr_symbolic_infos = direction.step_as_direction(rule.lr_rule)
                update_vars = net_symbols.current_params + step
                updates = net_symbols.current_params.update_list(update_vars)
                self.__update_symbolic_info = NullSymbolicInfos()

            output_list = self.__obj_fnc_symbolic_info.symbols + self.__dir_symbolic_infos.symbols + self.__lr_symbolic_infos.symbols + self.__update_symbolic_info.symbols
            self.__step = T.function([net_symbols.u, net_symbols.t], output_list,
                                     allow_input_downcast='true',
                                     on_unused_input='warn',
                                     updates=updates,
                                     name='train_step')

        def step(self, inputs, outputs):
            infos_symbols = self.__step(inputs, outputs)
            infos = self.__format_infos(infos_symbols)
            return infos

        def __format_infos(self, filled_symbols):

            obj_info = self.__obj_fnc_symbolic_info.fill_symbols(filled_symbols)
            start = len(self.__obj_fnc_symbolic_info.symbols)
            dir_info = self.__dir_symbolic_infos.fill_symbols(filled_symbols[start:])
            start += len(self.__dir_symbolic_infos.symbols)
            lr_info = self.__lr_symbolic_infos.fill_symbols(filled_symbols[start:])
            start += len(self.__lr_symbolic_infos.symbols)
            avg_info = self.__update_symbolic_info.fill_symbols(filled_symbols[start:])
            info = InfoList(obj_info, lr_info, dir_info, avg_info)

            return info
