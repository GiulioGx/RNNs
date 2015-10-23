import DescentDirectionRule
import LearningStepRule
import theano as T
import theano.tensor as TT
from ObjectiveFunction import ObjectiveFunction

__author__ = 'giulio'


class TrainingRule(object):
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

            # FIXME
            if len(dir_infos) > 0:
                penalty_grad_norm = dir_infos[1]
            else:
                penalty_grad_norm = TT.alloc(0)

            output_list = [self.__obj_symbols.grad_norm,
                           penalty_grad_norm] + self.__obj_symbols.infos + lr_infos + dir_infos

            new_params = net_symbols.current_params + (self.__dir_symbols.direction * lr)
            self.__step = T.function([net_symbols.u, net_symbols.t], output_list,
                                     allow_input_downcast='true',
                                     on_unused_input='warn',
                                     updates=net_symbols.current_params.update_dictionary(new_params))

        def step(self, inputs, outputs):
            infos = self.__step(inputs, outputs)

            # FIXME
            norm = infos[0]
            penalty_grad_norm = infos[1]
            description = self.__format_infos(infos)

            return description, norm, penalty_grad_norm

        def __format_infos(self, infos):
            infos = infos[2:len(infos)]
            obj_desc, infos = self.__obj_symbols.format_infos(infos)
            lr_desc, infos = self.__lr_symbols.format_infos(infos)
            dir_desc, infos = self.__dir_symbols.format_infos(infos)
            return obj_desc + ' ' + lr_desc + ' ' + dir_desc

    def __init__(self, desc_dir_rule: DescentDirectionRule, lr_rule: LearningStepRule):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule

    def compile(self, net_symbols, obj_symbols):
        return TrainingRule.TrainCompiled(self, net_symbols, obj_symbols)

    @property
    def desc_dir_rule(self):
        return self.__desc_dir_rule

    @property
    def lr_rule(self):
        return self.__lr_rule
