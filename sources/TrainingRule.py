import DescentDirectionRule
import LearningStepRule
import theano as T
from Rule import Rule
from theanoUtils import norm

__author__ = 'giulio'


class TrainingRule(Rule):

    def __init__(self, desc_dir_rule: DescentDirectionRule, lr_rule: LearningStepRule):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule

    def get_train_step_fnc(self, net_symbols, obj_symbols):
        """return a theano compiled function for the training step"""

        W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir, *dir_infos = self.__desc_dir_rule.get_dir(
            net_symbols, obj_symbols)

        lr, *lr_infos = self.__lr_rule.get_lr(net_symbols, obj_symbols, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir)

        gW_rec_norm = norm(obj_symbols.gW_rec)

        # FIXME
        penalty_grad_norm = dir_infos[1]

        output_list = [obj_symbols.grad_norm, penalty_grad_norm, gW_rec_norm, lr] + lr_infos + dir_infos

        return T.function([net_symbols.u, net_symbols.t], output_list,
                          allow_input_downcast='true',
                          on_unused_input='warn',
                          updates=[(net_symbols.W_rec, net_symbols.W_rec + lr * W_rec_dir),
                                   (net_symbols.W_in, net_symbols.W_in + lr * W_in_dir),
                                   (net_symbols.W_out, net_symbols.W_out + lr * W_out_dir),
                                   (net_symbols.b_rec, net_symbols.b_rec + lr * b_rec_dir),
                                   (net_symbols.b_out, net_symbols.b_out + lr * b_out_dir)])

    def format_infos(self, infos):
        desc = 'grad norm: {:07.3f}, gW_rec_norm: {:07.3f}, lr: {:02.4f}'.format(infos[0].item(), infos[2].item(), infos[3].item())
        infos = infos[4:len(infos)]
        lr_desc, infos = self.__lr_rule.format_infos(infos)
        dir_desc, infos = self.__desc_dir_rule.format_infos(infos)
        return desc + ' ' + lr_desc + '' + dir_desc