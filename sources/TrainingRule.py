import DescentDirectionRule
import LearningStepRule
import theano as T
from Rule import Rule

__author__ = 'giulio'


class TrainingRule(Rule):
    def __init__(self, desc_dir_rule: DescentDirectionRule, lr_rule: LearningStepRule):
        self.__desc_dir_rule = desc_dir_rule
        self.__lr_rule = lr_rule

    def get_train_step_fnc(self, symbol_closet):
        """return a theano compiled function for the training step"""

        W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir, *dir_infos = self.__desc_dir_rule.get_dir(
            symbol_closet)

        lr, *lr_infos = self.__lr_rule.get_lr(symbol_closet, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir)

        # FIXME
        penalty_grad_norm = dir_infos[1]

        output_list = [symbol_closet.grad_norm, penalty_grad_norm, lr] + lr_infos + dir_infos

        return T.function([symbol_closet.u, symbol_closet.t], output_list,
                          allow_input_downcast='true',
                          on_unused_input='warn',
                          updates=[(symbol_closet.W_rec, symbol_closet.W_rec + lr * W_rec_dir),
                                   (symbol_closet.W_in, symbol_closet.W_in + lr * W_in_dir),
                                   (symbol_closet.W_out, symbol_closet.W_out + lr * W_out_dir),
                                   (symbol_closet.b_rec, symbol_closet.b_rec + lr * b_rec_dir),
                                   (symbol_closet.b_out, symbol_closet.b_out + lr * b_out_dir)])

    def format_infos(self, infos):
        desc = 'lr: {:02.4f}'.format(infos[2].item())
        infos = infos[3:len(infos)]
        lr_desc, infos = self.__lr_rule.format_infos(infos)
        dir_desc, infos = self.__desc_dir_rule.format_infos(infos)
        return desc + ' ' + lr_desc + '' + dir_desc
