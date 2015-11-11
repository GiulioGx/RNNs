from theano import tensor as TT

from penalty.Penalty import Penalty
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from theanoUtils import norm, cos_between_dirs, get_dir_between_2_dirs

__author__ = 'giulio'


class FrozenGradient(DescentDirectionRule):
    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols):
            # compile penalty
            self.__penalty_symbols = rule.penalty.compile(net_symbols.current_params, net_symbols)

            penalty_grad = self.__penalty_symbols.penalty_grad

            gW_rec = obj_symbols.failsafe_grad.W_rec
            norm_gW_rec = norm(gW_rec)

            froze_condition = norm_gW_rec < 10000

            cos_p_g = cos_between_dirs(-penalty_grad, -gW_rec)
            rotated_pen = get_dir_between_2_dirs(-penalty_grad, -gW_rec, 0.3)  # TODO parameters
            new_dir = TT.switch(cos_p_g >= 0.3, -penalty_grad, rotated_pen * norm(penalty_grad))
            new_dir *= 0.001

            W_rec_dir = TT.switch(froze_condition, new_dir, -gW_rec)

            # gradient clipping
            c = TT.switch(froze_condition, 0.0001, 0.1)
            wn = TT.sqrt((W_rec_dir ** 2).sum())
            W_rec_dir = TT.switch(wn > c, c * W_rec_dir / wn, W_rec_dir)


            # statistics
            cos_d_p = cos_between_dirs(W_rec_dir, -penalty_grad)

            self.__direction = obj_symbols.failsafe_grad() * (-1)
            self.__direction.setW_rec(W_rec_dir)  # FIXME

            self.__infos = self.__penalty_symbols.infos + [froze_condition, cos_d_p]

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos_symbols):
            penalty_info, infos_symbols = self.__penalty_symbols.format_infos(infos_symbols)

            info = InfoGroup('dir', InfoList(
                penalty_info,
                PrintableInfoElement('dir_change', '', infos_symbols[0].item()),
                PrintableInfoElement('cos(d,p)', ':1.2f', infos_symbols[1].item()),
            ))
            return info, infos_symbols[info.length:len(infos_symbols)]

    def __init__(self, penalty: Penalty):
        self.__penalty = penalty

    @property
    def penalty(self):
        return self.__penalty

    def compile(self, net_symbols, obj_symbols):
        return FrozenGradient.Symbols(self, net_symbols, obj_symbols)