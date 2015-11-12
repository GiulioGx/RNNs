from theano import tensor as TT

from penalty.Penalty import Penalty
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from theanoUtils import norm, get_dir_between_2_dirs

__author__ = 'giulio'


class MidAnglePenaltyDirection(DescentDirectionRule):

    @property
    def infos(self):
        return InfoList(SimpleDescription('mid_angle'), self.__penalty.infos)

    @property
    def penalty(self):
        return self.__penalty

    def compile(self, net_symbols, obj_symbols):
        return MidAnglePenaltyDirection.Symbols(self, net_symbols, obj_symbols)

    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols):
            # add penalty term
            self.__penalty_symbols = rule.penalty.compile(net_symbols.current_params, net_symbols)

            penalty_grad = self.__penalty_symbols.penalty_grad
            penalty_grad_norm = norm(penalty_grad)

            gW_rec = obj_symbols.failsafe_grad.W_rec
            norm_gW_rec = norm(gW_rec)

            min_norm = TT.min([norm_gW_rec, penalty_grad_norm])

            # compute cosine
            cosine = TT.dot(-gW_rec.flatten(), -penalty_grad.flatten()) / (penalty_grad_norm * norm_gW_rec)

            # define descent dir
            t = 0.5  # cosine between dir and penalty_grad
            W_candidate = get_dir_between_2_dirs(- gW_rec, - penalty_grad, t)

            # use antigradient if convenient
            W_candidate = TT.switch(cosine > 0, -gW_rec, W_candidate)  # TODO scegliere diverso da 0

            # normalize W_candidate
            norm_W_candidate = TT.sqrt((W_candidate ** 2).sum())
            W_candidate_scaled = (W_candidate / norm_W_candidate) * min_norm * 0.5  # TODO move constant

            # check for non admissible values
            condition = TT.or_(TT.or_(TT.isnan(penalty_grad_norm),
                                      TT.isinf(penalty_grad_norm)),
                               TT.or_(penalty_grad_norm < 0,
                                      penalty_grad_norm > 1e17))

            W_rec_dir = TT.switch(condition, - gW_rec, W_candidate_scaled)

            # W_rec_dir = -symbol_closet.gW_rec
            # W_rec_dir = - penalty_grad

            # gradient clipping
            c = 1.
            wn = TT.sqrt((W_rec_dir ** 2).sum())
            W_rec_dir = TT.switch(wn > c, c * W_rec_dir / wn, W_rec_dir)

            # compute statistics
            norm_W_rec_dir = TT.sqrt((W_rec_dir ** 2).sum())
            new_cosine_grad = TT.dot(W_rec_dir.flatten(), (- obj_symbols.gW_rec).flatten()) / (
                norm_gW_rec * norm_W_rec_dir)
            new_cosine_pen = TT.dot(W_rec_dir.flatten(), (-penalty_grad).flatten()) / (
                norm_W_rec_dir * penalty_grad_norm)

            tr = TT.sqrt((penalty_grad ** 2).trace())

            a = (TT.dot(W_rec_dir.flatten(), penalty_grad.flatten()) < 0)
            b = (TT.dot(W_rec_dir.flatten(), obj_symbols.gW_rec.flatten()) < 0)
            is_disc_dir = TT.and_(a, b)

            self.__direction = obj_symbols.failsafe_grad() * (-1)
            self.__direction.setW_rec(W_rec_dir)  # FIXME

            self.__infos = self.__penalty_symbols.infos + [cosine, new_cosine_grad, new_cosine_pen, norm_W_rec_dir,
                                                           tr]

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
                PrintableInfoElement('cos(g,p)', ':1.2f', infos_symbols[0].item()),
                PrintableInfoElement('cos(d,p)', ':1.2f', infos_symbols[1].item()),
                PrintableInfoElement('cos(d,p)', ':1.2f', infos_symbols[2].item()),
                PrintableInfoElement('norm_dir', ':07.3f', infos_symbols[3].item()),
                PrintableInfoElement('tr', ':07.3f', infos_symbols[4].item())
            ))

            return info, infos_symbols[info.length:len(infos_symbols)]

    def __init__(self, penalty: Penalty):
        self.__penalty = penalty
