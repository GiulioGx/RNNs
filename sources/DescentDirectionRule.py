import abc
from Infos import PrintableInfoElement, InfoList, InfoGroup
from Penalty import Penalty, NullPenalty
import theano.tensor as TT
from InfoProducer import InfoProducer
from theanoUtils import norm, cos_between_dirs, get_dir_between_2_dirs

__author__ = 'giulio'


class DescentDirectionSymbols(InfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def direction(self):
        """return a symbol for the computed descent direction"""


class DescentDirectionRule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, symbol_closet, obj_symbols):
        """returns the compiled version"""


class AntiGradient(DescentDirectionRule):
    def __init__(self):
        self.__antigrad_with_pen = AntiGradientWithPenalty(NullPenalty())

    def compile(self, symbol_closet, obj_symbols):
        return self.__antigrad_with_pen.compile(symbol_closet, obj_symbols)


class AntiGradientWithPenalty(DescentDirectionRule):
    class Symbols(DescentDirectionSymbols):
        def __init__(self, rule, net_symbols, obj_symbols):
            # add penalty term
            self.__penalty_symbols = rule.penalty.compile(net_symbols.current_params, net_symbols)
            penalty_grad = self.__penalty_symbols.penalty_grad
            penalty_grad_norm = norm(penalty_grad)

            self.__direction = obj_symbols.grad * (-1)  # FIXME - operator

            W_rec_dir = - obj_symbols.grad.W_rec
            W_rec_dir = TT.switch(penalty_grad_norm > 0, W_rec_dir - rule.penalty_lambda * penalty_grad,
                                  W_rec_dir)

            self.__direction.setW_rec(W_rec_dir)

            self.__infos = self.__penalty_symbols.infos

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            return self.__penalty_symbols.format_infos(infos)

    def __init__(self, penalty: Penalty, penalty_lambda=0.001):
        self.__penalty = penalty
        self.__penalty_lambda = penalty_lambda

    @property
    def penalty(self):
        return self.__penalty

    @property
    def penalty_lambda(self):
        return self.__penalty_lambda

    def compile(self, symbol_closet, obj_symbols):
        return AntiGradientWithPenalty.Symbols(self, symbol_closet, obj_symbols)


class MidAnglePenaltyDirection(DescentDirectionRule):
    class Symbols(DescentDirectionSymbols):
        def __init__(self, rule, net_symbols, obj_symbols):
            # add penalty term
            self.__penalty_symbols = rule.penalty.compile(net_symbols.current_params, net_symbols)

            penalty_grad = self.__penalty_symbols.penalty_grad
            penalty_grad_norm = norm(penalty_grad)

            gW_rec = obj_symbols.grad.W_rec
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

            self.__direction = obj_symbols.grad() * (-1)
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

            return info, infos_symbols[5:len(infos_symbols)]

    def __init__(self, penalty: Penalty):
        self.__penalty = penalty

    @property
    def penalty(self):
        return self.__penalty

    def compile(self, symbol_closet, obj_symbols):
        return MidAnglePenaltyDirection.Symbols(self, symbol_closet, obj_symbols)


class FrozenGradient(DescentDirectionRule):
    class Symbols(DescentDirectionSymbols):
        def __init__(self, rule, net_symbols, obj_symbols):
            # compile penalty
            self.__penalty_symbols = rule.penalty.compile(net_symbols.current_params, net_symbols)

            penalty_grad = self.__penalty_symbols.penalty_grad

            gW_rec = obj_symbols.grad.W_rec
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

            self.__direction = obj_symbols.grad() * (-1)
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
            return info, infos_symbols[2:len(infos_symbols)]

    def __init__(self, penalty: Penalty):
        self.__penalty = penalty

    @property
    def penalty(self):
        return self.__penalty

    def compile(self, symbol_closet, obj_symbols):
        return FrozenGradient.Symbols(self, symbol_closet, obj_symbols)
