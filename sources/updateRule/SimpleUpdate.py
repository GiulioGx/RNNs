from theano.tensor.nlinalg import norm

from infos.Info import Info
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfo import NullSymbolicInfos, SymbolicInfo
from theanoUtils import norm2
from updateRule.UpdateRule import UpdateRule
import theano.tensor as TT


class SimpleUdpate(UpdateRule):
    def __init__(self):
        self.__updates = []

    @property
    def infos(self):
        return SimpleDescription('simple update')

    def compute_update(self, net, lr, direction):
        step = direction * lr
        updated_params = net.symbols.current_params + step
        return updated_params, UpdateInfos(updated_params, step)

    @property
    def updates(self):
        return []


class UpdateInfos(SymbolicInfo):
    def __init__(self, updated_params, step):
        self.__symbols = [TT.max(abs(updated_params.W_rec)), TT.sum(updated_params.W_rec), TT.max(abs(step.W_rec)),
                          TT.max(abs(step.W_in)), TT.max(abs(step.W_out)), TT.max(abs(step.b_rec)), TT.max(abs(step.b_out))]

    @property
    def symbols(self):
        return self.__symbols

    def fill_symbols(self, symbols_replacements: list) -> Info:
        max_w_rec = PrintableInfoElement('max', ':1.2f', symbols_replacements[0].item())
        mean_w_rec = PrintableInfoElement('mean', ':1.2f', symbols_replacements[1].item())
        max_w_rec_s = PrintableInfoElement('max_step_w_rec', ':1.2e', symbols_replacements[2].item())
        max_w_in_s = PrintableInfoElement('max_step_w_in', ':1.2e', symbols_replacements[3].item())
        max_w_out_s = PrintableInfoElement('max_step_w_out', ':1.2e', symbols_replacements[4].item())
        max_b_rec_s = PrintableInfoElement('max_step_b_rec', ':1.2e', symbols_replacements[5].item())
        max_b_out_s = PrintableInfoElement('max_step_b_out', ':1.2e', symbols_replacements[6].item())

        info = InfoGroup("W_rec", InfoList(max_w_rec, mean_w_rec, max_w_rec_s, max_w_in_s, max_w_out_s, max_b_rec_s,
                                           max_b_out_s))
        return info
