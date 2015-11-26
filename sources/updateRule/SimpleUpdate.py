from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import SimpleDescription
from learningRule.LearningRule import LearningStepRule
from updateRule.UpdateRule import UpdateRule
from infos.Info import NullInfo


class SimpleUdpate(UpdateRule):
    def compile(self, net, net_symbols, lr_symbols: LearningStepRule.Symbols,
                dir_symbols: DescentDirectionRule.Symbols):
        return SimpleUdpate.Symbols(net_symbols, lr_symbols, dir_symbols)

    @property
    def infos(self):
        return SimpleDescription('simple update')

    class Symbols(UpdateRule.Symbols):
        def __init__(self, net_symbols, lr_symbols: LearningStepRule.Symbols,
                     dir_symbols: DescentDirectionRule.Symbols):
            updated_params = net_symbols.current_params + (dir_symbols.direction * lr_symbols.learning_rate)
            self.__update_list = net_symbols.current_params.update_list(updated_params)

        @property
        def update_list(self):
            return self.__update_list

        @property
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos
