from averaging.AveragingRule import AveragingRule
from infos.Info import NullInfo
from model import Variables


class NullAveraging(AveragingRule):

    def compile(self, net, update_params: Variables):
        return NullAveraging.Symbols(update_params)

    @property
    def infos(self):
        return NullInfo()

    class Symbols(AveragingRule.Symbols):

        def __init__(self, update_params:Variables):

            self.__avg_vars = update_params

        @property
        def update_list(self):
            return []

        @property
        def averaged_params(self):
            return self.__avg_vars

        @property
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos
