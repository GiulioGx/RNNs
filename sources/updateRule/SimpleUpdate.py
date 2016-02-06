from infos.InfoElement import SimpleDescription
from infos.SymbolicInfo import NullSymbolicInfos
from updateRule.UpdateRule import UpdateRule


class SimpleUdpate(UpdateRule):

    def __init__(self):
        self.__updates = []

    @property
    def infos(self):
        return SimpleDescription('simple update')

    def compute_update(self, net, lr, direction):
        updated_params = net.symbols.current_params + (direction * lr)
        return updated_params, NullSymbolicInfos()

    @property
    def updates(self):
        return []
