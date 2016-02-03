from infos.InfoElement import SimpleDescription
from infos.SymbolicInfo import NullSymbolicInfos
from updateRule.UpdateRule import UpdateRule


class SimpleUdpate(UpdateRule):

    @property
    def infos(self):
        return SimpleDescription('simple update')

    def compute_update(self, net, lr, direction):
        updated_params = net.symbols.current_params + (direction * lr)
        update_list = net.symbols.current_params.update_list(updated_params)
        return update_list, NullSymbolicInfos()
