from infos.InfoElement import SimpleDescription
from output_fncs.OutputFunction import OutputFunction


class Linear(OutputFunction):

    def value(self, x):
        return x

    @property
    def infos(self):
        return SimpleDescription('linear_output_function')
