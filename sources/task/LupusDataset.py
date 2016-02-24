import numpy
from scipy.io import loadmat

from Configs import Configs
from Paths import Paths
from infos.InfoElement import SimpleDescription
from task.Dataset import Dataset


class LupusDataset(Dataset):
    def __init__(self, mat_file: str, seed: int = Configs.seed):
        mat_obj = loadmat(mat_file)
        print(mat_obj.keys())

        self.__rng = numpy.random.RandomState(seed)
        self.__n_in = 0
        self.__n_out = 0

    def get_train_batch(self, batch_size: int):
        return 'None'

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def infos(self):
        return SimpleDescription('Lupus Dataset')


if __name__ == '__main__':
    dataset = LupusDataset(Paths.lupus_path)
    batch = dataset.get_train_batch(batch_size=1)
    print(str(batch))
