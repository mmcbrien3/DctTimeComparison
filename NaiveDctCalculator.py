from AbstractDctCalculator import AbstractDctCalculator
import numpy as np
import scipy.fftpack as scifft


class NaiveDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)

    def perform_dct(self):
        dct = np.zeros((self.M, self.N))
        for m in range(self.M):
            dct[m, :] = self.one_dim_dct(self.image[m, :])
        for n in range(self.N):
            dct[:, n] = self.one_dim_dct(dct[:, n])

        return dct

    def one_dim_dct(self, vector):
        len_vec = len(vector)
        m = np.asarray([np.arange(0, len_vec)])
        k = np.asarray([np.arange(0, len_vec)])
        return np.sum(np.multiply(vector, np.cos(np.pi / len_vec * (m + 1/2) * k.T)), axis=1)

    def official_dct(self, vector):
        return scifft.dct(vector)