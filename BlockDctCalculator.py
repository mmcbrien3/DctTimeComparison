from AbstractDctCalculator import AbstractDctCalculator
import scipy.fftpack as scifft
import numpy as np
import dippykit as dip
from timeit import default_timer as timer
from threading import Thread, Lock


class BlockDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)
        self.bs = 32

    def perform_dct(self):
        dct = np.zeros((self.M, self.N))
        for m in range(0, self.M, self.bs):
            for n in range(0, self.N, self.bs):
                dct[m:m+self.bs, n:n+self.bs] = self.two_dim_dct(self.image[m:m+self.bs, n:n+self.bs])
        return dct

    def two_dim_dct(self, matrix):
        dct = np.zeros(matrix.shape)
        for m in range(matrix.shape[0]):
            dct[m, :] = self.one_dim_dct(matrix[m, :])
        for n in range(matrix.shape[1]):
            dct[:, n] = self.one_dim_dct(dct[:, n])
        return dct

    def one_dim_dct(self, vector):
        len_vec = len(vector)
        m = np.asarray([np.arange(0, len_vec)])
        return np.sum(np.multiply(vector, np.cos(np.pi / len_vec * (m + 1 / 2) * m.T)), axis=1)

    def perform_idct(self, F):
        return np.asarray(dip.block_process(F, scifft.idctn, (self.bs, self.bs)), np.float64)

