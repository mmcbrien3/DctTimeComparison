from AbstractDctCalculator import AbstractDctCalculator
import numpy as np
import scipy.fftpack as scifft
from threading import Thread

class NaiveThreadedDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)

    def perform_dct(self):
        size = 512
        num_threads = int(size/32)
        dct = np.zeros((self.M, self.N))
        threads = []

        for r in range(0, size, int(size/num_threads)):
            threads.append(Thread(target=self.row_threaded_function, args=(dct, self.image, np.arange(r, r + int(512 / num_threads)))))
            threads[-1].start()
        for t in threads:
            t.join()
        threads = []

        for r in range(0, size, int(size/num_threads)):
            threads.append(Thread(target=self.col_threaded_function, args=(dct, dct, np.arange(r, r + int(512 / num_threads)))))
            threads[-1].start()
        for t in threads:
            t.join()
        return dct

    def row_threaded_function(self, dct, vector, rows):
        for r in rows:
            dct[r, :] = self.one_dim_dct(vector[r, :])

    def col_threaded_function(self, dct, vector, rows):
        for r in rows:
            dct[:, r] = self.one_dim_dct(vector[:, r])

    def one_dim_dct(self, vector):
        len_vec = len(vector)
        m = np.asarray([np.arange(0, len_vec)])
        k = np.asarray([np.arange(0, len_vec)])
        return np.sum(np.multiply(vector, np.cos(np.pi / len_vec * (m + 1/2) * k.T)), axis=1)

    def official_dct(self, vector):
        return scifft.dct(vector)