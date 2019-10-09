from AbstractDctCalculator import AbstractDctCalculator
import numpy as np
from threading import Thread
import scipy.fftpack as scifft
from numba import jit

class OneDimThreadedDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)

    def perform_dct(self):
        size = 512
        num_threads = int(size/64)
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
            dct[r, :] = scifft.dct(vector[r, :])

    def col_threaded_function(self, dct, vector, rows):
        for r in rows:
            dct[:, r] = scifft.dct(vector[:, r])
