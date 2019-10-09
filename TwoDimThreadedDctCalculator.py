from AbstractDctCalculator import AbstractDctCalculator
import scipy.fftpack as scifft
import numpy as np
import dippykit as dip
from timeit import default_timer as timer
from threading import Thread, Lock


class TwoDimThreadedDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)
        self.bs = 128

    def perform_dct(self):

        dct_threads = []
        dct = np.zeros((self.M, self.N), np.float64)
        st = timer()
        for m in np.arange(0, self.M, self.bs):
            for n in np.arange(0, self.N, self.bs):
                dct_threads.append(Thread(target=self.threaded_dct_call,
                                          args=(dct, self.image[m:m+self.bs, n:n+self.bs], (m, m+self.bs, n, n+self.bs))))
                dct_threads[-1].start()
        for t in dct_threads:
            t.join()
        return dct

    def threaded_dct_call(self, dct, image_section, indices):
        dct[indices[0]:indices[1], indices[2]:indices[3]] = scifft.dctn(image_section)

    def perform_idct(self, F):
        return np.asarray(dip.block_process(F, scifft.idctn, (self.bs, self.bs)), np.float64)



