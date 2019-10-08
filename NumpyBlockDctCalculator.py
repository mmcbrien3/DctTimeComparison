from AbstractDctCalculator import AbstractDctCalculator
import scipy.fftpack as scifft
import numpy as np
import dippykit as dip
import numba as nb
from timeit import default_timer as timer

class NumpyBlockDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)

    @nb.jit
    def perform_dct(self):
        return _perform_dct(self.image, self.M, self.N)

    def perform_idct(self, F):
        return dip.block_process(F, scifft.idctn, (8, 8))


@nb.jit(nopython=True)
def _perform_dct(image, M, N):
    idct = np.zeros((M, N), np.float32)
    for m in np.arange(0, M, 8):
        for n in np.arange(0, N, 8):
            idct[m:m + 8, n:n + 8] = np.fft.fft2(image[m:m + 8, n:n + 8])
    return idct
