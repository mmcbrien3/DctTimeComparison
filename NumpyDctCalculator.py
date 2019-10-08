from AbstractDctCalculator import AbstractDctCalculator
import scipy.fftpack as scifft
import dippykit as dip
from numba import jit


class NumpyDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)

    @jit(forceobj=True)
    def perform_dct(self):
        return scifft.dctn(self.image)

