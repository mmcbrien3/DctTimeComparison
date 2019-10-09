from AbstractDctCalculator import AbstractDctCalculator
import scipy.fftpack as scifft

class ScipyDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)

    def perform_dct(self):
        return scifft.dctn(self.image)

