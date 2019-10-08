from abc import ABC, abstractmethod
import scipy.fftpack as scifft


class AbstractDctCalculator(ABC):

    def __init__(self, image):
        self.image = image
        self.M = len(self.image)
        self.N = len(self.image[0])
        super().__init__()

    @abstractmethod
    def perform_dct(self):
        pass

    def perform_idct(self, F):
        return scifft.idctn(F)
