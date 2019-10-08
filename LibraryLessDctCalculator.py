from AbstractDctCalculator import AbstractDctCalculator
import math


class LibraryLessDctCalculator(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)

    def perform_dct(self):
        F = [[0] * self.N] * self.M

        for k in range(self.M):
            for l in range(self.N):
                sum = 0
                for m in range(self.M):
                    for n in range(self.N):
                        sum += self.image[m][n] \
                               * math.cos(math.pi / self.M * (m + 0.5) * k) \
                               * math.cos(math.pi / self.N * (n + 0.5) * l)
                F[k][l] = sum

        return F
