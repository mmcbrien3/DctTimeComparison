from AbstractDctCalculator import AbstractDctCalculator
import numpy as np
from numba import cuda
import math
import os

os.environ['NUMBAPRO_NVVM'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\nvvm\bin\nvvm64_33_0.dll'
os.environ['NUMBAPRO_LIBDEVICE'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\nvvm\libdevice'

class CudaDctCalcualtor(AbstractDctCalculator):

    def __init__(self, image):
        super().__init__(image)

    def perform_dct(self):
        dct = np.zeros((self.M, self.N))
        dct_out = np.zeros((self.M, self.N))
        threadsperblock = 256
        blockspergrid = (dct.size + (threadsperblock - 1)) // threadsperblock
        print(blockspergrid)
        self.kernel_function[blockspergrid, threadsperblock](dct, self.image)
        self.kernel_function[blockspergrid, threadsperblock](dct_out, dct)

        return dct_out

    @staticmethod
    @cuda.jit
    def kernel_function(dct_array, image_array):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x
        blocks_per_row = 512 // bw
        col = tx + (bw * (blocks_per_row - (ty % blocks_per_row) - 1))
        row = (ty * bw - 1) // 512
        my_array = image_array[row, :]
        sum = 0
        for i in range(len(my_array)):
            sum += my_array[i] * math.cos(col * (i + 1/2) * math.pi / 512)
        dct_array[col, row] = sum
