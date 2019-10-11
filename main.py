import dippykit as dip
import numpy as np
import copy
from Scorer import Scorer
from NaiveDctCalculator import NaiveDctCalculator
from TwoDimThreadedDctCalculator import TwoDimThreadedDctCalculator
from OneDimThreadedDctCalculator import OneDimThreadedDctCalculator
from ScipyDctCalculator import ScipyDctCalculator
from NaiveThreadedDctCalculator import NaiveThreadedDctCalculator
from BlockThreadedDctCalculator import BlockThreadedDctCalculator
from BlockDctCalculator import BlockDctCalculator
from CudaDctCalculator import CudaDctCalcualtor
from CudaBlockDctCalculator import CudaBlockDctCalculator

def main(file):
    image = open_image(file)
    test_results = run_tests(image)


def open_image(file):
    three_layer_image = dip.im_to_float(dip.im_read(file))
    return np.dot(three_layer_image[..., :3], [0.2989, 0.5870, 0.1140])


def run_tests(image):
    scorer = Scorer(image, show_results=True)
    image = copy.deepcopy(image)
    cuda_calc = CudaDctCalcualtor(image)
    cuda_block_calc = CudaBlockDctCalculator(image)
    block_threaded_calc = BlockThreadedDctCalculator(image)
    # block_calc = BlockDctCalculator(image)
    # naive_calc = NaiveDctCalculator(image)
    # scipy_calc = ScipyDctCalculator(image)
    # naive_threaded_calc = NaiveThreadedDctCalculator(image)

    scorer.run_test(cuda_calc)
    scorer.run_test(cuda_block_calc)
    scorer.run_test(block_threaded_calc)
    # scorer.run_test(block_calc)
    # scorer.run_test(scipy_calc)
    # scorer.run_test(naive_threaded_calc)
    # scorer.run_test(naive_calc)

if __name__ == "__main__":
    filepath = "./images/ece_buzz_gray.jpg"
    main(filepath)
