import dippykit as dip
import numpy as np
import copy
from Scorer import Scorer
from NaiveDctCalculator import NaiveDctCalculator
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
    scorer = Scorer(image, show_results=False)
    image = copy.deepcopy(image)
    scorer.add_dct_calc_class(CudaDctCalcualtor(image))
    scorer.add_dct_calc_class(CudaBlockDctCalculator(image))
    scorer.add_dct_calc_class(BlockThreadedDctCalculator(image))
    scorer.add_dct_calc_class(BlockDctCalculator(image))
    scorer.add_dct_calc_class(NaiveDctCalculator(image))
    scorer.add_dct_calc_class(ScipyDctCalculator(image))
    scorer.add_dct_calc_class(NaiveThreadedDctCalculator(image))

    scorer.run_all_tests()

if __name__ == "__main__":
    filepath = "./images/ece_buzz_gray.jpg"
    main(filepath)
