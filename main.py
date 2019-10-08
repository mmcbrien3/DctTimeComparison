import dippykit as dip
import numpy as np
from Scorer import Scorer
from LibraryLessDctCalculator import LibraryLessDctCalculator
from NumpyBlockDctCalculator import NumpyBlockDctCalculator
from NumpyDctCalculator import NumpyDctCalculator


def main(file):
    image = open_image(file)
    test_results = run_tests(image)


def open_image(file):
    three_layer_image = dip.im_to_float(dip.im_read(file))
    return np.dot(three_layer_image[..., :3], [0.2989, 0.5870, 0.1140])


def run_tests(image):
    scorer = Scorer(image, show_results=True)
    np_calc = NumpyDctCalculator(image)
    np_block_calc = NumpyBlockDctCalculator(image)

    np_dct = scorer.run_test(np_calc)
    np_block_dct = scorer.run_test(np_block_calc)
    print(np_dct == np_block_dct)


if __name__ == "__main__":
    filepath = "./images/ece_buzz_gray.jpg"
    main(filepath)
