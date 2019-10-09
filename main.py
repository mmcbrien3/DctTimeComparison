import dippykit as dip
import numpy as np
from Scorer import Scorer
from NaiveDctCalculator import NaiveDctCalculator
from TwoDimThreadedDctCalculator import TwoDimThreadedDctCalculator
from OneDimThreadedDctCalculator import OneDimThreadedDctCalculator
from ScipyDctCalculator import ScipyDctCalculator
from NaiveThreadedDctCalculator import NaiveThreadedDctCalculator

def main(file):
    image = open_image(file)
    test_results = run_tests(image)


def open_image(file):
    three_layer_image = dip.im_to_float(dip.im_read(file))
    return np.dot(three_layer_image[..., :3], [0.2989, 0.5870, 0.1140])


def run_tests(image):
    scorer = Scorer(image, show_results=True)
    one_dim_calc = OneDimThreadedDctCalculator(image)
    two_dim_calc = TwoDimThreadedDctCalculator(image)
    naive_calc = NaiveDctCalculator(image)
    scipy_calc = ScipyDctCalculator(image)
    naive_threaded_calc = NaiveThreadedDctCalculator(image)

    # scorer.run_test(scipy_calc)
    # scorer.run_test(one_dim_calc)
    # scorer.run_test(two_dim_calc)
    scorer.run_test(naive_threaded_calc)
    scorer.run_test(naive_calc)

if __name__ == "__main__":
    filepath = "./images/ece_buzz_gray.jpg"
    main(filepath)
