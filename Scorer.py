import dippykit as dip
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


class Scorer:

    def __init__(self, image, show_results=False):
        self.image = image
        self.show_results = show_results
        self.dct_classes = []

    def add_dct_calc_class(self, dct_class):
        self.dct_classes.append(dct_class)

    def run_all_tests(self):
        dct_results = {}
        for d in self.dct_classes:
            dct_results[str(type(d).__name__)] = self.run_test(d)

        keys = list(dct_results.keys())
        vals = list(dct_results.values())
        keys = [x for _,x in sorted(zip(vals, keys), reverse=True)]
        vals.sort(reverse=True)
        plt.figure()
        plt.title("All DCT Calculators Time")
        plt.xlabel("DCT Method")
        plt.ylabel("Time (s)")
        plt.bar(keys, [k[0] for k in vals])
        plt.show()

    def run_test(self, dct_calculator):
        start = timer()
        F = dct_calculator.perform_dct()
        end = timer()
        elapsed = end - start
        reconstructed = dct_calculator.perform_idct(F)
        reconstructed /= np.max(reconstructed)
        score = self.calc_score(reconstructed)
        print("Time: {}".format(elapsed))
        print("Score: {}".format(score))


        if self.show_results:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(self.image, cmap="Greys_r")
            plt.title("Original")
            plt.xlabel("Time Elapsed: {0:.2f}".format(elapsed))
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed, cmap="Greys_r")
            plt.title(str(type(dct_calculator).__name__))
            plt.xlabel("PSNR: {0:.2f}".format(score))
            plt.show()

        return (elapsed, score)

    def calc_score(self, reconstructed_image):
        return dip.PSNR(self.image, reconstructed_image, np.max(self.image))
