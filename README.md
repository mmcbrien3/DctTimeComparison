# DctTimeComparison

This project attempts to implement different methods of computing the DCT of a 512x512 image.
The different methods focus on implementing different levels of parallelism (e.g. blocking, threading, GPU threading).
The ultimate comparison is SciPack's implementation of the 2D DCT, which is quite quick.

## Current Results
![RESULTS][results]

[results]: ./images/all_dct_calcs_comparison.png