# FFT Optimizations
Implementations see:
* [python-implementation](./fft.ipynb)
## Get Started
Following posts may be helpful to get started:
* [An-Excursion-Into-FFT-p1](https://mecha-mind.medium.com/an-excursion-into-fast-fourier-transform-part-1-8a6498ee0c10)
    : This post gives an intro to FFT from the perspective of Time-Series
* [An-Excursion-Into-FFT-p2](https://mecha-mind.medium.com/an-excursion-into-fast-fourier-transform-part-2-81461f125880)
    : This post gives other applications of FFT like:
        * Polynomial Multiplication
        * Combinational Sum
        * Pattern Matching
* [An-Excursion-Into-FFT-p3](https://mecha-mind.medium.com/fast-fourier-transform-optimizations-5c1fd108a8ed)
    : This post gives:
        * An optimization of Bit-reverse Operation 
        * Bailey's FFT Algorithm

You can also refer to CLRS(3rd Edition) Chapter 30 to get started.

## One Step Further
* [FFT-Kernel-Optimization](https://zhuanlan.zhihu.com/p/389325484)
    : This post from **ZhiHu** depicts the FFT picture well
    * It introduces high-radix Cooley-Turkey Algorithm
    * also cover inputs whose length is not power of 2
