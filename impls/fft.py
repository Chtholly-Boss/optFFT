import torch

class FFT_1d:
    def naive(self, a:torch.Tensor):
        if a.dtype != torch.complex64:
            raise ValueError("Input tensor must be of type torch.complex64")
        res = torch.fft.fft(a)
        return res