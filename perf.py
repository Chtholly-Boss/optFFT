import torch
import kernels_fft
import time

def std_fft_cuda(x):
    
    return torch.fft.fft(x)
class PerfKernel:
    def __init__(self, fft):
        self.size = 1 << 10
        self.x = torch.randn(self.size, device='cpu',dtype=torch.complex64)
        self.myfft = fft
    # count the time of a given function
    def time_count(self, func):
        start = time.time()
        func(self.x)
        end = time.time()
        return end - start
    def perf_time(self):
        # profile fft in pytorch
        t_std_cpu = self.time_count(torch.fft.fft)
        t_naive_gpu = self.time_count(self.myfft)
        # tabular the t
        print(f"std fft cpu: {t_std_cpu}")
        print(f"naive gpu: {t_naive_gpu}")
    def perf_precision(self):
        # profile precision
        # compare the result of std fft and naive gpu
        std_fft = torch.fft.fft(self.x)
        naive_fft = self.myfft(self.x)
        # compare the result
        print(f"diff norm: {torch.norm(std_fft - naive_fft) / torch.norm(std_fft)}")



if __name__ == '__main__':
    my_fft = kernels_fft.FFT_1d()
    perf = PerfKernel(my_fft.naive)
    perf.perf_precision()
    perf.perf_time()