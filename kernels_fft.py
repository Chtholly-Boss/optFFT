import torch
import myFFT

class FFT_1d:
    def reverse_bits(self, num, bits):
        res = 0
        for i in range(bits):
            res = (res << 1) | (num & 1)
            num >>= 1
        return res
    
    def bit_reverse(self, arr):
        n = arr.shape[0]
        bits = 0 
        while 1 << bits < n:
            bits += 1
        for i in range(n):
            j = self.reverse_bits(i,bits=bits)
            if i < j:
                # swap arr[i] and arr[j]
                tmp = arr[i].clone()
                arr[i] = arr[j].clone()
                arr[j] = tmp
        return arr
    
    def zero_pad(self, a):
        # zero pad a to length 2^log2Ceil(a) 
        n = a.shape[0]
        bits = 0 
        while 1 << bits < n:
            bits += 1
        n = 1 << bits
        res = torch.zeros(n, dtype=torch.complex64)
        res[:a.shape[0]] = a
        return res
    
    def naive(self, a:torch.Tensor):
        if a.dtype != torch.complex64:
            raise ValueError("Input tensor must be of type torch.ciimplex64")
        a = self.zero_pad(a)
        self.bit_reverse(a)
        res = myFFT.naive(a.cuda())
        res = res.cpu()
        return res