import torch
import time
import unittest
import sys
import os

# Add the ../impls directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../impls')))

from fft import FFT_1d  # Import your FFT_1d class

class TestFFT1D(unittest.TestCase):
    
    def setUp(self):
        self.fft_instance = FFT_1d()
        self.size = 1 << 23  
        self.tolerance = 1e-5  

    def test_fft_precision(self):
        # Create a random complex tensor
        a = torch.randn(self.size, dtype=torch.float32) + 1j * torch.randn(self.size, dtype=torch.float32)
        a = a.to(torch.complex64)

        naive_result = self.fft_instance.naive(a)
        expected_result = torch.fft.fft(a)

        # Check if the results are close enough
        self.assertTrue(torch.allclose(naive_result, expected_result, atol=self.tolerance),
                        msg="FFT result is not within the acceptable tolerance.")

    def test_fft_performance(self):
        # Create a random complex tensor for profiling
        a = torch.randn(self.size, dtype=torch.float32) + 1j * torch.randn(self.size, dtype=torch.float32)
        a = a.to(torch.complex64)

        # Benchmark custom FFT
        start_time_custom = time.time()
        _ = self.fft_instance.naive(a)  # Call the naive method
        end_time_custom = time.time()

        # Benchmark PyTorch's FFT
        start_time_torch = time.time()
        _ = torch.fft.fft(a)  # Call PyTorch's FFT
        end_time_torch = time.time()

        # Benchmark PyTorch's FFT with CUDA (if available)
        if torch.cuda.is_available():
            a_cuda = a.to('cuda')  # Move tensor to GPU
            start_time_cuda = time.time()
            _ = torch.fft.fft(a_cuda)  # Call PyTorch's FFT on CUDA
            end_time_cuda = time.time()
            cuda_time = end_time_cuda - start_time_cuda
        else:
            cuda_time = None

        # Calculate the time taken for each implementation
        custom_time = end_time_custom - start_time_custom
        torch_time = end_time_torch - start_time_torch

        # Print the results in a table format
        print(f"{'Implementation':<30} {'Time (seconds)':<20}")
        print("=" * 50)
        print(f"{'Custom FFT':<30} {custom_time:.6f}")
        print(f"{'Torch FFT':<30} {torch_time:.6f}")
        
        if cuda_time is not None:
            print(f"{'Torch FFT with CUDA':<30} {cuda_time:.6f}")
        
        print("=" * 50)

if __name__ == '__main__':
    unittest.main()
