from setuptools import setup
from torch.utils.cpp_extension import CppExtension,CUDAExtension,BuildExtension

setup(name="MyFFT",
      version="0.0.1",
      ext_modules=[
          CUDAExtension("myFFT", ["myFFT.cu"])],
      cmdclass={'build_ext': BuildExtension})