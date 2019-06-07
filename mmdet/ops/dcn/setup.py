from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
import sys

print( os.environ['TORCH_NVCC_FLAGS'] )

setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ],extra_compile_args=[
		    '-D__CUDA_NO_HALF_OPERATORS__'
		]),
        CUDAExtension('deform_pool_cuda', [
            'src/deform_pool_cuda.cpp', 'src/deform_pool_cuda_kernel.cu'
        ],extra_compile_args=[
		    '-D__CUDA_NO_HALF_OPERATORS__'
		]),
    ],
    cmdclass={'build_ext': BuildExtension})
