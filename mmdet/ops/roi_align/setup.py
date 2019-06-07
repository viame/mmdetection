from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_align_cuda',
    ext_modules=[
        CUDAExtension('roi_align_cuda', [
            'src/roi_align_cuda.cpp',
            'src/roi_align_kernel.cu',
        ],extra_compile_args=[
		    '-D__CUDA_NO_HALF_OPERATORS__'
		]),
    ],
    cmdclass={'build_ext': BuildExtension})
