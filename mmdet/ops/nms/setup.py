import os.path as osp
import os,re,sys,subprocess,copy

from os.path import join as pjoin
from setuptools import setup, Extension

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_args = dict(
    include_dirs=[np.get_include()],
    language='c++',
    extra_compile_args={
        'cc': [],
        'nvcc': ['-c', '--compiler-options', '-fPIC', '-D__CUDA_NO_HALF_OPERATORS__'],
    },
)

extensions = [
    Extension('soft_nms_cpu', ['src/soft_nms_cpu.pyx'], **ext_args),
]

def _is_cuda_file(path):
    return os.path.splitext(path)[1] in ['.cu', '.cuh']

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to cc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    print('self.compiler_type ',self._compile)
    self.src_extensions+= ['.cu', '.cuh']
    self._cpp_extensions += ['.cu', '.cuh']
    original_compile = self.compile
    original_spawn = self.spawn
    # save references to the default compiler_so and _comple methods
    #default_compiler_so = self.compiler_so
    super = self._compile
    def win_wrap_compile(sources,
                        output_dir=None,
                        macros=None,
                        include_dirs=None,
                        debug=0,
                        extra_preargs=None,
                        extra_postargs=None,
                        depends=None):

        self.cflags = copy.deepcopy(extra_postargs)
        #print(self.cflags)
        extra_postargs = None

        def spawn(cmd):
            orig_cmd = cmd
            # Using regex to match src, obj and include files

            src_regex = re.compile('/T(p|c)(.*)')
            src_list = [
                m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                if m
            ]

            obj_regex = re.compile('/Fo(.*)')
            obj_list = [
                m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                if m
            ]

            include_regex = re.compile(r'((\-|\/)I.*)')
            include_list = [
                m.group(1)
                for m in (include_regex.match(elem) for elem in cmd) if m
            ]

            if len(src_list) >= 1 and len(obj_list) >= 1:
                src = src_list[0]
                obj = obj_list[0]
                if _is_cuda_file(src):
                    print('compile cuda file--------------------------------')
                    nvcc = _join_cuda_home('bin', 'nvcc')
                    if isinstance(self.cflags, dict):
                        cflags = self.cflags['nvcc']
                    elif isinstance(self.cflags, list):
                        cflags = self.cflags
                    else:
                        cflags = []
                    cmd = [
                        nvcc, '-c', src, '-o', obj, '-Xcompiler',
                        '/wd4819', '-Xcompiler', '/MD'
                    ] + include_list + cflags
                    #print(cmd)
                elif isinstance(self.cflags, dict):
                    print('compile cpp file--------------------------------')
                    cflags = self.cflags['cc']
                    cmd += cflags
                elif isinstance(self.cflags, list):
                    cflags = self.cflags
                    cmd += cflags

            return original_spawn(cmd)

        try:
            self.spawn = spawn
            return original_compile(sources, output_dir, macros,
                                    include_dirs, debug, extra_preargs,
                                    extra_postargs, depends)
        finally:
            self.spawn = original_spawn
    if self.compiler_type == 'msvc':
        self.compile = win_wrap_compile
        print('c===========================================')
        print(self.compile)

class custom_build_ext(build_ext):

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


setup(
    name='soft_nms',
    cmdclass={'build_ext': custom_build_ext},
    ext_modules=cythonize(extensions),
)

setup(
    name='nms_cuda',
    ext_modules=[
        CUDAExtension('nms_cuda', [
            'src/nms_cuda.cpp',
            'src/nms_kernel.cu',
        ],extra_compile_args=[
		    '-D__CUDA_NO_HALF_OPERATORS__'
		]),
        CUDAExtension('nms_cpu', [
            'src/nms_cpu.cpp',
        ],extra_compile_args=[
		    '-D__CUDA_NO_HALF_OPERATORS__'
		]),
    ],
    cmdclass={'build_ext': BuildExtension})
