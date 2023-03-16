from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rasterizer',
    ext_modules=[
        CUDAExtension('native_rasterizer', [
            'rasterize_cuda.cpp',
            'rasterize_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
