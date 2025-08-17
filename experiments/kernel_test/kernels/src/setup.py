from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='modelsmith',
    ext_modules=[
        CppExtension(
            name='modelsmith',
            sources=['binary_matmul.cpp', 'kernel_helper.cpp', 'gemm.cpp'],
            extra_compile_args=['-std=c++17']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
