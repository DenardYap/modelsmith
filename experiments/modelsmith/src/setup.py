from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='modelsmith',
    version='0.1.0',
    description='Binary neural network layers and kernels',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='modelsmith._C',
            sources=['binary_matmul.cpp', 'kernel_helper.cpp', 'gemm.cpp'],
            extra_compile_args=['-std=c++17']
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch'],
    python_requires='>=3.8',
)
