from setuptools import setup
from torch.utils import cpp_extension
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy

# print(sys.argv)

setup(
    name="lightconvpoint",
    ext_modules=[],
    cmdclass={}
)

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'eval.src.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'eval/src/utils/libkdtree/pykdtree/kdtree.c',
        'eval/src/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)


# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'eval.src.utils.libmesh.triangle_hash',
    sources=[
        'eval/src/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)


# Gather all extension modules
ext_modules = [
    pykdtree,
    triangle_hash_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
