from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extension = Extension(
    "BKtree",
    sources=["BKtree.pyx"],
    include_dirs=[numpy.get_include()],
    language="c++"
)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extension),
)
