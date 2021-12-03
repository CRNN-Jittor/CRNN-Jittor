from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os
from config import curr_path

extension = Extension("BKtree",
                      sources=[os.path.join(curr_path, "BKtree.pyx")],
                      include_dirs=[numpy.get_include()],
                      language="c++")

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extension),
)
