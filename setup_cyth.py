from setuptools import setup
from Cython.Build import cythonize

setup(
    name='muskingum-cunge router',
    ext_modules=cythonize("src/muskingumcunge/route.pyx"),
)

### To build, run:
### python setup_cyth.py build_ext --inplace