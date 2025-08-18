from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "evacuation_model",
        ["evacuation_model.py"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="evacuation_model",
    ext_modules=cythonize(ext_modules),
)
