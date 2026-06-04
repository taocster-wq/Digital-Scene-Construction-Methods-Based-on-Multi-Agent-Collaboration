from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        name="ssrm",
        sources=["ssrm.py"],
    )
]


setup(
    name="ssrm",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
        },
    ),
)