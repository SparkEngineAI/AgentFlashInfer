from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "samfast._samfast",
        ["samfast/samfast.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    )
]

setup(
    name="samfast",
    version="0.1.0",
    description="GIL-free SAM implemented in C++ with pybind11",
    packages=["samfast"],
    ext_modules=ext_modules,
)
