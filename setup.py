import os
import sys
from setuptools import setup, Extension
import nanobind

ext_modules = [
    Extension(
        "xla_mem_bridge",
        [
            "xla_mem_bridge.cc",
            os.path.join(nanobind.include_dir(), "..", "src", "nb_combined.cpp"),
        ],
        include_dirs=[
            nanobind.include_dir(),
            os.path.join(nanobind.include_dir(), "..", "ext", "robin_map", "include"),
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-fPIC'],
        # on macOS, we need this to allow symbols to be resolved at runtime
        extra_link_args=['-Wl,-undefined,dynamic_lookup'] if sys.platform == 'darwin' else [],
    ),
]

setup(
    name="jax-memory-monitor",
    version="0.1.0",
    packages=["jax_memory_monitor"],
    ext_modules=ext_modules,
    install_requires=[
        "jax",
        "nanobind",
    ],
)
