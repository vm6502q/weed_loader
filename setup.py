# Adapted from https://github.com/Qiskit/qiskit-aer/blob/master/setup.py

import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py


VERSION = "0.1.0"

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()


class Build(build_py):
    def run(self):
        protoc_command = ["make", "build-deps"]
        if os.name != "nt":
            if subprocess.call(protoc_command) != 0:
                sys.exit(-1)
        super().run()


setup(
    name='weed_loader',
    version=VERSION,
    packages=['weed_loader', 'weed_loader.weed_system'],
    cmdclass={"build_py": Build},
    description="Weed (Loader) - Minimalist AI/ML inference and backprogation",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/vm6502q/weed",
    author="Daniel Strano",
    author_email="stranoj@gmail.com",
    license="MIT",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.5",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.0",
        "Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Machine Learning",
    ],
    keywords="weed ai ml inference gpu opencl",
    install_requires=[],
    setup_requires=[],
    extras_require={},
    include_package_data=True,
    zip_safe=False
)
