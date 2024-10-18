from setuptools import find_packages
from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='locoman',
    version='1.0.0',
    author='Changyi Lin',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='changyil@andrew.cmu.edu',
    description='Toolkit for deployment of loco-manipulation algorithms on the Unitree Go1.',
    install_requires=required,
)
