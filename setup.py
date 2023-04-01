import os
from setuptools import setup, find_packages


PACKAGENAME = "diffsky"
__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "diffsky", "_version.py"
)
with open(pth, "r") as fp:
    exec(fp.read())


setup(
    name=PACKAGENAME,
    version=__version__,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Library for differentiable generation of synthetic skies",
    long_description="Library for differentiable generation of synthetic skies",
    install_requires=["numpy", "jax", "diffmah", "diffstar", "dsps"],
    packages=find_packages(),
    url="https://github.com/ArgonneCPAC/diffsky",
    package_data={"diffsky": ("tests/testing_data/*.dat",)},
)
