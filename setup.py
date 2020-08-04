from setuptools import setup, find_packages

setup(
    name="pyems",
    version="0.1.0",
    author="Matt Huszagh",
    author_email="huszaghmatt@gmail.com",
    description="High-level python interface to OpenEMS with automatic mesh generation",
    license="GPL3",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    url="https://github.com/matthuszagh/pyems",
)
