from setuptools import setup, find_packages

setup(
    name="samosa",
    version="0.2.0",
    author="Sanjan Muchandimath",
    description="MCMC package with ML-MCMC functionalities",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "mpi4py>=3.0",
    ],
    extras_require={
        "test": ["pytest>=7.0"],
        "maps": ["MParT", "normflows", "torch>=1.9"],
    },
)
