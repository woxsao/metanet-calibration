from setuptools import setup, find_packages

setup(
    name="metanet-calibration",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "pyomo",
        "scipy",
    ],
    author="Monica Chan",
    author_email="mochan@mit.edu",
    description="METANET Network Calibration",
)
