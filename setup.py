from setuptools import find_packages, setup

# Setting up
setup(
    name="PVcircuit",
    version="0.0.4",
    author="John Geisz",
    author_email="<john.geiz@NREL.gov>",
    description="Multijunction PV circuit model",
    long_description="Optoelectronic model of tandem and multijunction solar cells",
    url="https://github.com/NREL/PVcircuit",
    license="LICENSE.txt",
    packages=find_packages(),
    install_requires=[
        "ipykernel>=6.15.2",
        "ipympl>=0.7.0",
        "ipywidgets>=7.6.5",
        "matplotlib>=2.1.0",
        "num2words>=0.5.10",
        "numpy>=1.13.3",
        "parse>=1.19.0",
        "pandas>=1.4.0",
        "scipy>=1.0.0",
        "tandems>=0.989",
        "openpyxl>=3.0.10",
        "et_xmlfile>=1.1.0",
        "tqdm>=4.64.0",
    ],
)
