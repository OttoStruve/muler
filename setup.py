import setuptools


setuptools.setup(
    name="muler",
    version="0.2.3",
    author="gully",
    author_email="igully@gmail.com",
    description="A Python package for working with data from IGRINS and HPF",
    long_description="A Python package for working with echelle spectra from IGRINS and HPF",
    long_description_content_type="text/markdown",
    url="https://github.com/OttoStruve/muler",
    install_requires=[
        "numpy",
        "scipy",
        "astropy>=4.1",
        "specutils>=1.2",
        "pandas",
        "celerite2",
        "matplotlib",
        "h5py",
    ],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        # If any package contains *.txt files, include them:
        "": ["*.csv"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
