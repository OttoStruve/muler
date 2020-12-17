import setuptools


setuptools.setup(
    name="muler",
    version="0.0.1",
    author="gully",
    author_email="igully@gmail.com",
    description="A Python package for working with data from IGRINS",
    long_description="A Python package for working with echelle spectra from IGRINS",
    long_description_content_type="text/markdown",
    url="https://github.com/OttoStruve/muler",
    install_requires=["numpy", "scipy", "astropy", "specutils", "pandas"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
