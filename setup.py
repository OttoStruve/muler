import setuptools

import os.path

readme = ""
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "rb") as stream:
        readme = stream.read().decode("utf8")


setuptools.setup(
    name="muler",
    version="0.4.0",
    author="gully",
    author_email="igully@gmail.com",
    description="A Python package for working with data from various echelle spectrographs",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/OttoStruve/muler",
    install_requires=[
        "numpy",
        "scipy",
        "astropy>=5.2",
        "specutils>=1.9",
        "pandas",
        "importlib_resources",
        "matplotlib",
    ],
    extras_require={"extra": ["celerite2", "h5py", "black"]},
    packages=setuptools.find_packages(where="src", exclude=["data/*, paper/*"]),
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
    python_requires=">=3.8",
)
