import setuptools

import os.path

readme = ""
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.rst")
if os.path.exists(readme_path):
    with open(readme_path, "rb") as stream:
        readme = stream.read().decode("utf8")


setuptools.setup(
    name="muler",
    version="0.2.5",
    author="gully",
    author_email="igully@gmail.com",
    description="A Python package for working with data from IGRINS and HPF",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/OttoStruve/muler",
    install_requires=[
        "numpy",
        "scipy",
        "astropy>=4.1,<5.0",
        "specutils>=1.2",
        "pandas",
        "importlib_resources",
        "gwcs<0.17",
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
    python_requires=">=3.7",
)
