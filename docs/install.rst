.. _installation:

**********************************
Installing the development version
**********************************




.. note::

    Conda installation is not yet available.

Pip installation is currently experimental, but should technically work ::

    pip install muler

Currently we recommend installing the developer version from source to help us with beta testing.


To install `muler` from source ::

    git clone https://github.com/OttoStruve/muler.git
    cd muler
    conda env create -f environment.yml
    conda activate muler_dev
    python setup.py develop


You can run the tests in the `tests/` directory to double-check that everything installed correctly::

    py.test -vs



Requirements
============

The project may work with a variety of Python 3 minor versions. We have tested it on both Linux and Windows Operating Systems.
