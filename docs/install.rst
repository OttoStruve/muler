.. _installation:

**********************************
Installing the development version
**********************************




.. note::

    Conda and pip are not yet available, check back soon


Currently only the developer version is available for beta testing.


To install `muler` from source ::

    $ git clone https://github.com/OttoStruve/muler.git
    $ cd muler
    $ conda env create -f environment.yml
    $ conda activate muler_dev
    $ python setup.py develop


You can run the tests in `tests` to double-check ::

    $ py.test -vs



Requirements
============

The project may work with a variety of Python 3 minor versions.  Python 2 is not supported.
