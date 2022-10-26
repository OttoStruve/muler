.. _installation:

**********************************
Installing the development version
**********************************


Developer Installation Guide
----------------------------

Currently we recommend installing the developer version from source to help us with beta testing.


To install `muler` from source ::

    git clone https://github.com/OttoStruve/muler.git
    cd muler
    conda env create -f environment.yml
    conda activate muler_dev
    python setup.py develop


The developer version may seem laborious to install, but it has these two perks:
The code changes quickly and bugs are found all the time, so the developer version will always have the latest and greatest features, and should be more reliable.
The developer version allows you to modify the source code yourself, and re-run the code with those changes automatically incorporated: it promotes experimentation!


Installation with pip and conda
-------------------------------

We also provide easier (but less-frequently updated) installation methods with conda-- ::

    conda install -c conda-forge muler

and pip-- ::

    pip install muler

or if you want to install all of the optional extra depedencies  ::

    pip install muler[extra]


Occasionally conda and pip will try to give you old versions when newer versions are available.  Simply pass in the release version you want with `conda install -c conda-forge muler==0.2.5`.



Getting example data
--------------------

HPF and IGRINS do not have public archives (yet) and so their data are not widely available without institutional access or your own local data.  As part of the muler project we have bundled some example data in a separate repository:

https://github.com/OttoStruve/muler_example_data

You can clone that repository and use your local paths to these `.fits`` files when experimenting with `muler` ::

    git clone https://github.com/OttoStruve/muler_example_data.git
    cd muler_example_data

The tutorials assume you have downloadeded this `muler_example_directory` into the same parent directory as the `muler` project-- ::
    
        
        muler
        ├── data
        ├── dist
        ├── docs
        ├── paper
        ├── src
        └── tests
        muler_example_data
        ├── HPF
        ├── IGRINS
        └── Keck_NIRSPEC

If you mimic this directory structure, you should be able to run the tutorials with no code changes.  If you download the `muler_example_data` somewhere else, then you'll have to alter the cells in the Jupyter notebook that define the `filename=` variables.


Running the tests
-----------------

Running the tests also requires some example data, so you'll want to follow the directions above for *Getting example data*.  You can then either copy the data into the `tests/data/` directory, or simlink it-- ::
    
        cd tests/data/
        ln -s ../../../muler_example_data ./muler_example_data

You can then run the tests in the `tests/` directory to double-check that everything installed correctly.  Run pytest from inside the top level directory or `tests/` directory ::

    py.test -vs

Requirements
============

We currently support Python 3.7-3.9, and we have spot-checked that `muler` works with python 3.6 under some conditions.  It works on MacOS, Linux, and Windows.