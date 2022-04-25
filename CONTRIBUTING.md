# Contributor Guide

Thank you for your interest in improving this project. This project is
open-source under the MIT License and welcomes contributions in the form of bug
reports, feature requests, and pull requests.

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/OttoStruve/muler/issues).

When filing an issue, you may, answer these questions if you think they are relevant to the issue at hand:

- Which operating system and Python version are you using?
- Which version (or commit SHA) of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a [Minimal, Reproducible
Example](https://stackoverflow.com/help/minimal-reproducible-example).  Arriving at this atomic 
example can take some concentration and time, so it's OK to just flag the issue as a first pass.

## How to request a feature

You are invited to request features on the [Issue
Tracker](https://github.com/OttoStruve/muler/issues).

## How to test the project

```bash
py.test -vs
```

You will need the example data files for this step to work, see our instructions [here](https://muler.readthedocs.io/en/latest/install.html#getting-example-data).

## How to submit changes

Open a [Pull Request](https://github.com/OttoStruve/muler/pulls).

Ideally you should use the [black python code formatter](https://black.readthedocs.io/en/stable/) on any `.py` file you push to a pull request.
This tool standardizes the code formatting so only logical code changes are highlighted in the `git diff`, not mere formatting or whitespace changes.  
This formatting is not currently a requirement since eventually someone else will run `black` on your code.  It's just a nice-to-have, and it may
help standardize your coding style along the way.  We think you will come to adore `black`'s automatic linting if you haven't used it already!

## Making a new release

New releases are currently infrequent enough that gully will take care of them.  Roughly a new release includes these steps:

1. Change the python setup.py file version number
2. Tag a release on GitHub
3. Submit a release to PyPI.
4. Submit a release to conda-forge.

As the release schedule involves more people we may formalize and document this process more.  Thank you for your patience!
