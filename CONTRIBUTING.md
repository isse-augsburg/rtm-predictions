# Use of packages
Use Python >3.5, best is 3.7.
3.6 has issues with multiprocessing.

# Multi-OS programming
To simplify the usage of the code on many OS, please use the python pathlib whenever you use paths.
For the sake of simplicity: On Windows use `X:\` as mount for `/cfs/home/` and `Y:\` as mount for `/cfs/share`

# Testing
Whenever possible, write simple unittests for you code: use the `unittest` module.
Makes handling a huge project much easier: trying out new things, merging etc.
Pull requests to master will only be accepted with proper testing.

# Documentation
Add documentation to your code: See `Pipeline/erfh5_pipeline.py` as reference.
We use the [sphinx](http://www.sphinx-doc.org/en/master/) for the creation of the documentation.