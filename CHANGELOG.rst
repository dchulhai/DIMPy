#########
Changelog
#########

All notable changes to this project will be documented in this file.


Unreleased
##########

Features
--------

- DIM with retardation and PBC
- DIM/QM with PySCF

Fixes
-----

- AtomParam 'rad' in atomic units

[0.8.0] - 2022-01-21
####################

Added
-----

- Use NumExpr to help speed up some array operations
- DIM with retardation (Meredith's work)

Fixes
-----

- fixed analyzer method for molecular nanoparticles (ndim = 0)
- Removed CG solver as an option (incorrect results)


[0.7.5] - 2022-01-10
####################

Added
-----

- Included use of numpy for simple shared-memory parallelization

[0.7.4] - 2022-01-10
####################

Added
-----

- Included updated documentation on some modules

[0.7.3] - 2021-07-02
####################

Changed
-------

- PBC calculations no longer has their own modules
- Each module in DIMPy/dimpy/methods only modifies the T2 tensor

[0.7.2] - 2021-07-01
####################

Fixed
-----

- Updated test/examples to account for new 'kdir' input

[0.7.1] - 2021-07-01
####################

Fixed
-----

- 'Kdir' key as a vector instead of a string

[0.7.0] - 2021-06-30
####################

Added
-----

- DIM with PBC (including examples and test cases)

[0.6.0] - 2021-06-29
####################

Added
-----

- DIM
- Analyzer

Changed
-------

- analyzer, input_file, methods, nanoparticle, tools modules are now
  their own subpackages of DIMPy


[0.5.0] - 2021-06-18
####################

Added
-----

- Test cases
- Documentation
- CHANGELOG.rst

[pre 0.5.0]
###########

Added
-----

- DDA
- DDA with retardation
- DDA with PBC
- DDA with PBC and retardation

