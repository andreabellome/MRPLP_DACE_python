# MRPLPpy

This folder contains scripts to solve the Multi Revolution Perturbed Lambert Problem (MRPLP) using Differential Algebra (DA) techniques in Python. DA techniques are implemented using the library Differential Algebra Computational Toolbox ([DACE](https://github.com/dacelib/dace)). The work is based on the publication by Armellin et al. [[1]](#1).

## Installation

To work on MRPLPpy, one can simply clone the repository in local machine:

```bash
git clone "https://github.com/andreabellome/MRPLP_DACE_python"
```

The toolbox uses common Python libraries: [numpy](https://numpy.org/), [scipy](https://scipy.org/) and [matplotlib](https://matplotlib.org/). If not already downloaded, please use [pip](https://pip.pypa.io/en/stable/) to do so:

```bash
pip install numpy
pip install scipy
pip install matplotlib
```

The toolbox also requires the [DACEyPy](https://pypi.org/project/daceypy/) library, that is a Python wrapper of [DACE](https://github.com/dacelib/dace), the Differential Algebra Computational Toolbox. Please use [pip](https://pip.pypa.io/en/stable/) to install it:

```bash
pip install daceypy
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Currently, only invited developers can contribute to the repository.

## License

The work is under license [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc/4.0/), that is an Attribution Non-Commercial license.

## References
<a id="1">[1]</a> 
Armellin, R., Gondelach, D., & San Juan, J. F. (2018). 
Multiple revolution perturbed Lambert problem solvers.
Journal of Guidance, Control, and Dynamics, 41(9), 2019-2032.
https://doi.org/10.2514/1.G003531