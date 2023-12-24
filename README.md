# MRPLP using DACE

This folder contains script to solve the Multi Revolution Perturbed Lambert Problem (MRPLP) using Differential Algebra (DA) techniques. DA techniques are implemented using the library Differential Algebra Computational Toolbox ([DACE](https://github.com/dacelib/dace)). The work is based on the publication by Armellin et al. [[1]](#1).

## Installation

One can clone the repository in local. The toolbox uses common Python libraries: [numpy](https://numpy.org/), [scipy](https://scipy.org/) and [matplotlib](https://matplotlib.org/). If not already downloaded, please use [pip](https://pip.pypa.io/en/stable/) to do so:

```bash
pip install numpy
pip install scipy
pip install matplotlib
```

The toolbox also requires the [DACEyPy](https://pypi.org/project/daceypy/) library, that is a Python wrapper of DACE, the Differential Algebra Computational Toolbox. Please use [pip](https://pip.pypa.io/en/stable/) to install it:

```bash
pip install daceypy
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
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

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## References
<a id="1">[1]</a> 
Armellin, R., Gondelach, D., & San Juan, J. F. (2018). 
Multiple revolution perturbed Lambert problem solvers.
Journal of Guidance, Control, and Dynamics, 41(9), 2019-2032.
https://doi.org/10.2514/1.G003531