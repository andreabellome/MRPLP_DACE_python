# MRPLPpy

This folder contains scripts to solve the Multi Revolution Perturbed Lambert Problem (MRPLP) using Differential Algebra (DA) techniques in Python. DA techniques are implemented using the library Differential Algebra Computational Toolbox ([DACE](https://github.com/dacelib/dace)). The work is based on the publication by Armellin et al. [[1]](#1).

## Installation

To work with MRPLPpy, one can simply clone the repository in the local machine:

```bash
git clone "https://github.com/andreabellome/MRPLP_DACE_python"
```

The toolbox uses common Python libraries: [numpy](https://numpy.org/), [scipy](https://scipy.org/) and [matplotlib](https://matplotlib.org/). If not already downloaded, please use [pip](https://pip.pypa.io/en/stable/) to do so:

```bash
pip install numpy
pip install scipy
pip install matplotlib
```

The toolbox also requires the [DACEyPy](https://pypi.org/project/daceypy/) library, that is a Python wrapper of [DACE](https://github.com/dacelib/dace). Please use [pip](https://pip.pypa.io/en/stable/) to install it:

```bash
pip install daceypy
```

To cite this work, refer to Armellin et al. [[1]](#1).

## Usage and test cases

To use the repository, one finds two different test scripts.

### Test script 1: solving the MRPLP using DACE

The reference script is: [final_mrplp_j2_analytic.py](https://github.com/andreabellome/MRPLP_DACE_python/blob/main/final_mrplp_j2_analytic.py). This script is used to solve the MRPLP using DA techniques. It all starts by including the required libraries:

```python
from functions.MRPLP_J2_analytic import MultiRevolutionPerturbedLambertSolver
from functions.expansion_perturbed_lambert import ExpansionPerturbedLambert
```

that are used to access classes to solve the MRPLP and to expand the solution.

One then loads common Python libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
```

The constants of motion are defined, as well as initial and final states and time of flight to solve the MRPLP:

```python
# define the constant of motion for the central body (Earth in this case)
mu = 398600.4418  # km^3/s^2
J2 = 1.08262668e-3
rE = 6378.137 # km

# initial guess, target position and time of flight
rr1  = np.array( [-3173.91245750977, -1863.35865746, -6099.31199561] ) # initial position - (km)
vv1 = np.array( [-6.37541145277431, -1.26857476842513, 3.70632783068748] ) # initial velocity - (km/s)
rr2  = np.array( [6306.80758519, 3249.39062728,  794.06530085] ) # final position - (km)
vv2 = np.array( [1.1771075218004, -0.585047636781159, -7.370399738227] ) # final velocity - (km/s)

vv1g = vv1 # it should work also with a very brutal first guess --> in this case the initial velocity
tof = 1.5*3600.0 # time of flight (s)
```

The class should be initialised and the input parameters are defined:

```python
# initialise the classes for MRPLP solver and expansion of perturbed Lambert
MRPLPsolver = MultiRevolutionPerturbedLambertSolver() # MRPLP solver

# set the parameters for the MRPLP solver
order = 5 # order of the Taylor expansion
parameters = MRPLPsolver.mrplp_J2_analytic_parameters(rr1, rr2, tof, vv1g, mu, rE, J2,
                                            order, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200 )

```

The MRPLP is then solved and the outplut is printed:

```python
# extract the solution
vv1Sol = output.vv1Sol # initial velocity on Lambert arc - km/s
vv2Sol = output.vv2Sol # final velocity on Lambert arc - km/s

# compute the DV
dvv1 = vv1Sol - vv1          # initial DV on Lambert arc - km/s
dvv2 = vv2 - vv2Sol          # final DV on Lambert arc - km/s
dv1 = np.linalg.norm( dvv1 ) # initial DV magnitude on Lambert arc - km/s
dv2 = np.linalg.norm( dvv2 ) # final DV magnitude on Lambert arc - km/s
dvtot = dv1 + dv2            # total DV magnitude on Lambert arc - km/s

# print the output
print(f"-------------------------------------------------------")
print(f"                 OUTPUT SUMMARY")
print(f"-------------------------------------------------------")
print(f"Order of the expansion  : {order}")
print(f"Success                 : {output.success}")
print(f"Elapsed time            : {output.elapsed_time} seconds")
print(f"Final pos. error (norm) : {np.linalg.norm( output.rr2DA - rr2 )} km")
print(f"-------------------------------------------------------")
print(f"Delta_v1                : {dv1} km/s")
print(f"Delta_v2                : {dv2} km/s")
print(f"Delta_vtot              : {dvtot} km/s")
print(f"-------------------------------------------------------")
```

The printed summary will look like the following:

```bash
-------------------------------------------------------
                 OUTPUT SUMMARY
-------------------------------------------------------
Order of the expansion  : 5
Success                 : True
Elapsed time            : 0.1555500030517578 seconds
Final pos. error (norm) : 5.963966430719651e-11 km
-------------------------------------------------------
Delta_v1                : 2.1191021863094743 km/s
Delta_v2                : 1.4891976119021597 km/s
Delta_vtot              : 3.6082997982116343 km/s
-------------------------------------------------------
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
https://doi.org/10.2514/1.G003531.