from functions.MRPLP_J2_analytic import MultiRevolutionPerturbedLambertSolver
from functions.expansion_perturbed_lambert import ExpansionPerturbedLambert
import numpy as np
import matplotlib.pyplot as plt

# define the constant of motion for the central body (Earth in this case)
mu = 398600.4418  # km^3/s^2
J2 = 1.08262668e-3
rE = 6378.137 # km

# initial guess, target position and time of flight
rr1  = np.array( [-3173.91245750977, -1863.35865746, -6099.31199561] )
vv1 = np.array( [-6.37541145277431, -1.26857476842513, 3.70632783068748] )
rr2  = np.array( [6306.80758519, 3249.39062728,  794.06530085] )
vv2 = np.array( [1.1771075218004, -0.585047636781159, -7.370399738227] )
vv1g = vv1 # --> it should work also with a very brutal first guess
tof = 1.5*3600.0

# initialise the classes for MRPLP solver and expansion of perturbed Lambert
MRPLPsolver = MultiRevolutionPerturbedLambertSolver() # MRPLP solver
EXPpertLamb = ExpansionPerturbedLambert() # Expansion of perturbed Lambert

# ------------ STEP 1: SOLVE THE MRPLP ------------

# set the parameters for the MRPLP solver
order=5
parameters = MRPLPsolver.mrplp_J2_analytic_parameters(rr1, rr2, tof, vv1g, mu, rE, J2,
                                            order, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200 )

# solve the MRPLP
output = MRPLPsolver.mrplp_J2_analytic(parameters)

# extract the solution
vv1Sol = output.vv1Sol
vv2Sol = output.vv2Sol

# compute the DV
dvv1 = vv1Sol - vv1
dvv2 = vv2 - vv2Sol
dv1 = np.linalg.norm( dvv1 )
dv2 = np.linalg.norm( dvv2 )
dvtot = dv1 + dv2

# print the output
print(f"-------------------------------------------------------")
print(f"                MRPLP OUTPUT SUMMARY")
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

# ------------ STEP 2: FIND THE PRIMER VECTOR EVOLUTION ------------

# propagate the entire trajectory and get the STM
print(f"-------------------------------------------------------")
print(f"Propagating state and primer vector...")
Npoints = 1000.0
output_primer_vector_propagation = MRPLPsolver.propagatePrimerVector( rr1, vv1Sol, tof, dvv1, dvv2, Npoints, parameters )
print(f"Done!")
print(f"-------------------------------------------------------")

# extract the propagated trajectory and the primer vector
states = output_primer_vector_propagation.states
vectof = output_primer_vector_propagation.vecttof
p = output_primer_vector_propagation.p # primer vector magnitude history
pd = output_primer_vector_propagation.pd # primer vector derivative history

pmax = np.amax(p) # maximum of the primer vector
index = np.argmax(p) # index of the maximum of the primer vector

# find the state and time at the maximum of the primer vector
tofM = vectof[index]
stateM = states[index,:]

# ------------ STEP 3: EXPAND AND APPLY A PERTURBATION ------------
parameters.order = 7 # increase the order of the expansion just in case (this will increase computational time)

x0DAepx1, xfDAexpM = EXPpertLamb.expansionOfPerturbedLambert(rr1, vv1Sol, tofM, parameters)
x0DAepxM, xfDAexp2 = EXPpertLamb.expansionOfPerturbedLambert(stateM[0:3], stateM[3:6], tof-tofM, parameters)

# evaluate the expanded states in the perturbation of rf
drr0 = np.array( [0.0, 0.0, 0.0] ) # perturbation on rri
drrM = np.array( [1.0, 1.0, 0.0] ) # perturbation on rri
dtM = 0.0
dX1 = np.concatenate( [drr0, drrM, np.array([dtM])] )

# evaluate the expanded states in the perturbation of rf
drrM2 = drrM # perturbation on rri
drr2 = np.array( [0.0, 0.0, 0.0] ) # perturbation on rri
dt2 = 0.0
dX2 = np.concatenate( [drrM2, drr2, np.array([dt2])] )

x0DAepx1Eval = x0DAepx1.eval( dX1 )
xfDAexpMEval = xfDAexpM.eval( dX1 )
x0DAepxMEval = x0DAepxM.eval( dX2 )
xfDAexp2Eval = xfDAexp2.eval( dX2 )

dvv1 = x0DAepx1Eval[3:6] - vv1
dvvM = x0DAepxMEval[3:6] - xfDAexpMEval[3:6]
dvv2 = vv2 - xfDAexp2Eval[3:6]

dv1 = np.linalg.norm(dvv1)
dvM = np.linalg.norm(dvvM)
dv2 = np.linalg.norm(dvv2)

st = 1
