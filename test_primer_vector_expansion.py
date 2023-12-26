from functions.MRPLP_J2_analytic import MultiRevolutionPerturbedLambertSolver
from functions.expansion_perturbed_lambert import ExpansionPerturbedLambert
from functions.primerVectorMinimization import minimizationPrimerVector
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
tof = 1.0*3600.0 # this works
# tof = 1.5 * 3600.0 # his works
tof = 3.0 *3600.0

# initialise the classes for MRPLP solver and expansion of perturbed Lambert
MRPLPsolver = MultiRevolutionPerturbedLambertSolver() # MRPLP solver
EXPpertLamb = ExpansionPerturbedLambert() # Expansion of perturbed Lambert
PVMinim = minimizationPrimerVector() # minimization

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
states = output_primer_vector_propagation.states  # trajectory of the spacecraft
vectof = output_primer_vector_propagation.vecttof # times
p = output_primer_vector_propagation.p            # primer vector magnitude history
pd = output_primer_vector_propagation.pd          # primer vector derivative history
pp = output_primer_vector_propagation.pp          # primer vector history

pmax = np.amax(p)    # maximum of the primer vector
index = np.argmax(p) # index of the maximum of the primer vector

# find the state and time at the maximum of the primer vector
tofM = vectof[index]
stateM = states[index,:]
ppm = pp[index,0:3]

# ------------ STEP 3: EXPAND AND APPLY A PERTURBATION ------------
parameters.order = 7 # increase the order of the expansion just in case (this will slightly increase computational time)

# print(f"-------------------------------------------------------")
# print(f"Performing the expansion...")
# x0DAepx1, xfDAexpM = EXPpertLamb.expansionOfPerturbedLambert(rr1, vv1Sol, tofM, parameters)
# x0DAepxM, xfDAexp2 = EXPpertLamb.expansionOfPerturbedLambert(stateM[0:3], stateM[3:6], tof-tofM, parameters)
# print(f"Done!")
# print(f"-------------------------------------------------------")

# ------------ STEP 4: FIND THE PERTURBATION THAT MINIMISES THE COST ------------

STM1M = MRPLPsolver.stateTransitionMatrix( rr1, vv1, tofM, parameters )
STMM2 = MRPLPsolver.stateTransitionMatrix( stateM[0:3], stateM[3:6], tof - tofM, parameters )
rrM = stateM[0:3]
vvM = stateM[3:6]

# generate initial guess
Mmf = STMM2[0:3,0:3]
Nmf = STMM2[0:3,3:6]
T0m = STM1M[3:6,3:6]
N0m = STM1M[0:3,3:6]
A = -( np.transpose(Mmf) @ np.transpose(np.linalg.inv(Nmf)) + T0m @ np.linalg.inv(N0m) )

beta = 5.0e-3
eps = beta*np.linalg.norm( rrM )/( abs( np.linalg.inv(A) @ ppm ) )
drrm = eps * np.linalg.inv(A) @ ppm

# solve the MRPLP --> from 1 to M
parameters = MRPLPsolver.mrplp_J2_analytic_parameters(rr1, rrM+drrm, tofM, vv1g, mu, rE, J2,
                                            order, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200 )
output = MRPLPsolver.mrplp_J2_analytic(parameters)
vv1_1M = output.vv1Sol
vv2_1M = output.vv2Sol

# solve the MRPLP --> from M to 2
parameters = MRPLPsolver.mrplp_J2_analytic_parameters(rrM+drrm, rr2, tof-tofM, vvM, mu,
                                                      rE, J2,
                                                    order, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200 )
output = MRPLPsolver.mrplp_J2_analytic(parameters)
vv1_M2 = output.vv1Sol
vv2_M2 = output.vv2Sol

dvv1 = vv1_1M - vv1
dvv2 = vv2 - vv2_M2
dvvM = vv1_M2 - vv2_1M
dv1n = np.linalg.norm( dvv1 )
dv2n = np.linalg.norm( dvv2 )
dvM = np.linalg.norm( dvvM )

dvtotNew = dv1n + dvM + dv2n

st = 1

# # evaluate the expanded states in the perturbation of rf
# drr0 = np.array( [0.0, 0.0, 0.0] ) # perturbation on rri
# drrM = drrm # perturbation on rri
# dtM = 0.0
# dX1 = np.concatenate( [drr0, drrM, np.array([dtM])] )

# # evaluate the expanded states in the perturbation of rf
# drrM2 = drrM                       # perturbation on rri --> this is equal to the one of the previous leg
# drr2 = np.array( [0.0, 0.0, 0.0] ) # perturbation on rri
# dt2 = 0.0
# dX2 = np.concatenate( [drrM2, drr2, np.array([dt2])] )

# x0DAepx1Eval = x0DAepx1.eval( dX1 )
# xfDAexpMEval = xfDAexpM.eval( dX1 )
# x0DAepxMEval = x0DAepxM.eval( dX2 )
# xfDAexp2Eval = xfDAexp2.eval( dX2 )

# dvv1 = x0DAepx1Eval[3:6] - vv1
# dvvM = x0DAepxMEval[3:6] - xfDAexpMEval[3:6]
# dvv2 = vv2 - xfDAexp2Eval[3:6]

# dv1 = np.linalg.norm(dvv1)
# dvM = np.linalg.norm(dvvM)
# dv2 = np.linalg.norm(dvv2)

# dvtotNew = dv1 + dvM + dv2

beta = 5.0e-3
result = PVMinim.minimizationWithPrimerVector( rr1, rr2, vv1, vv2, rrM, vvM, 
                                              beta, STM1M, STMM2, ppm,
                                              tof, tofM, parameters )

dXX = result.x
drrm = dXX[0:3]
dt = dXX[-1]

dx = dXX[0:3]
dt = dXX[-1]

# solve the MRPLP --> from 1 to M
parameters = MRPLPsolver.mrplp_J2_analytic_parameters(rr1, rrM+dx, tofM+dt, vv1, mu, rE, J2,
                                            order, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200 )
output = MRPLPsolver.mrplp_J2_analytic(parameters)
vv1_1M = output.vv1Sol
vv2_1M = output.vv2Sol

# solve the MRPLP --> from M to 2
parameters = MRPLPsolver.mrplp_J2_analytic_parameters(rrM+dx, rr2, tof-(tofM+dt), vvM, mu, rE, J2,
                                                    order, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200 )
output = MRPLPsolver.mrplp_J2_analytic(parameters)
vv1_M2 = output.vv1Sol
vv2_M2 = output.vv2Sol

dvv1 = vv1_1M - vv1
dvv2 = vv2 - vv2_M2
dvvM = vv1_M2 - vv2_1M
dv1n = np.linalg.norm( dvv1 )
dv2n = np.linalg.norm( dvv2 )
dvM = np.linalg.norm( dvvM )

dvtotNew = dv1n + dvM + dv2n

print(f"-------------------------------------------------------")
print(f"Propagating state and primer vector...")
output_primer_vector_propagation_1M = MRPLPsolver.propagatePrimerVector( rr1, vv1_1M, tofM+dt, dvv1, dvvM, Npoints, parameters )

output_primer_vector_propagation_M2 = MRPLPsolver.propagatePrimerVector( rrM+dx, vv1_M2, tof-(tofM+dt), dvvM, dvv2, Npoints, parameters )
print(f"Done!")
print(f"-------------------------------------------------------")

# extract the propagated primer vector
vectof1M = output_primer_vector_propagation_1M.vecttof
p1M = output_primer_vector_propagation_1M.p # primer vector magnitude history
pd1M = output_primer_vector_propagation_1M.pd # primer vector derivative history

# extract the propagated primer vector
vectofM2 = vectof1M[-1] + output_primer_vector_propagation_M2.vecttof
pM2 = output_primer_vector_propagation_M2.p # primer vector magnitude history
pdM2 = output_primer_vector_propagation_M2.pd # primer vector derivative history

# plot the primer vector magnitude
plt.subplot(2, 1, 1)
plt.plot(vectof1M/3600.0, p1M, linestyle='-', color = 'blue')
plt.plot(vectof1M[0]/3600.0, p1M[0], marker='*', color='red')
plt.plot(vectof1M[-1]/3600.0, p1M[-1], marker='*', color='red')

plt.plot(vectofM2/3600.0, pM2, linestyle='-', color = 'blue')
plt.plot(vectofM2[0]/3600.0, pM2[0], marker='*', color='red')
plt.plot(vectofM2[-1]/3600.0, pM2[-1], marker='*', color='red')

plt.xlabel('Time of flight (hours)')
plt.ylabel('Primer vector magnitude')

# plot the primer vector derivative
plt.subplot(2, 1, 2)
plt.plot(vectof1M/3600.0, pd1M, linestyle='-', color = 'blue')
plt.plot(vectof1M[0]/3600.0, pd1M[0], marker='*', color='red')
plt.plot(vectof1M[-1]/3600.0, pd1M[-1], marker='*', color='red')

plt.plot(vectofM2/3600.0, pdM2, linestyle='-', color = 'blue')
plt.plot(vectofM2[0]/3600.0, pdM2[0], marker='*', color='red')
plt.plot(vectofM2[-1]/3600.0, pdM2[-1], marker='*', color='red')

plt.xlabel('Time of flight (hours)')
plt.ylabel('Primer vector derivative')

plt.tight_layout()
plt.show()

st = 1
