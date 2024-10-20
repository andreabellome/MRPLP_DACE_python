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
# vv1g = np.array( [-5.33966929176339, -2.54974554812496, 4.35921075804661] ) --> TBP-LP
vv1g = vv1 # --> it should work also with a very brutal first guess
tof  = 20*86400.0
# tof = 1.5*3600.0
# t1 = 23783.1433383453
# t2 = 23786.1433383453

# initialise the classes for MRPLP solver and expansion of perturbed Lambert
MRPLPsolver = MultiRevolutionPerturbedLambertSolver() # MRPLP solver
EXPpertLamb = ExpansionPerturbedLambert() # Expansion of perturbed Lambert

# set the parameters for the MRPLP solver
order=5
parameters = MRPLPsolver.mrplp_J2_analytic_parameters(rr1, rr2, tof, vv1g, mu, rE, J2,
                                            order, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200 )

# solve the MRPLP
output = MRPLPsolver.mrplp_J2_analytic(parameters)

# extract the solution
vv1Sol = output.vv1Sol
vv2Sol = output.vv2Sol
rr2DA = output.rr2DA

# compute the DV
dvv1 = vv1Sol - vv1
dvv2 = vv2 - vv2Sol
dv1 = np.linalg.norm( dvv1 )
dv2 = np.linalg.norm( dvv2 )
dvtot = dv1 + dv2

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

# expansion w.r.t. (ri, rf, tof)
parameters.order = 7 # increase the order of the expansion just in case (this will increase computational time)
x0DAepx, xfDAexp = EXPpertLamb.expansionOfPerturbedLambert(rr1, vv1Sol, tof, parameters)

# evaluate the expanded states in the perturbation of rf
drr0 = np.array( [0.0, 0.0, 0.0] ) # perturbation on rri
drrf = np.array( [0.0, 0.0, 0.0] ) # perturbation on rri
dt = 10.0
dX = np.concatenate( [drr0, drrf, np.array([dt])] )

x0Reference = np.concatenate( [rr1, vv1Sol] )
xfReference = np.concatenate( [rr2, vv2Sol] )

x0DAepxEval = x0DAepx.eval( dX )
xfDAexpEval = xfDAexp.eval( dX )

# # propagate the entire trajectory and get the STM
# Npoints = 1000.0
# output_primer_vector_propagation = MRPLPsolver.propagatePrimerVector( rr1, vv1Sol, tof, dvv1, dvv2, Npoints, parameters )

# # extract the propagated primer vector
# vectof = output_primer_vector_propagation.vecttof
# p = output_primer_vector_propagation.p # primer vector magnitude history
# pd = output_primer_vector_propagation.pd # primer vector derivative history

# pmax = np.amax(p) # maximum of the primer vector
# index = np.argmax(p) # index of the maximum of the primer vector

# # plot the primer vector magnitude
# plt.subplot(2, 1, 1)
# plt.plot(vectof/3600.0, p, linestyle='-', color = 'blue')
# plt.plot(vectof[0]/3600.0, p[0], marker='*', color='red')
# plt.plot(vectof[-1]/3600.0, p[-1], marker='*', color='red')
# plt.plot(vectof[index]/3600.0, p[index], marker='*', color='red')
# plt.xlabel('Time of flight (hours)')
# plt.ylabel('Primer vector magnitude')

# # plot the primer vector derivative
# plt.subplot(2, 1, 2)
# plt.plot(vectof/3600.0, pd, linestyle='-', color = 'blue')
# plt.plot(vectof[0]/3600.0, pd[0], marker='*', color='red')
# plt.plot(vectof[-1]/3600.0, pd[-1], marker='*', color='red')
# plt.plot(vectof[index]/3600.0, pd[index], marker='*', color='red')
# plt.xlabel('Time of flight (hours)')
# plt.ylabel('Primer vector derivative')

# # adjust layout for better spacing
# plt.tight_layout()

# # show the plots
# plt.show()

# STM12 = output_primer_vector_propagation.STM[-1,:,:] # STM between t1 and t2
STM12check = MRPLPsolver.stateTransitionMatrix( rr1, vv1Sol, tof, parameters )
primerVectorInitialFinalCond = MRPLPsolver.primerVectorInitialAndFinalConditions(dvv1, dvv2, STM12check)

st = 1
