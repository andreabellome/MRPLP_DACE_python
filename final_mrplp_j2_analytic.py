from functions.MRPLP_J2_analytic import *
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
vv1g = vv1 # --> it should work also with a very brutal first guess --> usually better than TBP-LP
tof  = 3*86400.0
# t1 = 23783.1433383453
# t2 = 23786.1433383453

# set the parameters for the MRPLP solver
parameters = mrplp_J2_analytic_parameters( rr1, rr2, tof, vv1g, mu, rE, J2,
                                            order=5, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200)

# solve the MRPLP
output = mrplp_J2_analytic(parameters)

# double check the propagation
x0 = array( np.append(rr1, output.vv1Sol) )
xxf = analyticJ2propHill( x0, tof, mu, rE, J2, output.paramsSol.cont )

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

# propagate the entire trajectory and get the STM
Npoints = 1000.0
output_primer_vector_propagation = propagatePrimerVector( rr1, vv1Sol, tof,
                                                         dvv1, dvv2,
                                                           Npoints, parameters )

vectof = output_primer_vector_propagation.vecttof
p = output_primer_vector_propagation.p
# plt.plot(vectof, p, linestyle='-')
# plt.show()

STM12 = output_primer_vector_propagation.STM[-1,:,:] # STM between t1 and t2
primerVectorInitialFinalCond = primerVectorInitialAndFinalConditions(dvv1, dvv2, STM12)

# computational time and error on the final position
elapsed_time = output.elapsed_time
errorvec = rr2DA - rr2
error = np.linalg.norm( errorvec )

# print the output
print(f"OUTPUT SUMMARY          :")
print(f"Success                 : {output.success}")
print(f"Elapsed time            : {elapsed_time} seconds")
print(f"Final pos. error (norm) : {error} km")
print(f"Delta_v1                : {dv1} km/s")
print(f"Delta_v2                : {dv2} km/s")
print(f"Delta_vtot              : {dvtot} km/s")

st = 1
