from functions.analytic_approx_dv_j2 import *
from functions.MRPLP_J2_analytic import *
import numpy as np

# define the constant of motion for the central body (Earth in this case)
mu = 398600.4418e9/1000.0**3  # km^3/s^2
J2 = 1.08262668e-3
rE = 6378137/1000.0 # km

# define initial and final point and epochs
rr1 = np.array([ -2144461.88570992, 4385305.93248107, 5327996.61073502])/1000.0
vv1 = np.array([-636.454980956152, 5578.34465234474, -4863.74571083629])/1000.0
rr2 = np.array([1622482.2343302  ,       -6749986.50462197   ,       1666520.49053527])/1000.0
vv2 = np.array([-1509.16186692981     ,     1412.72768065514 ,         7165.11607531926])/1000.0
t1 = 26272.8
t2 = 26273.61
tof = (t2 - t1)*86400.0

# approx. DV under J2 dynamics
paramsApproxDV = approx_DV_parameters( rr1, vv1, rr2, vv2, t1, t2, mu, rE, J2 )
dvApprox = approx_DV(paramsApproxDV)

# set the parameters for the MRPLP solver and solve the MRPLP
order = 5 # order of the Taylor polynomial expansion
parameters = mrplp_J2_analytic_parameters( rr1, rr2, tof, vv1, mu, rE, J2,
                                            order, tol=1.0e-6, cont=0.0, dcontMin=0.1, scl=1.0e-3, itermax=200)
output = mrplp_J2_analytic(parameters) # solve the MRPLP

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

# computational time and error on the final position
elapsed_time = output.elapsed_time
errorvec = rr2DA - rr2
error = np.linalg.norm( errorvec )

# print the output
print(f"-------------------------------------------------------")
print(f"                 OUTPUT SUMMARY")
print(f"-------------------------------------------------------")
print(f"Order of the expansion  : {order}")
print(f"Success                 : {output.success}")
print(f"Elapsed time            : {elapsed_time} seconds")
print(f"Final pos. error (norm) : {error} km")
print(f"-------------------------------------------------------")
print(f"Delta_v1                : {dv1} km/s")
print(f"Delta_v2                : {dv2} km/s")
print(f"Delta_vtot              : {dvtot} km/s")
print(f"-------------------------------------------------------")
print(f"Approx. Delta_vtot      : {dvApprox} km/s")
print(f"Perc. difference        : {(dvApprox - dvtot)/dvtot*100} %")
print(f"-------------------------------------------------------")

st = 1
