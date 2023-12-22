from functions.MRPLP_J2_analytic import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# TBP J2 dynamics using numpy arrays
def TBP_J2_numpy(t: float, x: np.array, mu: float, R: float, J2: float) -> array:

    pos = x[0:3]
    vel = x[3:6]
    r = np.linalg.norm(pos)
    dxdt = vel
    dvdt = -mu * pos / (r ** 3)
    dvdt[0] = dvdt[0] + 1.5*mu*J2*R**2/(r**5)*(5.*(pos[2]**2)/(r**2) - 1.)*pos[0]
    dvdt[1] = dvdt[1] + 1.5*mu*J2*R**2/(r**5)*(5.*(pos[2]**2)/(r**2) - 1.)*pos[1]
    dvdt[2] = dvdt[2] + 1.5*mu*J2*R**2/(r**5)*(5.*(pos[2]**2)/(r**2) - 3.)*pos[2]
    return np.concatenate([dxdt, dvdt])

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
order=5
parameters = mrplp_J2_analytic_parameters( rr1, rr2, tof, vv1g, mu, rE, J2,
                                            order, tol=1.0e-6, cont=0.0,
                                              dcontMin=0.1, scl=1.0e-3, itermax=200)

# solve the MRPLP
output = mrplp_J2_analytic(parameters)

# double check the propagation - analytical J2 prop
xx0 = array( np.append(rr1, output.vv1Sol) )
xxf = analyticJ2propHill( xx0, tof, mu, rE, J2, output.paramsSol.cont )
xx0 = xx0.cons() # initial state - constant part of the Taylor expansion
xxf = xxf.cons() # final state - constant part of the Taylor expansion

# double check the propagation - numerical J2 prop
solver = integrate.RK45(fun=lambda t, x: TBP_J2_numpy(t, x, mu, rE, J2), 
                        t0 = 0.0, y0=xx0, t_bound=tof, max_step=np.inf, 
                        rtol=1.0e-12, atol=1.0e-12)

# Lists to store results
t_values = [solver.t]
y_values = [solver.y]

# Perform integration
while solver.status == 'running':
    solver.step()
    t_values.append(solver.t)
    y_values.append(solver.y.copy())

# Convert results to NumPy arrays
t_values = np.array(t_values)
y_values = np.array(y_values)

# print the output
print(f"-------------------------------------------------------")
print(f"                 OUTPUT SUMMARY")
print(f"-------------------------------------------------------")
print(f"Order of the expansion                 : {order}")
print(f"Success                                : {output.success}")
print(f"Elapsed time                           : {output.elapsed_time} seconds")
print(f"Final pos. error (norm)                : {np.linalg.norm( output.rr2DA - rr2 )} km")
print(f"Final pos. error (norm) with numerical : {np.linalg.norm( y_values[-1] - xxf )} km")
print(f"-------------------------------------------------------")

st = 1

