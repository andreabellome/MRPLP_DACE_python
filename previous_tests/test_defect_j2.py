
from typing import Callable, Type

import numpy as np
from numpy.typing import NDArray
from daceypy import DA, array, RK, integrator
from scipy.io import loadmat
from scipy.optimize import fsolve

# define the constant of motion
mu = 398600.4418  # km^3/s^2
J2 = 1.08262668e-3
Req = 6378.137 # km

# true to eccentric anomaly
def true2eccAnomaly( theta: DA, e: DA ):
    return 2.0 * np.arctan2(np.sqrt(1. - e)*np.sin(theta / 2.), np.sqrt(1. + e) * np.cos(theta / 2.))

# true to mean anomaly
def true2meanAnomaly(theta: DA, e: DA):
    E = true2eccAnomaly(theta, e)
    return E - e*np.sin(E)

# mean to eccentric anomaly
def mean2eccAnomaly( M: DA, e: DA ):
    E = M
    for i in range(20):
        E = M + e*np.sin(E)
    
    return E

# eccentric to true anomaly
def ecc2trueAnomaly( E: DA, e: DA ):
    return 2.0 * np.arctan2(np.sqrt(1. + e)*np.sin(E / 2.), np.sqrt(1. - e) * np.cos(E / 2.))

# mean to true anomaly
def mean2trueAnomaly( M: DA, e:DA ):
    E = mean2eccAnomaly(M, e)
    return ecc2trueAnomaly(E, e)

# cartesian elements to keplerian
def cart2kep(x0: array, mu: float):
    x0 = x0.copy()

    rr = x0[0:3]
    vv = x0[3:6]
    r = np.linalg.norm(rr)
    v = np.linalg.norm(vv)
    hh = np.cross(rr, vv)

    sma = mu / ( 2.0 * (mu/r - v**2/2.0) )
    h1sqr = hh[0]**2
    h2sqr = hh[1]**2

    if (h1sqr + h2sqr).cons() == 0.0:
        RAAN = 0.0
    else:
        sinOMEGA = hh[0] / np.sqrt( h1sqr + h2sqr )
        cosOMEGA = -1.0*hh[1] / np.sqrt( h1sqr + h2sqr )
        if cosOMEGA.cons() >= 0.0:
            if sinOMEGA.cons() >= 0.0:
                RAAN = np.arcsin( hh[0] / np.sqrt( h1sqr + h2sqr ) )
            else:
                RAAN = 2.0 * np.pi + np.arcsin( hh[0] / np.sqrt( h1sqr + h2sqr ) )
        else:
            if sinOMEGA.cons() >= 0.0:
                RAAN = np.arccos( -1.0*hh[1] / np.sqrt( h1sqr + h2sqr ) )
            else:
                RAAN = 2.0 * np.pi - np.arccos( -1.0*hh[1] / np.sqrt( h1sqr + h2sqr ) )

    ee = 1.0/mu * np.cross(vv, hh) - rr/r
    e = np.linalg.norm(ee)
    i = np.arccos( hh[2]/np.linalg.norm(hh) )
    
    if e.cons() <= 1.0e-8 and i.cons() <= 1e-8:
        e = 0.0
        omega = np.arctan2(rr[1], rr[0])
        theta = 0.0
        kep = array( [sma, e, i, RAAN, omega, theta] )
        return kep
    
    if e.cons() <= 1.0e-8 and i.cons() > 1e-8:
        omega = 0.0
        P = array.zeros(3)
        Q = array.zeros(3)
        W = array.zeros(3)
        P[0] = np.cos(omega)*np.cos(RAAN) - np.sin(omega)*np.sin(i)*np.sin(RAAN)
        P[1] = -1.0*np.sin(omega)*np.cos(RAAN) - np.cos(omega)*np.cos(i)*np.sin(RAAN)
        P[2] = np.sin(RAAN)*np.sin(i)
        Q[0] = np.cos(omega)*np.sin(RAAN) + np.sin(omega)*np.cos(i)*np.cos(RAAN)
        Q[1] = -1.0*np.sin(omega)*np.sin(RAAN) + np.cos(omega)*np.cos(i)*np.cos(RAAN)
        Q[2] = -1.0*np.cos(RAAN)*np.sin(i)
        W[0] = np.sin(omega)*np.sin(i)
        W[1] = np.cos(omega)*np.sin(i)
        W[2] = np.cos(i)
        rrt = P*rr[0] + Q*rr[1] + W*rr[2]
        theta = np.arctan2(rrt[1], rrt[0])
        kep = array( [sma, e, i, RAAN, omega, theta] )
        return kep
    
    dotRxE = np.dot(rr, ee)
    RxE = np.linalg.norm(rr)*np.linalg.norm(ee)
    if abs((dotRxE).cons()) > abs((RxE).cons()) and abs((dotRxE).cons()) - abs((RxE).cons()) < abs(1.0e-6*(dotRxE).cons()):
        dotRxE = 1.0e-6*dotRxE
    
    theta = np.arccos(dotRxE / RxE)

    if np.dot(rr, vv).cons() < 0.0:
        theta = 2.0 * np.pi - theta
    
    if (i.cons() <= 1.0e-8 and e.cons() >= 1.0e-8):
        i = 0.0
        omega = np.arctan2(ee[1], ee[0])
        kep = array( [sma, e, i, RAAN, omega, theta] )
        return kep
    
    sino = rr[2] / r / np.sin(i)
    coso = (rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r

    if (coso.cons() >= 0.0):
        if sino.cons() >= 0.0:
            argLat = np.arcsin(rr[2] / r / np.sin(i))
        else:
            argLat = 2.0 * np.pi + np.arcsin(rr[2] / r / np.sin(i))
    else:
        if coso.cons() >= 0.0:
            argLat = np.arccos((rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r)
        else:
            argLat = 2.0 * np.pi - np.arccos((rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r)

    omega = argLat - theta

    if omega.cons() < 0.0:
        omega = omega + 2.0 * np.pi
    
    kep = array( [sma, e, i, RAAN, omega, theta] )
    return kep

# keplerian elements to Hill
def kep2Hill( kep: array, mu: float ):
    
    hill = array.zeros(6)
    p = kep[0]*(1.0 - kep[1]*kep[1])
    f = kep[5]

    hill[4] = np.sqrt(mu*p)
    hill[0] = p/(1.0 + kep[1]*np.cos(f))
    hill[1] = f + kep[4]
    hill[2] = kep[3]
    hill[3] = (hill[4]/p)*kep[1]*np.sin(f)
    hill[5] = hill[4]*np.cos(kep[2])

    return hill

# Hill to cartesian elements
def hill2cart( hill: array, mu: float ):
    r = hill[0]
    th = hill[1]
    nu = hill[2]
    R = hill[3]
    Th = hill[4]
    ci = hill[5]/hill[4]
    si = np.sqrt(1.0 - ci*ci)

    u = array.zeros(3)
    u[0] = np.cos(th)*np.cos(nu) - ci*np.sin(th)*np.sin(nu)
    u[1] = np.cos(th)*np.sin(nu) + ci*np.sin(th)*np.cos(nu)
    u[2] = si*np.sin(th)

    cart = array.zeros(6)
    cart[0] = r*u[0]
    cart[1] = r*u[1]
    cart[2] = r*u[2]
    cart[3] = (R*np.cos(th) - Th*np.sin(th)/r)*np.cos(nu) - (R*np.sin(th) + Th*np.cos(th)/r)*np.sin(nu)*ci
    cart[4] = (R*np.cos(th) - Th*np.sin(th)/r)*np.sin(nu) + (R*np.sin(th) + Th*np.cos(th)/r)*np.cos(nu)*ci
    cart[5] = (R*np.sin(th) + Th*np.cos(th)/r)*si

    return cart

# from osculating to mean Hill
def osculating2meanHill( hillOsc: array, mu: float, J2: float, rE: float, cont: float ):
    r = hillOsc[0]
    th = hillOsc[1]
    nu = hillOsc[2]
    R = hillOsc[3]
    Th = hillOsc[4]
    Nu = hillOsc[5]

    ci = Nu/Th
    si = np.sqrt(1.0-ci*ci)
    cs = (-1.0 + pow(Th,2)/(mu*r))*np.cos(th) + (R*Th*np.sin(th))/mu
    ss = -((R*Th*np.cos(th))/mu) + (-1.0 + pow(Th,2)/(mu*r))*np.sin(th)
    e = np.sqrt(cs*cs+ss*ss)
    eta = np.sqrt(1.0-e*e)

    beta = 1.0/(1.0+eta)
 
    p = Th*Th/mu
    costrue = 1/e*(p/r-1)
    
    f = np.arccos(costrue)

    if R.cons() < 0.0:
        f = 2.0*np.pi-f
    
    M = true2meanAnomaly(f,e)
    phi = f - M

    rMean = r + (cont)*((pow(rE,2)*beta*J2)/(2.*r) - (3*pow(rE,2)*beta*J2*pow(si,2))/(4.*r) +
    (pow(rE,2)*eta*J2*pow(mu,2)*r)/pow(Th,4) - (3*pow(rE,2)*eta*J2*pow(mu,2)*r*pow(si,2))/(2.*pow(Th,4)) + (pow(rE,2)*J2*mu)/(2.*pow(Th,2)) - (pow(rE,2)*beta*J2*mu)/(2.*pow(Th,2)) -
    (3.*pow(rE,2)*J2*mu*pow(si,2))/(4.*pow(Th,2)) + (3*pow(rE,2)*beta*J2*mu*pow(si,2))/(4.*pow(Th,2)) -
    (pow(rE,2)*J2*mu*pow(si,2)*np.cos(2*th))/(4.*pow(Th,2)))

    thMean = th + (cont)*((-3.*pow(rE,2)*J2*pow(mu,2)*phi)/pow(Th,4) + (15.*pow(rE,2)*J2*pow(mu,2)*phi*pow(si,2))/(4.*pow(Th,4)) -
    (5.*pow(rE,2)*J2*mu*R)/(2.*pow(Th,3)) - (pow(rE,2)*beta*J2*mu*R)/(2.*pow(Th,3)) +
    (3.*pow(rE,2)*J2*mu*R*pow(si,2))/pow(Th,3) + (3.*pow(rE,2)*beta*J2*mu*R*pow(si,2))/(4.*pow(Th,3)) -
    (pow(rE,2)*beta*J2*R)/(2.*r*Th) + (3.*pow(rE,2)*beta*J2*R*pow(si,2))/(4.*r*Th) +
    (-(pow(rE,2)*J2*mu*R)/(2.*pow(Th,3)) + (pow(rE,2)*J2*mu*R*pow(si,2))/pow(Th,3))*np.cos(2.*th) +
    (-(pow(rE,2)*J2*pow(mu,2))/(4.*pow(Th,4)) + (5.*pow(rE,2)*J2*pow(mu,2)*pow(si,2))/(8.*pow(Th,4)) +
    (pow(rE,2)*J2*mu)/(r*pow(Th,2)) - (3.*pow(rE,2)*J2*mu*pow(si,2))/(2.*r*pow(Th,2)))*np.sin(2.*th))

    nuMean = nu + (cont)*((3.*pow(rE,2)*ci*J2*pow(mu,2)*phi)/(2.*pow(Th,4)) + (3.*pow(rE,2)*ci*J2*mu*R)/(2.*pow(Th,3)) +
    (pow(rE,2)*ci*J2*mu*R*np.cos(2.*th))/(2.*pow(Th,3)) +
    ((pow(rE,2)*ci*J2*pow(mu,2))/(4.*pow(Th,4)) - (pow(rE,2)*ci*J2*mu)/(r*pow(Th,2)))*np.sin(2.*th))

    RMean = R + (cont)*(-(pow(rE,2)*beta*J2*R)/(2.*pow(r,2)) + (3.*pow(rE,2)*beta*J2*R*pow(si,2))/(4.*pow(r,2)) -
    (pow(rE,2)*eta*J2*pow(mu,2)*R)/(2.*pow(Th,4)) + (3.*pow(rE,2)*eta*J2*pow(mu,2)*R*pow(si,2))/(4.*pow(Th,4)) +
    (pow(rE,2)*J2*mu*pow(si,2)*np.sin(2.*th))/(2.*pow(r,2)*Th))

    ThMean = Th + (cont)*(((pow(rE,2)*J2*pow(mu,2)*pow(si,2))/(4.*pow(Th,3)) - (pow(rE,2)*J2*mu*pow(si,2))/(r*Th))*np.cos(2.*th) -
    (pow(rE,2)*J2*mu*R*pow(si,2)*np.sin(2.*th))/(2.*pow(Th,2)))

    NuMean = Nu + 0.

    hillMean = array( [ rMean, thMean, nuMean, RMean, ThMean, NuMean ] )

    return hillMean

# from mean to osculating Hill
def mean2osculatingHill( hillMean: array, mu: float, J2: float, rE: float, cont: float ):
    r = hillMean[0] # l
    th = hillMean[1] # g
    nu = hillMean[2] # h
    R = hillMean[3] # L
    Th = hillMean[4] # G
    Nu = hillMean[5] # H
    ci = Nu/Th
    si = np.sqrt(1.0-ci*ci)
    cs = (-1.0 + pow(Th,2)/(mu*r))*np.cos(th) + (R*Th*np.sin(th))/mu
    ss = -((R*Th*np.cos(th))/mu) + (-1.0 + pow(Th,2)/(mu*r))*np.sin(th)
    e = np.sqrt(cs*cs+ss*ss)
    eta = np.sqrt(1.0-e*e)
    beta = 1.0/(1.0+eta)
    p = Th*Th/mu
    costrue = 1/e*(p/r-1)
    f = np.arccos(costrue)

    if R.cons() < 0.0:
        f = 2.0 * np.pi - f

    M = true2meanAnomaly(f,e)
    phi = f - M

    rOsc = r - (cont)*((pow(rE,2)*beta*J2)/(2.*r) - (3*pow(rE,2)*beta*J2*pow(si,2))/(4.*r) +
    (pow(rE,2)*eta*J2*pow(mu,2)*r)/pow(Th,4) - (3*pow(rE,2)*eta*J2*pow(mu,2)*r*pow(si,2))/(2.*pow(Th,4)) +
    (pow(rE,2)*J2*mu)/(2.*pow(Th,2)) - (pow(rE,2)*beta*J2*mu)/(2.*pow(Th,2)) -
    (3.*pow(rE,2)*J2*mu*pow(si,2))/(4.*pow(Th,2)) + (3*pow(rE,2)*beta*J2*mu*pow(si,2))/(4.*pow(Th,2)) -
    (pow(rE,2)*J2*mu*pow(si,2)*np.cos(2*th))/(4.*pow(Th,2)))

    thOsc = th - (cont)*((-3.*pow(rE,2)*J2*pow(mu,2)*phi)/pow(Th,4) + (15.*pow(rE,2)*J2*pow(mu,2)*phi*pow(si,2))/(4.*pow(Th,4)) - (5.*pow(rE,2)*J2*mu*R)/(2.*pow(Th,3)) - (pow(rE,2)*beta*J2*mu*R)/(2.*pow(Th,3)) +
    (3.*pow(rE,2)*J2*mu*R*pow(si,2))/pow(Th,3) + (3.*pow(rE,2)*beta*J2*mu*R*pow(si,2))/(4.*pow(Th,3)) -
    (pow(rE,2)*beta*J2*R)/(2.*r*Th) + (3.*pow(rE,2)*beta*J2*R*pow(si,2))/(4.*r*Th) +
    (-(pow(rE,2)*J2*mu*R)/(2.*pow(Th,3)) + (pow(rE,2)*J2*mu*R*pow(si,2))/pow(Th,3))*np.cos(2.*th) +
    (-(pow(rE,2)*J2*pow(mu,2))/(4.*pow(Th,4)) + (5.*pow(rE,2)*J2*pow(mu,2)*pow(si,2))/(8.*pow(Th,4)) +
    (pow(rE,2)*J2*mu)/(r*pow(Th,2)) - (3.*pow(rE,2)*J2*mu*pow(si,2))/(2.*r*pow(Th,2)))*np.sin(2.*th))

    nuOsc = nu - (cont)*((3.*pow(rE,2)*ci*J2*pow(mu,2)*phi)/(2.*pow(Th,4)) + (3.*pow(rE,2)*ci*J2*mu*R)/(2.*pow(Th,3)) +
    (pow(rE,2)*ci*J2*mu*R*np.cos(2.*th))/(2.*pow(Th,3)) +
    ((pow(rE,2)*ci*J2*pow(mu,2))/(4.*pow(Th,4)) - (pow(rE,2)*ci*J2*mu)/(r*pow(Th,2)))*np.sin(2.*th))

    ROsc = R - (cont)*(-(pow(rE,2)*beta*J2*R)/(2.*pow(r,2)) + (3.*pow(rE,2)*beta*J2*R*pow(si,2))/(4.*pow(r,2)) -
    (pow(rE,2)*eta*J2*pow(mu,2)*R)/(2.*pow(Th,4)) + (3.*pow(rE,2)*eta*J2*pow(mu,2)*R*pow(si,2))/(4.*pow(Th,4)) +
    (pow(rE,2)*J2*mu*pow(si,2)*np.sin(2.*th))/(2.*pow(r,2)*Th))

    ThOsc = Th - (cont)*(((pow(rE,2)*J2*pow(mu,2)*pow(si,2))/(4.*pow(Th,3)) - (pow(rE,2)*J2*mu*pow(si,2))/(r*Th))*np.cos(2.*th) - (pow(rE,2)*J2*mu*R*pow(si,2)*np.sin(2.*th))/(2.*pow(Th,2)))

    NuOsc = Nu + 0.

    hillOsc = array.zeros(6)

    hillOsc[0] = rOsc
    hillOsc[1] = thOsc
    hillOsc[2] = nuOsc
    hillOsc[3] = ROsc
    hillOsc[4] = ThOsc
    hillOsc[5] = NuOsc

    return hillOsc

# Hill to keplerian
def hill2kep( hill: array, mu: float ):
    r = hill[0]
    th = hill[1]
    nu = hill[2]
    R = hill[3]
    Th = hill[4]
    Nu = hill[5]

    i = np.arccos(Nu/Th)
    cs = (-1.0 + pow(Th,2)/(mu*r))*np.cos(th) + (R*Th*np.sin(th))/mu
    ss = -((R*Th*np.cos(th))/mu) + (-1.0 + pow(Th,2)/(mu*r))*np.sin(th)
    e = np.sqrt(cs*cs+ss*ss)
    p = Th*Th/mu
    costrue = 1.0/e*(p/r-1.0)
    f = np.arccos(costrue)

    if R.cons() < 0.0:
        f = 2.0*np.pi-f
    
    a = p/(1-e*e)

    kep = array( [a, e, i, nu, th-f, f] )
    return kep

# keplerian to Hill
def kep2hill( kep: array, mu: float ):
    p = kep[0]*(1.0 - kep[1]*kep[1])
    f = kep[5]
    
    hill = array.zeros(6)
    hill[4] = np.sqrt(mu*p)
    hill[0] = p/(1.0 + kep[1]*np.cos(f))
    hill[1] = f + kep[4]
    hill[2] = kep[3]
    hill[3] = (hill[4]/p)*kep[1]*np.sin(f)
    hill[5] = hill[4]*np.cos(kep[2])

    return hill

# kep to delaunay
def kep2delaunay(kep: array, mu: float):
    a = kep[0]
    e = kep[1]
    i = kep[2]
    RAAN = kep[3]
    omega = kep[4]
    M = kep[5]

    delaunay = array.zeros(6)
    delaunay[0] = M
    delaunay[1] = omega
    delaunay[2] = RAAN
    delaunay[3] = np.sqrt(mu*a)
    delaunay[4] = np.sqrt(1 - pow(e, 2)) * delaunay[3]
    delaunay[5] = np.cos(i)*delaunay[4]

    return delaunay

# averaged Delaunay
def averagedJ2rhs(xxx: array, mu: float, J2: float, rE: float, cont: float ):

    l = xxx[0]
    g = xxx[1]
    h = xxx[2]
    L = xxx[3]
    G = xxx[4]
    H = xxx[5]
    eta = G/L
    ci = H/G
    si = np.sin(np.arccos(ci))

    dldt = pow(mu, 2) / pow(L, 3) + (cont)*((3.0 * J2*pow(rE, 2)*pow(mu, 4)) / (2.0*pow(L, 7)*pow(eta, 3)) -
	(9.0 * J2*pow(si, 2)*pow(rE, 2)*pow(mu, 4)) / (4.0*pow(L, 7)*pow(eta, 3)))

    dgdt = (cont)*((3. * J2*pow(rE, 2)*pow(mu, 4)) / (2.*pow(L, 7)*pow(eta, 4)) -
	(9. * J2*pow(si, 2)*pow(rE, 2)*pow(mu, 4)) / (4.*pow(L, 7)*pow(eta, 4)) +
	(3. * pow(ci, 2)*J2*pow(rE, 2)*pow(mu, 4)) / (2.*G*pow(L, 6)*pow(eta, 3)))

    dhdt = (cont)*(-(3. * pow(ci, 2)*J2*pow(rE, 2)*pow(mu, 4)) / (2.*H*pow(L, 6)*pow(eta, 3)))

    ff = array.zeros(6)
    ff[0] = dldt
    ff[1] = dgdt
    ff[2] = dhdt
    ff[3] = 0
    ff[4] = 0
    ff[5] = 0

    return ff

# Delaunay to kep
def delaunay2kep( delaunay: array, mu: float ):
    l = delaunay[0]
    g = delaunay[1]
    h = delaunay[2]
    L = delaunay[3]
    G = delaunay[4]
    H = delaunay[5]

    kep = array.zeros(6)
    kep[0] = pow(L, 2) / mu
    kep[1] = np.sqrt(1 - pow((G / L), 2))
    kep[2] = np.arccos(H / G)
    kep[3] = h
    kep[4] = g
    kep[5] = l

    return kep

# analytic J2 propagation
def analyticJ2propHill( x0: array, tof: float, mu: float, rE: float, J2: float, cont: float ):
    
    # cartesian to keplerian elements
    kep0 = cart2kep(x0, mu)

    # keplerian elements to Hill
    hill0 = kep2Hill(kep0, mu)

    # osculating to mean
    hill0Mean = osculating2meanHill(hill0, mu, J2, rE, cont)

    # hill to kep
    kep0Mean = hill2kep(hill0Mean, mu)
    kep0Mean[5] = true2meanAnomaly(kep0Mean[5], kep0Mean[1])

    del0Mean = kep2delaunay(kep0Mean, mu)
    delfMean = averagedJ2rhs(del0Mean, mu, J2, rE, cont)
    delfMean = delfMean*tof+del0Mean

    kepfMean = delaunay2kep(delfMean, mu)
    kepfMean[5] = mean2trueAnomaly(kepfMean[5], kepfMean[1])
    hillfMean = kep2hill(kepfMean, mu)
    hillf = mean2osculatingHill(hillfMean, mu, J2, rE, cont)

    xxf = hill2cart(hillf, mu)

    return xxf


# class of MRPLP using analytic J2 propagation
class mrplp_J2_analytic_parameters():
    def __init__(self, rr1: np.array, rr2: np.array, tof: float, vv1g: np.array, order: int, tol: float, cont: float, dcontMin: float, scl: float, itermax: int, mu: float, rE: float, J2: float):
        self.rr1 = rr1 # initial position vector - (km)
        self.rr2 = rr2 # final position vector - (km)
        self.tof = tof # time of flight - (s)
        self.vv1g = vv1g # initial guess on velocity vector - (km/s)
        self.order = order # order of Taylor expansion
        self.tol = tol # tolerance on convergence radius
        self.cont = cont # continuation (?)
        self.dcontMin = dcontMin # continuation (?)
        self.scl = scl # scaling parameter for the velocity 
        self.itermax = itermax # maximum number of iterations
        self.mu = mu # gravitational parameter - (km3/s2)
        self.rE = rE # radius of the central body - (km)
        self.J2 = J2 # J2 of the central body

# defect analytic J2
def defectAnalyticJ2( params: mrplp_J2_analytic_parameters ):
    st = 1

    # initial guess, target position and time of flight
    rr1 = params.rr1
    vv1 = params.vv1g
    rr2 = params.rr2
    tof = params.tof

    # scaling
    Lsc = params.rE
    Vsc = np.sqrt(params.mu/params.rE)
    Tsc = Lsc/Vsc
    muSc = mu/Lsc/Lsc/Lsc*Tsc*Tsc

    tol = params.tol
    scl = params.scl/Vsc
    scl2 = 1.0
    cont = params.cont

    # apply the scaling
    rr1 = rr1/Lsc
    vv1 = vv1/Vsc
    rr2 = rr2/Lsc
    tof = tof/Tsc

    # Taylor expansion around the initial velocity vector
    x0 = array([ rr1[0], rr1[1], rr1[2], vv1[0]+scl*DA(1), vv1[1]+scl*DA(2), vv1[2]+scl*DA(3) ])

    # propagate
    xfDA = analyticJ2propHill( x0, tof, muSc, Req/Lsc, J2, cont+scl2*DA(4) )

    # compute the map --> difference between propagated and target state
    mapD    = array.zeros(4)
    mapD[0] = xfDA[0] - rr2[0]
    mapD[1] = xfDA[1] - rr2[1]
    mapD[2] = xfDA[2] - rr2[2]
    mapD[3] = DA(4)

    # evaluate the map in the zero perturbation to get the convergence radius
    dxDA    = array.zeros(4)
    dxDA[0] = 0*DA(1)
    dxDA[1] = 0*DA(2)
    dxDA[2] = 0*DA(3)
    dxDA[3] = DA(4)
    dxDA    = mapD.eval(dxDA)

    # convergence radius
    cr = array.zeros(3)
    cr[0] = DA.convRadius(dxDA[0], tol)
    cr[1] = DA.convRadius(dxDA[1], tol)
    cr[2] = DA.convRadius(dxDA[2], tol)

    # new dcont
    J2eps = scl2 * min( min(cr[0].cons(), cr[1].cons()), min(cr[0].cons(), cr[2].cons()) )

    # scale back
    mapD[0:3] = mapD[0:3]*Lsc
    xfDA[0:3] = xfDA[0:3]*Lsc
    xfDA[3:6] = xfDA[3:6]*Vsc

    return mapD, J2eps

# fsolve from map
def fsolveFromMap( dvv1Guess: np.array, mapD: array, dcont: float ):
    res = mapD.eval(dvv1Guess.append(dcont))
    res = res[0:3]
    return [res[0], res[1], res[2]]

# MRPLP using analytic J2 propagation
def mrplp_J2_analytic(params: mrplp_J2_analytic_parameters):
    st = 1

    itermax = params.itermax

    residual = 10
    errormax = 1e-2
    iter = 0
    exit = 0
    epsilon = []

    while (residual > errormax) or (exit < 1 and iter < itermax):
        st = 1

        # compute the maps and the new dcont
        mapD, dcont = defectAnalyticJ2( params )
        if dcont < 1.0e-4 and cont < 1:
            return
        cont = cont + dcont
        if cont >= 1:
            cont = 1
            dcont = 0
            exit = exit + 1
        epsilon.append(cont)

        # solve using fsolve
        dvv1Guess = np.array([0, 0, 0])
        result = fsolve(fsolveFromMap, dvv1Guess, args=(mapD, dcont))

        st = 1
        
def main():

    # initialise DA variables --> DA.init(order, num_variables)
    DA.init(4, 4)

    # initial guess, target position and time of flight
    rr1 = np.array( [-3173.91245750977, -1863.35865746, -6099.31199561] )
    vv1 = np.array( [-5.33966929176339, -2.54974554812496, 4.35921075804661] )
    rr2 = np.array( [6306.80758519, 3249.39062728,  794.06530085] )
    tof = 259200.0

    # scaling
    Lsc = Req
    Vsc = np.sqrt(mu/Req)
    Tsc = Lsc/Vsc
    muSc = mu/Lsc/Lsc/Lsc*Tsc*Tsc
    scl = 1.0e-3
    scl2 = 1.0
    cont = 0.0
    tol = 1.0e-6

    # apply the scaling
    rr1 = rr1/Lsc
    vv1 = vv1/Vsc
    rr2 = rr2/Lsc
    tof = tof/Tsc
    scl = scl/Vsc

    # Taylor expansion around the initial velocity vector
    x0 = array([ rr1[0], rr1[1], rr1[2], vv1[0]+scl*DA(1), vv1[1]+scl*DA(2), vv1[2]+scl*DA(3) ])

    # propagate analytical J2 dynamics --> much faster than numerical
    cont = 0.0
    xfDA = analyticJ2propHill( x0, tof, muSc, Req/Lsc, J2, cont+scl2*DA(4) )

    # compute the map --> difference between propagated and target state
    mapD    = array.zeros(4)
    mapD[0] = xfDA[0] - rr2[0]
    mapD[1] = xfDA[1] - rr2[1]
    mapD[2] = xfDA[2] - rr2[2]
    mapD[3] = DA(4)

    # evaluate the map in the zero perturbation to get the convergence radius
    dxDA    = array.zeros(4)
    dxDA[0] = 0*DA(1)
    dxDA[1] = 0*DA(2)
    dxDA[2] = 0*DA(3)
    dxDA[3] = DA(4)
    dxDA    = mapD.eval(dxDA)

    # convergence radius
    cr = array.zeros(3)
    cr[0] = DA.convRadius(dxDA[0], tol)
    cr[1] = DA.convRadius(dxDA[1], tol)
    cr[2] = DA.convRadius(dxDA[2], tol)

    # new dcont
    J2eps = scl2 * min( min(cr[0].cons(), cr[1].cons()), min(cr[0].cons(), cr[2].cons()) )

    # scale back
    mapD[0:3] = mapD[0:3]*Lsc
    xfDA[0:3] = xfDA[0:3]*Lsc
    xfDA[3:6] = xfDA[3:6]*Vsc

    # evaluate the map in the perturbation
    # mapD.eval([0, 0, 0])

    # save as CSV file
    csv_file_path = "mapD.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        file.write(str(mapD))

    st = 1

if __name__ == "__main__":
    main()