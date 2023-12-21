
from typing import Callable, Type

import numpy as np
from numpy.typing import NDArray
from daceypy import DA, array, RK, integrator
from scipy.io import loadmat

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

# define the integrator --> Runge-Kutta 78
def RK78(Y0: array, X0: float, X1: float, f: Callable[[array, float], array]):

    Y0 = Y0.copy()

    N = len(Y0)

    H0 = 0.001
    HS = 0.1
    H1 = 100.0
    EPS = 1.e-12
    BS = 20 * EPS

    Z = array.zeros((N, 16))
    Y1 = array.zeros(N)

    VIHMAX = 0.0

    HSQR = 1.0 / 9.0
    A = np.zeros(13)
    B = np.zeros((13, 12))
    C = np.zeros(13)
    D = np.zeros(13)

    A = np.array([
        0.0, 1.0/18.0, 1.0/12.0, 1.0/8.0, 5.0/16.0, 3.0/8.0, 59.0/400.0,
        93.0/200.0, 5490023248.0/9719169821.0, 13.0/20.0,
        1201146811.0/1299019798.0, 1.0, 1.0,
    ])

    B[1, 0] = 1.0/18.0
    B[2, 0] = 1.0/48.0
    B[2, 1] = 1.0/16.0
    B[3, 0] = 1.0/32.0
    B[3, 2] = 3.0/32.0
    B[4, 0] = 5.0/16.0
    B[4, 2] = -75.0/64.0
    B[4, 3] = 75.0/64.0
    B[5, 0] = 3.0/80.0
    B[5, 3] = 3.0/16.0
    B[5, 4] = 3.0/20.0
    B[6, 0] = 29443841.0/614563906.0
    B[6, 3] = 77736538.0/692538347.0
    B[6, 4] = -28693883.0/1125000000.0
    B[6, 5] = 23124283.0/1800000000.0
    B[7, 0] = 16016141.0/946692911.0
    B[7, 3] = 61564180.0/158732637.0
    B[7, 4] = 22789713.0/633445777.0
    B[7, 5] = 545815736.0/2771057229.0
    B[7, 6] = -180193667.0/1043307555.0
    B[8, 0] = 39632708.0/573591083.0
    B[8, 3] = -433636366.0/683701615.0
    B[8, 4] = -421739975.0/2616292301.0
    B[8, 5] = 100302831.0/723423059.0
    B[8, 6] = 790204164.0/839813087.0
    B[8, 7] = 800635310.0/3783071287.0
    B[9, 0] = 246121993.0/1340847787.0
    B[9, 3] = -37695042795.0/15268766246.0
    B[9, 4] = -309121744.0/1061227803.0
    B[9, 5] = -12992083.0/490766935.0
    B[9, 6] = 6005943493.0/2108947869.0
    B[9, 7] = 393006217.0/1396673457.0
    B[9, 8] = 123872331.0/1001029789.0
    B[10, 0] = -1028468189.0/846180014.0
    B[10, 3] = 8478235783.0/508512852.0
    B[10, 4] = 1311729495.0/1432422823.0
    B[10, 5] = -10304129995.0/1701304382.0
    B[10, 6] = -48777925059.0/3047939560.0
    B[10, 7] = 15336726248.0/1032824649.0
    B[10, 8] = -45442868181.0/3398467696.0
    B[10, 9] = 3065993473.0/597172653.0
    B[11, 0] = 185892177.0/718116043.0
    B[11, 3] = -3185094517.0/667107341.0
    B[11, 4] = -477755414.0/1098053517.0
    B[11, 5] = -703635378.0/230739211.0
    B[11, 6] = 5731566787.0/1027545527.0
    B[11, 7] = 5232866602.0/850066563.0
    B[11, 8] = -4093664535.0/808688257.0
    B[11, 9] = 3962137247.0/1805957418.0
    B[11, 10] = 65686358.0/487910083.0
    B[12, 0] = 403863854.0/491063109.0
    B[12, 3] = - 5068492393.0/434740067.0
    B[12, 4] = -411421997.0/543043805.0
    B[12, 5] = 652783627.0/914296604.0
    B[12, 6] = 11173962825.0/925320556.0
    B[12, 7] = -13158990841.0/6184727034.0
    B[12, 8] = 3936647629.0/1978049680.0
    B[12, 9] = -160528059.0/685178525.0
    B[12, 10] = 248638103.0/1413531060.0

    C = np.array([
        14005451.0/335480064.0, 0.0, 0.0, 0.0, 0.0, -59238493.0/1068277825.0,
        181606767.0/758867731.0, 561292985.0/797845732.0,
        -1041891430.0/1371343529.0, 760417239.0/1151165299.0,
        118820643.0/751138087.0, -528747749.0/2220607170.0, 1.0/4.0,
    ])

    D = np.array([
        13451932.0/455176623.0, 0.0, 0.0, 0.0, 0.0, -808719846.0/976000145.0,
        1757004468.0/5645159321.0, 656045339.0/265891186.0,
        -3867574721.0/1518517206.0, 465885868.0/322736535.0,
        53011238.0/667516719.0, 2.0/45.0, 0.0,
    ])

    Z[:, 0] = Y0

    H = abs(HS)
    HH0 = abs(H0)
    HH1 = abs(H1)
    X = X0
    RFNORM = 0.0
    ERREST = 0.0

    while X != X1:

        # compute new stepsize
        if RFNORM != 0:
            H = H * min(4.0, np.exp(HSQR * np.log(EPS / RFNORM)))
        if abs(H) > abs(HH1):
            H = HH1
        elif abs(H) < abs(HH0) * 0.99:
            H = HH0
            print("--- WARNING, MINIMUM STEPSIZE REACHED IN RK")

        if (X + H - X1) * H > 0:
            H = X1 - X

        for j in range(13):

            for i in range(N):

                Y0[i] = 0.0
                # EVALUATE RHS AT 13 POINTS
                for k in range(j):
                    Y0[i] = Y0[i] + Z[i, k + 3] * B[j, k]

                Y0[i] = H * Y0[i] + Z[i, 0]

            Y1 = f(Y0, X + H * A[j])

            for i in range(N):
                Z[i, j + 3] = Y1[i]

        for i in range(N):

            Z[i, 1] = 0.0
            Z[i, 2] = 0.0
            # EXECUTE 7TH,8TH ORDER STEPS
            for j in range(13):
                Z[i, 1] = Z[i, 1] + Z[i, j + 3] * D[j]
                Z[i, 2] = Z[i, 2] + Z[i, j + 3] * C[j]

            Y1[i] = (Z[i, 2] - Z[i, 1]) * H
            Z[i, 2] = Z[i, 2] * H + Z[i, 0]

        Y1cons = Y1.cons()

        # ESTIMATE ERROR AND DECIDE ABOUT BACKSTEP
        RFNORM = np.linalg.norm(Y1cons, np.inf)  # type: ignore
        if RFNORM > BS and abs(H / H0) > 1.2:
            H = H / 3.0
            RFNORM = 0
        else:
            for i in range(N):
                Z[i, 0] = Z[i, 2]
            X = X + H
            VIHMAX = max(VIHMAX, H)
            ERREST = ERREST + RFNORM

    Y1 = Z[:, 0]

    return Y1

def TBP_J2_scaled(x: array, t: float) -> array:

    Lsc = Req
    Vsc = np.sqrt(mu/Req)
    Tsc = Lsc/Vsc
    muSc = mu/Lsc/Lsc/Lsc*Tsc*Tsc

    Rnorm = Req/Lsc
    muNorm = mu/muSc

    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -muNorm * pos / (r ** 3)
    acc[0] = acc[0] + 1.5*J2*Rnorm**2/(2.*r**5)*(5.*(pos[2]**2)/(r**2) - 1.)*pos[0]
    acc[1] = acc[1] + 1.5*J2*Rnorm**2/(2.*r**5)*(5.*(pos[2]**2)/(r**2) - 1.)*pos[1]
    acc[2] = acc[2] + 1.5*J2*Rnorm**2/(2.*r**5)*(5.*(pos[2]**2)/(r**2) - 3.)*pos[2]
    dx = vel.concat(acc)
    return dx

def TBP_J2(x: array, t: float) -> array:
    pos: array = x[:3]
    vel: array = x[3:]
    r = pos.vnorm()
    acc: array = -mu * pos / (r ** 3)
    acc[0] = acc[0] + 1.5*J2*Req**2/(2.*r**5)*(5.*(pos[2]**2)/(r**2) - 1.)*pos[0]
    acc[1] = acc[1] + 1.5*J2*Req**2/(2.*r**5)*(5.*(pos[2]**2)/(r**2) - 1.)*pos[1]
    acc[2] = acc[2] + 1.5*J2*Req**2/(2.*r**5)*(5.*(pos[2]**2)/(r**2) - 3.)*pos[2]
    dx = vel.concat(acc)
    return dx

class TBP_J2_integrator(integrator):
    def __init__(self, RK: RK.RKCoeff = RK.RK78(),  stateType: Type = array):
        super(TBP_J2_integrator, self).__init__(RK, stateType)

    def f(self, x, t):
        return TBP_J2(x,t)

def main():
    # initialise DA variables --> order: 4, number of variables: 4
    DA.init(4, 4)

    annots = loadmat('solfulFsolveMap.mat')
    solfulFsolveMap = annots['solfulFsolveMap']

    # extract the intial and final states
    rowStruc = 0 # row of the loaded struc
    t1  = solfulFsolveMap[0][rowStruc][2][:, 0][0] # initial epoch - (MJD2000)
    t2  = solfulFsolveMap[0][rowStruc][3][:, 0][0] # arrival epoch - (MJD2000)
    rr1 = solfulFsolveMap[0][rowStruc][4][:, 0]    # initial position vector - (km)
    vv1 = solfulFsolveMap[0][rowStruc][6][:, 0]    # initial velocity vector - (km/s)
    rr2 = solfulFsolveMap[0][rowStruc][5][:, 0]    # arrival position vector - (km)
    vv2 = solfulFsolveMap[0][rowStruc][7][:, 0]    # arrival velocity vector - (km/s)
    tof = (t2 - t1)*86400.0                        # time of flight - s

    xx1J2An = solfulFsolveMap[0][rowStruc][10][:, 0] # solution to MRPLP - (km, km/s)
    xx2J2An = solfulFsolveMap[0][rowStruc][11][:, 0]

    # scaling
    Lsc = Req
    Vsc = np.sqrt(mu/Req)
    Tsc = Lsc/Vsc
    muSc = mu/Lsc/Lsc/Lsc*Tsc*Tsc
    scl = 1.0e-3

    # apply the scaling
    rr1 = rr1/Lsc
    vv1 = vv1/Vsc
    rr2 = rr2/Lsc
    vv2 = vv2/Lsc
    tof = tof/Tsc

    # Taylor expansion around the initial velocity vector
    x0 = array([ rr1[0], rr1[1], rr1[2], vv1[0]+DA(1), vv1[1]+DA(2), vv1[2]+DA(3) ])
    xx1J2AnDA = array([ xx1J2An[0], xx1J2An[1], xx1J2An[2], xx1J2An[3], xx1J2An[4], xx1J2An[5] ])
    xx1J2AnDA[0:3] = xx1J2AnDA[0:3]/Lsc
    xx1J2AnDA[3:6] = xx1J2AnDA[3:6]/Vsc

    # propagate analytical J2 dynamics --> much faster than numerical
    xxf = analyticJ2propHill( xx1J2AnDA, tof, muSc, Req/Lsc, J2, 1.0+1.0*DA(4) )
    xxf[0:3] = xxf[0:3]*Lsc # scale back the position vector - (km)
    xxf[3:6] = xxf[3:6]*Vsc # scale back the velocity vector - (km/s)


    st = 1

if __name__ == "__main__":
    main()