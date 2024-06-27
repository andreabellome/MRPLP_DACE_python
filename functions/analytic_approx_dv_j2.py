import numpy as np

class AnalyticDVApproxJ2:
    """
    This class provides functions for approximate ΔV calculation in Earth-related J2 dynamics.
    
    Reference:
    Shen, H. X., & Casalino, L. (2021). Simple ΔV approximation for optimization of debris-to-debris transfers. 
    Journal of Spacecraft and Rockets, 58(2), 575-580. https://doi.org/10.2514/1.A34831
    """

    def __init__(self):
        pass

    # true to eccentric anomaly
    @staticmethod
    def true2eccAnomaly_numpy( theta: float, e: float ):
        """
        Convert true anomaly to eccentric anomaly using numpy functions.

        Args:
        - theta (float): True anomaly in radians.
        - e (float): Eccentricity of the orbit.

        Returns:
        - float: Eccentric anomaly in radians.
        """

        return 2.0 * np.arctan2(np.sqrt(1. - e)*np.sin(theta / 2.), np.sqrt(1. + e) * np.cos(theta / 2.))

    # true to mean anomaly
    @staticmethod
    def true2meanAnomaly_numpy(theta: float, e: float):
        """
        Convert true anomaly to mean anomaly using numpy functions.

        Args:
        - theta (float): True anomaly in radians.
        - e (float): Eccentricity of the orbit.

        Returns:
        - float: Mean anomaly in radians.
        """

        E = AnalyticDVApproxJ2.true2eccAnomaly_numpy(theta, e)
        return E - e*np.sin(E)

    # mean to eccentric anomaly
    @staticmethod
    def mean2eccAnomaly_numpy( M: float, e: float ):
        """
        Convert mean anomaly to eccentric anomaly using numerical iteration with numpy functions.

        Args:
        - M (float): Mean anomaly in radians.
        - e (float): Eccentricity of the orbit.

        Returns:
        - float: Eccentric anomaly in radians.
        """

        E = M
        for i in range(20):
            E = M + e*np.sin(E)
        
        return E

    # eccentric to true anomaly
    @staticmethod
    def ecc2trueAnomaly_numpy( E: float, e: float ):
        """
        Convert eccentric anomaly to true anomaly using numpy functions.

        Args:
        - E (float): Eccentric anomaly in radians.
        - e (float): Eccentricity of the orbit.

        Returns:
        - float: True anomaly in radians.
        """

        return 2.0 * np.arctan2(np.sqrt(1. + e)*np.sin(E / 2.), np.sqrt(1. - e) * np.cos(E / 2.))

    # mean to true anomaly
    @staticmethod
    def mean2trueAnomaly_numpy( M: float, e: float ):

        """
        Convert mean anomaly to true anomaly using numerical methods with numpy functions.

        Args:
        - M (float): Mean anomaly in radians.
        - e (float): Eccentricity of the orbit.

        Returns:
        - float: True anomaly in radians.
        """

        E = AnalyticDVApproxJ2.mean2eccAnomaly_numpy(M, e)
        return AnalyticDVApproxJ2.ecc2trueAnomaly_numpy(E, e)

    # class of INPUT for approx. DV under Earth-J2 dynamics
    class approx_DV_parameters():

        """
        Class representing input parameters for approximate ΔV calculation under Earth-J2 dynamics.

        Attributes:
        - rr1 (np.array): Initial position vector at time t1 (km).
        - vv1 (np.array): Initial velocity vector at time t1 (km/s).
        - rr2 (np.array): Final position vector at time t2 (km).
        - vv2 (np.array): Final velocity vector at time t2 (km/s).
        - t1 (float): Initial epoch in Modified Julian Date 2000 (MJD2000).
        - t2 (float): Final epoch in Modified Julian Date 2000 (MJD2000).
        - mu (float): Gravitational parameter (km3/s2).
        - rE (float): Radius of the central body (km).
        - J2 (float): J2 coefficient of the central body.
        """

        def __init__(self, rr1: np.array, vv1: np.array, rr2: np.array, vv2: np.array,
                    t1: float, t2: float,
                        mu: float, rE: float, J2: float):
            self.rr1 = rr1 # initial position vector (at time t1) - (km)
            self.vv1 = vv1 # initial velocity vector (at time t1) - (km/s)

            self.rr2 = rr2 # final position vector (at time t2) - (km)
            self.vv2 = vv2 # final velocity vector (at time t2) - (km/s)

            self.t1 = t1 # initial epoch - (MJD2000)
            self.t2 = t2 # final epoch - (MJD2000)

            self.mu = mu # gravitational parameter - (km3/s2)
            self.rE = rE # radius of the central body - (km)
            self.J2 = J2 # J2 of the central body

    class output_meanPropJ2_numpy():

        """
        Class representing output of mean propagation under J2 dynamics in Keplerian elements.

        Attributes:
        - kep (np.array): Keplerian elements [a, e, i, RAAN, omega, theta].
        - cart (np.array): Cartesian elements [x, y, z, vx, vy, vz].
        - Omdot (float): Rate of change of right ascension of the ascending node (rad/s).
        - omdot (float): Rate of change of argument of perigee (rad/s).
        """

        def __init__(self, kep: np.array, cart: np.array,
                    Omdot: float, omdot: float):
            self.kep = kep
            self.cart = cart
            self.Omdot = Omdot
            self.omdot = omdot

    # cartesian elements to keplerian
    @staticmethod
    def cart2kep_numpy(x0: np.array, mu: float):

        """
        Convert Cartesian elements to Keplerian elements using numpy functions.

        Args:
        - x0 (np.array): State vector [rx, ry, rz, vx, vy, vz].
        - mu (float): Gravitational parameter (km3/s2).

        Returns:
        - np.array: Keplerian elements [a, e, i, RAAN, omega, theta].
        """

        x0 = x0.copy()

        rr = x0[0:3]
        vv = x0[3:6]
        r = np.linalg.norm(rr)
        v = np.linalg.norm(vv)
        hh = np.cross(rr, vv)

        sma = mu / ( 2.0 * (mu/r - v**2/2.0) )
        h1sqr = hh[0]**2
        h2sqr = hh[1]**2

        if (h1sqr + h2sqr) == 0.0:
            RAAN = 0.0
        else:
            sinOMEGA = hh[0] / np.sqrt( h1sqr + h2sqr )
            cosOMEGA = -1.0*hh[1] / np.sqrt( h1sqr + h2sqr )
            if cosOMEGA >= 0.0:
                if sinOMEGA >= 0.0:
                    RAAN = np.arcsin( hh[0] / np.sqrt( h1sqr + h2sqr ) )
                else:
                    RAAN = 2.0 * np.pi + np.arcsin( hh[0] / np.sqrt( h1sqr + h2sqr ) )
            else:
                if sinOMEGA >= 0.0:
                    RAAN = np.arccos( -1.0*hh[1] / np.sqrt( h1sqr + h2sqr ) )
                else:
                    RAAN = 2.0 * np.pi - np.arccos( -1.0*hh[1] / np.sqrt( h1sqr + h2sqr ) )

        ee = 1.0/mu * np.cross(vv, hh) - rr/r
        e = np.linalg.norm(ee)
        i = np.arccos( hh[2]/np.linalg.norm(hh) )
        
        if e <= 1.0e-8 and i <= 1e-8:
            e = 0.0
            omega = np.arctan2(rr[1], rr[0])
            theta = 0.0
            kep = np.array( [sma, e, i, RAAN, omega, theta] )
            return kep
        
        if e <= 1.0e-8 and i > 1e-8:
            omega = 0.0
            P = np.zeros(3)
            Q = np.zeros(3)
            W = np.zeros(3)
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
            kep = np.array( [sma, e, i, RAAN, omega, theta] )
            return kep
        
        dotRxE = np.dot(rr, ee)
        RxE = np.linalg.norm(rr)*np.linalg.norm(ee)
        if abs((dotRxE)) > abs((RxE)) and abs((dotRxE)) - abs((RxE)) < abs(1.0e-6*(dotRxE)):
            dotRxE = 1.0e-6*dotRxE
        
        theta = np.arccos(dotRxE / RxE)

        if np.dot(rr, vv) < 0.0:
            theta = 2.0 * np.pi - theta
        
        if (i <= 1.0e-8 and e >= 1.0e-8):
            i = 0.0
            omega = np.arctan2(ee[1], ee[0])
            kep = np.array( [sma, e, i, RAAN, omega, theta] )
            return kep
        
        sino = rr[2] / r / np.sin(i)
        coso = (rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r

        if (coso >= 0.0):
            if sino >= 0.0:
                argLat = np.arcsin(rr[2] / r / np.sin(i))
            else:
                argLat = 2.0 * np.pi + np.arcsin(rr[2] / r / np.sin(i))
        else:
            if coso >= 0.0:
                argLat = np.arccos((rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r)
            else:
                argLat = 2.0 * np.pi - np.arccos((rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r)

        omega = argLat - theta

        if omega < 0.0:
            omega = omega + 2.0 * np.pi
        
        kep = np.array( [sma, e, i, RAAN, omega, theta] )
        return kep

    # mean propagation J2 dynamics
    @staticmethod
    def meanPropJ2_numpy(rr1: np.array, vv1: np.array, tof: float, parameters: approx_DV_parameters):

        """
        Perform mean propagation under J2 dynamics and return Keplerian elements and related rates of change.

        Args:
        - rr1 (np.array): Initial position vector at time t1 (km).
        - vv1 (np.array): Initial velocity vector at time t1 (km/s).
        - tof (float): Time of flight (s).
        - parameters (approx_DV_parameters): Input parameters.

        Returns:
        - output_meanPropJ2_numpy: Output containing Keplerian elements and rates of change.
        """

        state1 = np.append( rr1, vv1 )
        kep1 = AnalyticDVApproxJ2.cart2kep_numpy( state1, parameters.mu )
        a = kep1[0]
        e = kep1[1]
        inc = kep1[2]
        Om1 = kep1[3]
        om1 = kep1[4]
        th = kep1[5]
        M1 = AnalyticDVApproxJ2.true2meanAnomaly_numpy(th, e)

        n = np.sqrt(parameters.mu/(a**3))
        p = a*(1 - e**2)

        Omdot = -1.5*parameters.J2*n*((parameters.rE/p)**2)*np.cos(inc)
        omdot = 0.75*n*parameters.J2*((parameters.rE/p)**2)*(5.0*((np.cos(inc))**2)-1)

        Om = Om1 + Omdot*tof
        om = om1 + omdot*tof
        M  = M1  + n*tof

        while (Om<0.0) or (Om>2.0*np.pi):
            if Om < 0.0:
                Om = Om + 2.0*np.pi
            elif Om>2.0*np.pi:
                Om = Om - 2.0*np.pi
        
        while (om<0.0) or (om>2.0*np.pi):
            if om < 0.0:
                om = om + 2.0*np.pi
            elif om>2.0*np.pi:
                om = om - 2.0*np.pi

        while (M<0.0) or (M>2.0*np.pi):
            if M < 0.0:
                M = M + 2.0*np.pi
            elif M>2.0*np.pi:
                M = M - 2.0*np.pi
        
        th  = AnalyticDVApproxJ2.mean2trueAnomaly_numpy(M, e)
        kep  = np.array([a, e, inc, Om, om, th])

        gam = np.arctan(e*np.sin(th)/(1+e*np.cos(th)))
        r = p/(1+e*np.cos(th))
        v = np.sqrt(2*parameters.mu/r-parameters.mu/a)

        x = r*(np.cos(th+om)*np.cos(Om) - np.sin(th+om)*np.cos(inc)*np.sin(Om))
        y = r*(np.cos(th+om)*np.sin(Om) + np.sin(th+om)*np.cos(inc)*np.cos(Om))
        z = r*(np.sin(th+om)*np.sin(inc))

        vx = v*(-np.sin(th+om-gam)*np.cos(Om)-np.cos(th+om-gam)*np.cos(inc)*np.sin(Om))
        vy = v*(-np.sin(th+om-gam)*np.sin(Om)+np.cos(th+om-gam)*np.cos(inc)*np.cos(Om))
        vz = v*(np.cos(th+om-gam)*np.sin(inc))

        cart = np.array([x, y, z, vx, vy, vz])

        return AnalyticDVApproxJ2.output_meanPropJ2_numpy(kep, cart, Omdot, omdot)
        
    # approx. DV calculation
    @staticmethod
    def approx_DV(parameters: approx_DV_parameters):

        """
        Perform approximate ΔV calculation using input parameters.

        Args:
        - parameters (approx_DV_parameters): Input parameters for ΔV calculation.

        Returns:
        - float: Approximate ΔV magnitude (km/s).
        """

        # time of flight - (s)
        tof = ( parameters.t2 - parameters.t1 )*86400.0

        # cartesian to keplerian elements of the first debris at t1 and t2
        state1 = np.append( parameters.rr1, parameters.vv1 )
        kep_id1_t1 = AnalyticDVApproxJ2.cart2kep_numpy( state1, parameters.mu )

        out_id1_t2 = AnalyticDVApproxJ2.meanPropJ2_numpy(parameters.rr1, parameters.vv1, tof, parameters)
        Om_id1_t2 = out_id1_t2.kep[3]

        Omdot_id1 = out_id1_t2.Omdot
        Om_id1_t1 = kep_id1_t1[3]
        a1 = kep_id1_t1[0]
        e1 = kep_id1_t1[1]
        i1 = kep_id1_t1[2]

        # cartesian to keplerian elements of the second debris at t1 and t2
        state2 = np.append( parameters.rr2, parameters.vv2 )
        kep_id2_t2 = AnalyticDVApproxJ2.cart2kep_numpy( state2, parameters.mu )
        a2  = kep_id2_t2[0]
        e2 = kep_id2_t2[1]
        i2 = kep_id2_t2[2]
        Om_id2_t2 = kep_id2_t2[3]

        out_id2_t1 = AnalyticDVApproxJ2.meanPropJ2_numpy(parameters.rr2, parameters.vv2, -tof, parameters)
        Om_id2_t1 = out_id2_t1.kep[3]
        Omdot_id2 = out_id2_t1.Omdot

        a0     = 0.5*( a1 + a2 )
        v0     = np.sqrt(parameters.mu/a0)
        i0     = 0.5*( i1 + i2 )
        Omdot0 = 0.5*( Omdot_id1 + Omdot_id2 )

        K = np.append(np.arange(0.0, 1000.0, 1), 1000.0)
        topt  = ( Om_id1_t1 - Om_id2_t1 + 2*K*np.pi )/( Omdot_id2 - Omdot_id1 )

        if np.any((topt > tof) & (topt <= tof)):

            # transfer time is sufficient to make the Omegas to be aligned
            dv = v0*np.sqrt( (0.5*(a2 - a1)/a0)**2 + (i2 - i1)**2 + (0.5*(e2 - e1))**2 )
        else:

            # transfer time is too short to make the Omegas to be aligned
            m = -7*Omdot0*np.sin(i0)*tof
            n = -Omdot0*np.sin(i0)*np.tan(i0)*tof
            
            x = ( Om_id2_t2 - Om_id1_t2 )*v0*np.sin(i0)
            y = ( a2 - a1 )/(2*a0)*v0
            z = ( i2 - i1 )*v0
            
            Sx = -( m*y - 2*x + n*z )/( x*( m**2 + n**2 + 4 ) )
            Sy = ( y*n**2 - m*z*n + 4*y + 2*m*x )/( 2*y*( m**2 + n**2 + 4 ) )
            Sz = ( z*m**2 - n*y*m + 4*z + 2*n*x )/( 2*z*( m**2 + n**2 + 4 ) )
            Dx = m*Sy*y + n*Sz*z
            
            dv1 = np.sqrt( (Sx*x)**2 + (Sy*y)**2 + (Sz*z)**2 )
            dv2 = np.sqrt( (x - Sx*x - Dx)**2 + (y - Sy*y)**2 + (z - Sz*z)**2 )
            
            om1 = out_id1_t2.kep[4]
            om2 = kep_id2_t2[4]

            ex_id1 = e1*np.sin( om1 )
            ey_id1 = e1*np.cos( om1 )

            ex_id2 = e2*np.sin( om2 )
            ey_id2 = e2*np.sin( om2 )
            
            dve = 0.5*v0*np.sqrt( (ex_id1 - ex_id2)**2 + (ey_id1 - ey_id2)**2 ) # eccentricity correction on DV

            dv = np.sqrt( dv1**2 + (0.5*dve)**2 ) + np.sqrt( dv2**2 + (0.5*dve)**2 )

        return dv
