from typing import Callable, Type

import numpy as np
from daceypy import DA, array
from scipy.optimize import fsolve
import time
import warnings

""" In this class, all the functions needed to solve the Multi-Revolution Perturbed Lambert Problem (MRPLP) are given.
    The solution uses the analytic J2 propagation (hopefully it is sufficient...) --> this is because the numerical propagation is very slow.
    The function propagatePerturbedKepler can be used to propagate the perturbed trajectory and to obtain the STM --> use this for plot purposes only. """


# The `MultiRevolutionPerturbedLambertSolver` class provides methods for solving Lambert's problem
# with perturbations using analytic J2 propagation and primer vector techniques.
class MultiRevolutionPerturbedLambertSolver:

    """
    MultiRevolutionPerturbedLambertSolver: A class to solve the Multi-Revolution Perturbed Lambert Problem (MRPLP).

    This class provides methods to solve Lambert's problem with perturbations using analytic J2 propagation
    and primer vector techniques.

    Notes:
        The solution uses analytic J2 propagation due to the slow nature of numerical propagation.
        The function propagatePerturbedKepler can be used to propagate the perturbed trajectory and to obtain the
        state transition matrix (STM); recommended for plotting purposes only.
    """

    def __init__(self):
        pass

    # true to eccentric anomaly
    @staticmethod
    def true2eccAnomaly(theta: DA, e: DA):

        """
        Convert true anomaly to eccentric anomaly using the given eccentricity.

        Args:
            theta (DA): True anomaly.
            e (DA): Eccentricity.

        Returns:
            DA: Eccentric anomaly.
        """

        return 2.0 * np.arctan2(
            np.sqrt(1.0 - e) * np.sin(theta / 2.0),
            np.sqrt(1.0 + e) * np.cos(theta / 2.0),
        )

    # true to mean anomaly
    @staticmethod
    def true2meanAnomaly(theta: DA, e: DA):

        """
        Convert true anomaly to mean anomaly using the given eccentricity.

        Args:
            theta (DA): True anomaly.
            e (DA): Eccentricity.

        Returns:
            DA: Mean anomaly.
        """


        E = MultiRevolutionPerturbedLambertSolver.true2eccAnomaly(theta, e)
        return E - e * np.sin(E)

    # mean to eccentric anomaly
    @staticmethod
    def mean2eccAnomaly(M: DA, e: DA):
        """
        Convert mean anomaly to eccentric anomaly using the given eccentricity.

        Args:
            M (DA): Mean anomaly.
            e (DA): Eccentricity.

        Returns:
            DA: Eccentric anomaly.
        """
        E = M
        for i in range(20):
            E = M + e * np.sin(E)

        return E

    # eccentric to true anomaly
    @staticmethod
    def ecc2trueAnomaly(E: DA, e: DA):
        """
        Convert eccentric anomaly to true anomaly using the given eccentricity.

        Args:
            E (DA): Mean anomaly.
            e (DA): Eccentricity.

        Returns:
            DA: True anomaly.
        """
        return 2.0 * np.arctan2(
            np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0)
        )

    # mean to true anomaly
    @staticmethod
    def mean2trueAnomaly(M: DA, e: DA):
        """
        Convert mean anomaly to true anomaly using the given eccentricity.

        Args:
            M (DA): Mean anomaly.
            e (DA): Eccentricity.

        Returns:
            DA: True anomaly.
        """
        E = MultiRevolutionPerturbedLambertSolver.mean2eccAnomaly(M, e)
        return MultiRevolutionPerturbedLambertSolver.ecc2trueAnomaly(E, e)

    # cartesian elements to keplerian
    @staticmethod
    def cart2kep(x0: array, mu: float):
        """
        Convert Cartesian elements to Keplerian elements.

        Parameters:
        x0 (array): Cartesian elements [x, y, z, vx, vy, vz].
        mu (float): Standard gravitational parameter.

        Returns:
        array: Keplerian elements [a, e, i, RAAN, omega, theta].
        """

        x0 = x0.copy()

        rr = x0[0:3]
        vv = x0[3:6]
        r = np.linalg.norm(rr)
        v = np.linalg.norm(vv)
        hh = np.cross(rr, vv)

        sma = mu / (2.0 * (mu / r - v**2 / 2.0))
        h1sqr = hh[0] ** 2
        h2sqr = hh[1] ** 2

        if (h1sqr + h2sqr).cons() == 0.0:
            RAAN = 0.0
        else:
            sinOMEGA = hh[0] / np.sqrt(h1sqr + h2sqr)
            cosOMEGA = -1.0 * hh[1] / np.sqrt(h1sqr + h2sqr)
            if cosOMEGA.cons() >= 0.0:
                if sinOMEGA.cons() >= 0.0:
                    RAAN = np.arcsin(hh[0] / np.sqrt(h1sqr + h2sqr))
                else:
                    RAAN = 2.0 * np.pi + np.arcsin(hh[0] / np.sqrt(h1sqr + h2sqr))
            else:
                if sinOMEGA.cons() >= 0.0:
                    RAAN = np.arccos(-1.0 * hh[1] / np.sqrt(h1sqr + h2sqr))
                else:
                    RAAN = 2.0 * np.pi - np.arccos(
                        -1.0 * hh[1] / np.sqrt(h1sqr + h2sqr)
                    )

        ee = 1.0 / mu * np.cross(vv, hh) - rr / r
        e = np.linalg.norm(ee)
        i = np.arccos(hh[2] / np.linalg.norm(hh))

        if e.cons() <= 1.0e-8 and i.cons() <= 1e-8:
            e = 0.0
            omega = np.arctan2(rr[1], rr[0])
            theta = 0.0
            kep = array([sma, e, i, RAAN, omega, theta])
            return kep

        if e.cons() <= 1.0e-8 and i.cons() > 1e-8:
            omega = 0.0
            P = array.zeros(3)
            Q = array.zeros(3)
            W = array.zeros(3)
            P[0] = np.cos(omega) * np.cos(RAAN) - np.sin(omega) * np.sin(i) * np.sin(
                RAAN
            )
            P[1] = -1.0 * np.sin(omega) * np.cos(RAAN) - np.cos(omega) * np.cos(
                i
            ) * np.sin(RAAN)
            P[2] = np.sin(RAAN) * np.sin(i)
            Q[0] = np.cos(omega) * np.sin(RAAN) + np.sin(omega) * np.cos(i) * np.cos(
                RAAN
            )
            Q[1] = -1.0 * np.sin(omega) * np.sin(RAAN) + np.cos(omega) * np.cos(
                i
            ) * np.cos(RAAN)
            Q[2] = -1.0 * np.cos(RAAN) * np.sin(i)
            W[0] = np.sin(omega) * np.sin(i)
            W[1] = np.cos(omega) * np.sin(i)
            W[2] = np.cos(i)
            rrt = P * rr[0] + Q * rr[1] + W * rr[2]
            theta = np.arctan2(rrt[1], rrt[0])
            kep = array([sma, e, i, RAAN, omega, theta])
            return kep

        dotRxE = np.dot(rr, ee)
        RxE = np.linalg.norm(rr) * np.linalg.norm(ee)
        if abs((dotRxE).cons()) > abs((RxE).cons()) and abs((dotRxE).cons()) - abs(
            (RxE).cons()
        ) < abs(1.0e-6 * (dotRxE).cons()):
            dotRxE = 1.0e-6 * dotRxE

        theta = np.arccos(dotRxE / RxE)

        if np.dot(rr, vv).cons() < 0.0:
            theta = 2.0 * np.pi - theta

        if i.cons() <= 1.0e-8 and e.cons() >= 1.0e-8:
            i = 0.0
            omega = np.arctan2(ee[1], ee[0])
            kep = array([sma, e, i, RAAN, omega, theta])
            return kep

        sino = rr[2] / r / np.sin(i)
        coso = (rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r

        if coso.cons() >= 0.0:
            if sino.cons() >= 0.0:
                argLat = np.arcsin(rr[2] / r / np.sin(i))
            else:
                argLat = 2.0 * np.pi + np.arcsin(rr[2] / r / np.sin(i))
        else:
            if coso.cons() >= 0.0:
                argLat = np.arccos((rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r)
            else:
                argLat = 2.0 * np.pi - np.arccos(
                    (rr[0] * np.cos(RAAN) + rr[1] * np.sin(RAAN)) / r
                )

        omega = argLat - theta

        if omega.cons() < 0.0:
            omega = omega + 2.0 * np.pi

        kep = array([sma, e, i, RAAN, omega, theta])
        return kep

    # keplerian elements to Hill
    def kep2Hill(kep: array, mu: float):
        """
        Convert Keplerian elements to Hill coordinates.

        Parameters:
        kep (array): Keplerian elements [a, e, i, RAAN, omega, theta].
        mu (float): Standard gravitational parameter.

        Returns:
        array: Hill coordinates [l, g, h, L, G, H].
        """

        hill = array.zeros(6)
        p = kep[0] * (1.0 - kep[1] * kep[1])
        f = kep[5]

        hill[4] = np.sqrt(mu * p)
        hill[0] = p / (1.0 + kep[1] * np.cos(f))
        hill[1] = f + kep[4]
        hill[2] = kep[3]
        hill[3] = (hill[4] / p) * kep[1] * np.sin(f)
        hill[5] = hill[4] * np.cos(kep[2])

        return hill

    # Hill to cartesian elements
    @staticmethod
    def hill2cart(hill: array, mu: float):
        r = hill[0]
        th = hill[1]
        nu = hill[2]
        R = hill[3]
        Th = hill[4]
        ci = hill[5] / hill[4]
        si = np.sqrt(1.0 - ci * ci)

        u = array.zeros(3)
        u[0] = np.cos(th) * np.cos(nu) - ci * np.sin(th) * np.sin(nu)
        u[1] = np.cos(th) * np.sin(nu) + ci * np.sin(th) * np.cos(nu)
        u[2] = si * np.sin(th)

        cart = array.zeros(6)
        cart[0] = r * u[0]
        cart[1] = r * u[1]
        cart[2] = r * u[2]
        cart[3] = (R * np.cos(th) - Th * np.sin(th) / r) * np.cos(nu) - (
            R * np.sin(th) + Th * np.cos(th) / r
        ) * np.sin(nu) * ci
        cart[4] = (R * np.cos(th) - Th * np.sin(th) / r) * np.sin(nu) + (
            R * np.sin(th) + Th * np.cos(th) / r
        ) * np.cos(nu) * ci
        cart[5] = (R * np.sin(th) + Th * np.cos(th) / r) * si

        return cart

    # from osculating to mean Hill
    @staticmethod
    def osculating2meanHill(
        hillOsc: array, mu: float, J2: float, rE: float, cont: float
    ):
        r = hillOsc[0]
        th = hillOsc[1]
        nu = hillOsc[2]
        R = hillOsc[3]
        Th = hillOsc[4]
        Nu = hillOsc[5]

        ci = Nu / Th
        si = np.sqrt(1.0 - ci * ci)
        cs = (-1.0 + pow(Th, 2) / (mu * r)) * np.cos(th) + (R * Th * np.sin(th)) / mu
        ss = -((R * Th * np.cos(th)) / mu) + (-1.0 + pow(Th, 2) / (mu * r)) * np.sin(th)
        e = np.sqrt(cs * cs + ss * ss)
        eta = np.sqrt(1.0 - e * e)

        beta = 1.0 / (1.0 + eta)

        p = Th * Th / mu
        costrue = 1 / e * (p / r - 1)

        f = np.arccos(costrue)

        if R.cons() < 0.0:
            f = 2.0 * np.pi - f

        M = MultiRevolutionPerturbedLambertSolver.true2meanAnomaly(f, e)
        phi = f - M

        rMean = r + (cont) * (
            (pow(rE, 2) * beta * J2) / (2.0 * r)
            - (3 * pow(rE, 2) * beta * J2 * pow(si, 2)) / (4.0 * r)
            + (pow(rE, 2) * eta * J2 * pow(mu, 2) * r) / pow(Th, 4)
            - (3 * pow(rE, 2) * eta * J2 * pow(mu, 2) * r * pow(si, 2))
            / (2.0 * pow(Th, 4))
            + (pow(rE, 2) * J2 * mu) / (2.0 * pow(Th, 2))
            - (pow(rE, 2) * beta * J2 * mu) / (2.0 * pow(Th, 2))
            - (3.0 * pow(rE, 2) * J2 * mu * pow(si, 2)) / (4.0 * pow(Th, 2))
            + (3 * pow(rE, 2) * beta * J2 * mu * pow(si, 2)) / (4.0 * pow(Th, 2))
            - (pow(rE, 2) * J2 * mu * pow(si, 2) * np.cos(2 * th)) / (4.0 * pow(Th, 2))
        )

        thMean = th + (cont) * (
            (-3.0 * pow(rE, 2) * J2 * pow(mu, 2) * phi) / pow(Th, 4)
            + (15.0 * pow(rE, 2) * J2 * pow(mu, 2) * phi * pow(si, 2))
            / (4.0 * pow(Th, 4))
            - (5.0 * pow(rE, 2) * J2 * mu * R) / (2.0 * pow(Th, 3))
            - (pow(rE, 2) * beta * J2 * mu * R) / (2.0 * pow(Th, 3))
            + (3.0 * pow(rE, 2) * J2 * mu * R * pow(si, 2)) / pow(Th, 3)
            + (3.0 * pow(rE, 2) * beta * J2 * mu * R * pow(si, 2)) / (4.0 * pow(Th, 3))
            - (pow(rE, 2) * beta * J2 * R) / (2.0 * r * Th)
            + (3.0 * pow(rE, 2) * beta * J2 * R * pow(si, 2)) / (4.0 * r * Th)
            + (
                -(pow(rE, 2) * J2 * mu * R) / (2.0 * pow(Th, 3))
                + (pow(rE, 2) * J2 * mu * R * pow(si, 2)) / pow(Th, 3)
            )
            * np.cos(2.0 * th)
            + (
                -(pow(rE, 2) * J2 * pow(mu, 2)) / (4.0 * pow(Th, 4))
                + (5.0 * pow(rE, 2) * J2 * pow(mu, 2) * pow(si, 2)) / (8.0 * pow(Th, 4))
                + (pow(rE, 2) * J2 * mu) / (r * pow(Th, 2))
                - (3.0 * pow(rE, 2) * J2 * mu * pow(si, 2)) / (2.0 * r * pow(Th, 2))
            )
            * np.sin(2.0 * th)
        )

        nuMean = nu + (cont) * (
            (3.0 * pow(rE, 2) * ci * J2 * pow(mu, 2) * phi) / (2.0 * pow(Th, 4))
            + (3.0 * pow(rE, 2) * ci * J2 * mu * R) / (2.0 * pow(Th, 3))
            + (pow(rE, 2) * ci * J2 * mu * R * np.cos(2.0 * th)) / (2.0 * pow(Th, 3))
            + (
                (pow(rE, 2) * ci * J2 * pow(mu, 2)) / (4.0 * pow(Th, 4))
                - (pow(rE, 2) * ci * J2 * mu) / (r * pow(Th, 2))
            )
            * np.sin(2.0 * th)
        )

        RMean = R + (cont) * (
            -(pow(rE, 2) * beta * J2 * R) / (2.0 * pow(r, 2))
            + (3.0 * pow(rE, 2) * beta * J2 * R * pow(si, 2)) / (4.0 * pow(r, 2))
            - (pow(rE, 2) * eta * J2 * pow(mu, 2) * R) / (2.0 * pow(Th, 4))
            + (3.0 * pow(rE, 2) * eta * J2 * pow(mu, 2) * R * pow(si, 2))
            / (4.0 * pow(Th, 4))
            + (pow(rE, 2) * J2 * mu * pow(si, 2) * np.sin(2.0 * th))
            / (2.0 * pow(r, 2) * Th)
        )

        ThMean = Th + (cont) * (
            (
                (pow(rE, 2) * J2 * pow(mu, 2) * pow(si, 2)) / (4.0 * pow(Th, 3))
                - (pow(rE, 2) * J2 * mu * pow(si, 2)) / (r * Th)
            )
            * np.cos(2.0 * th)
            - (pow(rE, 2) * J2 * mu * R * pow(si, 2) * np.sin(2.0 * th))
            / (2.0 * pow(Th, 2))
        )

        NuMean = Nu + 0.0

        hillMean = array([rMean, thMean, nuMean, RMean, ThMean, NuMean])

        return hillMean

    # from mean to osculating Hill
    @staticmethod
    def mean2osculatingHill(
        hillMean: array, mu: float, J2: float, rE: float, cont: float
    ):
        r = hillMean[0]  # l
        th = hillMean[1]  # g
        nu = hillMean[2]  # h
        R = hillMean[3]  # L
        Th = hillMean[4]  # G
        Nu = hillMean[5]  # H
        ci = Nu / Th
        si = np.sqrt(1.0 - ci * ci)
        cs = (-1.0 + pow(Th, 2) / (mu * r)) * np.cos(th) + (R * Th * np.sin(th)) / mu
        ss = -((R * Th * np.cos(th)) / mu) + (-1.0 + pow(Th, 2) / (mu * r)) * np.sin(th)
        e = np.sqrt(cs * cs + ss * ss)
        eta = np.sqrt(1.0 - e * e)
        beta = 1.0 / (1.0 + eta)
        p = Th * Th / mu
        costrue = 1 / e * (p / r - 1)
        f = np.arccos(costrue)

        if R.cons() < 0.0:
            f = 2.0 * np.pi - f

        M = MultiRevolutionPerturbedLambertSolver.true2meanAnomaly(f, e)
        phi = f - M

        rOsc = r - (cont) * (
            (pow(rE, 2) * beta * J2) / (2.0 * r)
            - (3 * pow(rE, 2) * beta * J2 * pow(si, 2)) / (4.0 * r)
            + (pow(rE, 2) * eta * J2 * pow(mu, 2) * r) / pow(Th, 4)
            - (3 * pow(rE, 2) * eta * J2 * pow(mu, 2) * r * pow(si, 2))
            / (2.0 * pow(Th, 4))
            + (pow(rE, 2) * J2 * mu) / (2.0 * pow(Th, 2))
            - (pow(rE, 2) * beta * J2 * mu) / (2.0 * pow(Th, 2))
            - (3.0 * pow(rE, 2) * J2 * mu * pow(si, 2)) / (4.0 * pow(Th, 2))
            + (3 * pow(rE, 2) * beta * J2 * mu * pow(si, 2)) / (4.0 * pow(Th, 2))
            - (pow(rE, 2) * J2 * mu * pow(si, 2) * np.cos(2 * th)) / (4.0 * pow(Th, 2))
        )

        thOsc = th - (cont) * (
            (-3.0 * pow(rE, 2) * J2 * pow(mu, 2) * phi) / pow(Th, 4)
            + (15.0 * pow(rE, 2) * J2 * pow(mu, 2) * phi * pow(si, 2))
            / (4.0 * pow(Th, 4))
            - (5.0 * pow(rE, 2) * J2 * mu * R) / (2.0 * pow(Th, 3))
            - (pow(rE, 2) * beta * J2 * mu * R) / (2.0 * pow(Th, 3))
            + (3.0 * pow(rE, 2) * J2 * mu * R * pow(si, 2)) / pow(Th, 3)
            + (3.0 * pow(rE, 2) * beta * J2 * mu * R * pow(si, 2)) / (4.0 * pow(Th, 3))
            - (pow(rE, 2) * beta * J2 * R) / (2.0 * r * Th)
            + (3.0 * pow(rE, 2) * beta * J2 * R * pow(si, 2)) / (4.0 * r * Th)
            + (
                -(pow(rE, 2) * J2 * mu * R) / (2.0 * pow(Th, 3))
                + (pow(rE, 2) * J2 * mu * R * pow(si, 2)) / pow(Th, 3)
            )
            * np.cos(2.0 * th)
            + (
                -(pow(rE, 2) * J2 * pow(mu, 2)) / (4.0 * pow(Th, 4))
                + (5.0 * pow(rE, 2) * J2 * pow(mu, 2) * pow(si, 2)) / (8.0 * pow(Th, 4))
                + (pow(rE, 2) * J2 * mu) / (r * pow(Th, 2))
                - (3.0 * pow(rE, 2) * J2 * mu * pow(si, 2)) / (2.0 * r * pow(Th, 2))
            )
            * np.sin(2.0 * th)
        )

        nuOsc = nu - (cont) * (
            (3.0 * pow(rE, 2) * ci * J2 * pow(mu, 2) * phi) / (2.0 * pow(Th, 4))
            + (3.0 * pow(rE, 2) * ci * J2 * mu * R) / (2.0 * pow(Th, 3))
            + (pow(rE, 2) * ci * J2 * mu * R * np.cos(2.0 * th)) / (2.0 * pow(Th, 3))
            + (
                (pow(rE, 2) * ci * J2 * pow(mu, 2)) / (4.0 * pow(Th, 4))
                - (pow(rE, 2) * ci * J2 * mu) / (r * pow(Th, 2))
            )
            * np.sin(2.0 * th)
        )

        ROsc = R - (cont) * (
            -(pow(rE, 2) * beta * J2 * R) / (2.0 * pow(r, 2))
            + (3.0 * pow(rE, 2) * beta * J2 * R * pow(si, 2)) / (4.0 * pow(r, 2))
            - (pow(rE, 2) * eta * J2 * pow(mu, 2) * R) / (2.0 * pow(Th, 4))
            + (3.0 * pow(rE, 2) * eta * J2 * pow(mu, 2) * R * pow(si, 2))
            / (4.0 * pow(Th, 4))
            + (pow(rE, 2) * J2 * mu * pow(si, 2) * np.sin(2.0 * th))
            / (2.0 * pow(r, 2) * Th)
        )

        ThOsc = Th - (cont) * (
            (
                (pow(rE, 2) * J2 * pow(mu, 2) * pow(si, 2)) / (4.0 * pow(Th, 3))
                - (pow(rE, 2) * J2 * mu * pow(si, 2)) / (r * Th)
            )
            * np.cos(2.0 * th)
            - (pow(rE, 2) * J2 * mu * R * pow(si, 2) * np.sin(2.0 * th))
            / (2.0 * pow(Th, 2))
        )

        NuOsc = Nu + 0.0

        hillOsc = array.zeros(6)

        hillOsc[0] = rOsc
        hillOsc[1] = thOsc
        hillOsc[2] = nuOsc
        hillOsc[3] = ROsc
        hillOsc[4] = ThOsc
        hillOsc[5] = NuOsc

        return hillOsc

    # Hill to keplerian
    @staticmethod
    def hill2kep(hill: array, mu: float):
        r = hill[0]
        th = hill[1]
        nu = hill[2]
        R = hill[3]
        Th = hill[4]
        Nu = hill[5]

        i = np.arccos(Nu / Th)
        cs = (-1.0 + pow(Th, 2) / (mu * r)) * np.cos(th) + (R * Th * np.sin(th)) / mu
        ss = -((R * Th * np.cos(th)) / mu) + (-1.0 + pow(Th, 2) / (mu * r)) * np.sin(th)
        e = np.sqrt(cs * cs + ss * ss)
        p = Th * Th / mu
        costrue = 1.0 / e * (p / r - 1.0)
        f = np.arccos(costrue)

        if R.cons() < 0.0:
            f = 2.0 * np.pi - f

        a = p / (1 - e * e)

        kep = array([a, e, i, nu, th - f, f])
        return kep

    # keplerian to Hill
    def kep2hill(kep: array, mu: float):
        p = kep[0] * (1.0 - kep[1] * kep[1])
        f = kep[5]

        hill = array.zeros(6)
        hill[4] = np.sqrt(mu * p)
        hill[0] = p / (1.0 + kep[1] * np.cos(f))
        hill[1] = f + kep[4]
        hill[2] = kep[3]
        hill[3] = (hill[4] / p) * kep[1] * np.sin(f)
        hill[5] = hill[4] * np.cos(kep[2])

        return hill

    # kep to delaunay
    @staticmethod
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
        delaunay[3] = np.sqrt(mu * a)
        delaunay[4] = np.sqrt(1 - pow(e, 2)) * delaunay[3]
        delaunay[5] = np.cos(i) * delaunay[4]

        return delaunay

    # averaged Delaunay
    @staticmethod
    def averagedJ2rhs(xxx: array, mu: float, J2: float, rE: float, cont: float):

        l = xxx[0]
        g = xxx[1]
        h = xxx[2]
        L = xxx[3]
        G = xxx[4]
        H = xxx[5]
        eta = G / L
        ci = H / G
        si = np.sin(np.arccos(ci))

        dldt = pow(mu, 2) / pow(L, 3) + (cont) * (
            (3.0 * J2 * pow(rE, 2) * pow(mu, 4)) / (2.0 * pow(L, 7) * pow(eta, 3))
            - (9.0 * J2 * pow(si, 2) * pow(rE, 2) * pow(mu, 4))
            / (4.0 * pow(L, 7) * pow(eta, 3))
        )

        dgdt = (cont) * (
            (3.0 * J2 * pow(rE, 2) * pow(mu, 4)) / (2.0 * pow(L, 7) * pow(eta, 4))
            - (9.0 * J2 * pow(si, 2) * pow(rE, 2) * pow(mu, 4))
            / (4.0 * pow(L, 7) * pow(eta, 4))
            + (3.0 * pow(ci, 2) * J2 * pow(rE, 2) * pow(mu, 4))
            / (2.0 * G * pow(L, 6) * pow(eta, 3))
        )

        dhdt = (cont) * (
            -(3.0 * pow(ci, 2) * J2 * pow(rE, 2) * pow(mu, 4))
            / (2.0 * H * pow(L, 6) * pow(eta, 3))
        )

        ff = array.zeros(6)
        ff[0] = dldt
        ff[1] = dgdt
        ff[2] = dhdt
        ff[3] = 0
        ff[4] = 0
        ff[5] = 0

        return ff

    # Delaunay to kep
    @staticmethod
    def delaunay2kep(delaunay: array, mu: float):
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
    @staticmethod
    def analyticJ2propHill(
        x0: array, tof: float, mu: float, rE: float, J2: float, cont: float
    ):

        # cartesian to keplerian elements
        kep0 = MultiRevolutionPerturbedLambertSolver.cart2kep(x0, mu)

        # keplerian elements to Hill
        hill0 = MultiRevolutionPerturbedLambertSolver.kep2Hill(kep0, mu)

        # osculating to mean
        hill0Mean = MultiRevolutionPerturbedLambertSolver.osculating2meanHill(
            hill0, mu, J2, rE, cont
        )

        # hill to kep
        kep0Mean = MultiRevolutionPerturbedLambertSolver.hill2kep(hill0Mean, mu)
        kep0Mean[5] = MultiRevolutionPerturbedLambertSolver.true2meanAnomaly(
            kep0Mean[5], kep0Mean[1]
        )

        del0Mean = MultiRevolutionPerturbedLambertSolver.kep2delaunay(kep0Mean, mu)
        delfMean = MultiRevolutionPerturbedLambertSolver.averagedJ2rhs(
            del0Mean, mu, J2, rE, cont
        )
        delfMean = delfMean * tof + del0Mean

        kepfMean = MultiRevolutionPerturbedLambertSolver.delaunay2kep(delfMean, mu)
        kepfMean[5] = MultiRevolutionPerturbedLambertSolver.mean2trueAnomaly(
            kepfMean[5], kepfMean[1]
        )
        hillfMean = MultiRevolutionPerturbedLambertSolver.kep2hill(kepfMean, mu)
        hillf = MultiRevolutionPerturbedLambertSolver.mean2osculatingHill(
            hillfMean, mu, J2, rE, cont
        )

        xxf = MultiRevolutionPerturbedLambertSolver.hill2cart(hillf, mu)

        return xxf

    # class of INPUT for MRPLP using analytic J2 propagation
    class mrplp_J2_analytic_parameters:
        def __init__(
            self,
            rr1: np.array,
            rr2: np.array,
            tof: float,
            vv1g: np.array,
            mu: float,
            rE: float,
            J2: float,
            order: int,
            tol: float,
            cont: float,
            dcontMin: float,
            scl: float,
            itermax: int,
        ):
            self.rr1 = rr1  # initial position vector - (km)
            self.rr2 = rr2  # final position vector - (km)
            self.tof = tof  # time of flight - (s)
            self.vv1g = vv1g  # initial guess on velocity vector - (km/s)
            self.order = order  # order of Taylor expansion
            self.tol = tol  # tolerance on convergence radius
            self.cont = cont  # continuation (?)
            self.dcontMin = dcontMin  # continuation (?)
            self.scl = scl  # scaling parameter for the velocity
            self.itermax = itermax  # maximum number of iterations
            self.mu = mu  # gravitational parameter - (km3/s2)
            self.rE = rE  # radius of the central body - (km)
            self.J2 = J2  # J2 of the central body

    # class of OUTPUT for MRPLP using analytic J2 propagation
    class output_mrplp_J2_analytic:
        def __init__(
            self,
            vv1Sol: np.array,
            vv2Sol: np.array,
            rr2DA: np.array,
            res: list,
            iter: int,
            elapsed_time: float,
            success: bool,
        ):
            self.vv1Sol = vv1Sol
            self.vv2Sol = vv2Sol
            self.rr2DA = rr2DA
            self.res = res
            self.iter = iter
            self.elapsed_time = elapsed_time
            self.success = success

    # defect analytic J2
    @staticmethod
    def defectAnalyticJ2(params: mrplp_J2_analytic_parameters):

        # initialise DA variables --> DA.init(order, num_variables)
        DA.init(params.order, 4)

        # initial guess, target position and time of flight
        rr1 = params.rr1
        vv1 = params.vv1g
        rr2 = params.rr2
        tof = params.tof

        # scaling
        Lsc = params.rE
        Vsc = np.sqrt(params.mu / params.rE)
        Tsc = Lsc / Vsc
        muSc = params.mu / Lsc / Lsc / Lsc * Tsc * Tsc

        tol = params.tol
        scl = params.scl / Vsc
        scl2 = 1.0
        cont = params.cont

        # apply the scaling
        rr1 = rr1 / Lsc
        vv1 = vv1 / Vsc
        rr2 = rr2 / Lsc
        tof = tof / Tsc

        # Taylor expansion around the initial velocity vector
        x0 = array(
            [
                rr1[0],
                rr1[1],
                rr1[2],
                vv1[0] + scl * DA(1),
                vv1[1] + scl * DA(2),
                vv1[2] + scl * DA(3),
            ]
        )

        # propagate
        xfDA = MultiRevolutionPerturbedLambertSolver.analyticJ2propHill(
            x0, tof, muSc, params.rE / Lsc, params.J2, cont + scl2 * DA(4)
        )

        # compute the map --> difference between propagated and target state
        mapD = array.zeros(4)
        mapD[0] = xfDA[0] - rr2[0]
        mapD[1] = xfDA[1] - rr2[1]
        mapD[2] = xfDA[2] - rr2[2]
        mapD[3] = DA(4)

        # evaluate the map in the zero perturbation to get the convergence radius
        dxDA = array.zeros(4)
        dxDA[0] = 0 * DA(1)
        dxDA[1] = 0 * DA(2)
        dxDA[2] = 0 * DA(3)
        dxDA[3] = DA(4)
        dxDA = mapD.eval(dxDA)

        # convergence radius
        cr = array.zeros(3)
        cr[0] = DA.convRadius(dxDA[0], tol)
        cr[1] = DA.convRadius(dxDA[1], tol)
        cr[2] = DA.convRadius(dxDA[2], tol)

        # new dcont
        J2eps = scl2 * min(
            min(cr[0].cons(), cr[1].cons()), min(cr[0].cons(), cr[2].cons())
        )

        # scale back
        mapD[0:3] = mapD[0:3] * Lsc
        xfDA[0:3] = xfDA[0:3] * Lsc
        xfDA[3:6] = xfDA[3:6] * Vsc

        return mapD, J2eps, xfDA.cons()

    # fsolve from map
    @staticmethod
    def fsolveFromMap(dvv1Guess: np.array, mapD: array, dcont: float):
        res = mapD.eval(np.append(dvv1Guess, dcont))
        res = res[0:3]
        return [res[0], res[1], res[2]]

    # MRPLP using analytic J2 propagation
    @staticmethod
    def mrplp_J2_analytic(params: mrplp_J2_analytic_parameters):

        itermax = params.itermax
        cont = params.cont

        residual = 10
        errormax = 1e-3
        iter = 0
        exit = 0
        epsilon = []
        res = []

        # start counting the elapsed time
        start_time = time.time()

        while (residual > errormax) or (exit < 1 and iter < itermax):

            # update the iteration number
            iter = iter + 1

            # solution did not converge --> try with different initial guess
            if iter > itermax:
                if residual > errormax:
                    vv1Sol = np.ones(3) * np.nan
                    vv2Sol = np.ones(3) * np.nan
                    rr2DA = np.ones(3) * np.nan
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    success = False
                    return (
                        MultiRevolutionPerturbedLambertSolver.output_mrplp_J2_analytic(
                            vv1Sol, vv2Sol, rr2DA, res, iter, elapsed_time, success
                        )
                    )

            # compute the maps and the new dcont
            mapD, dcont, finalState = (
                MultiRevolutionPerturbedLambertSolver.defectAnalyticJ2(params)
            )
            if dcont < 1.0e-4 and cont < 1:
                vv1Sol = np.ones(3) * np.nan
                vv2Sol = np.ones(3) * np.nan
                rr2DA = np.ones(3) * np.nan
                end_time = time.time()
                elapsed_time = end_time - start_time
                success = False
                return MultiRevolutionPerturbedLambertSolver.output_mrplp_J2_analytic(
                    vv1Sol, vv2Sol, rr2DA, res, iter, elapsed_time, success
                )
            cont = cont + dcont
            if cont >= 1:
                cont = 1.0
                dcont = 0.0
                exit = exit + 1
            epsilon.append(cont)
            residual = np.linalg.norm(
                mapD.eval([0, 0, 0, 0])[0:3]
            )  # check if you achieved the required target point
            res.append(residual)

            # solve using fsolve and the maps
            dvv1Guess = np.array([0, 0, 0])
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning
            )  # suppress the warnings about the run time
            result = fsolve(
                MultiRevolutionPerturbedLambertSolver.fsolveFromMap,
                dvv1Guess,
                args=(mapD, dcont),
            )

            # update the params
            params.vv1g = params.vv1g + params.scl * result  # this is the solution
            params.cont = cont

        # count the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # extract the solution
        vv1Sol = params.vv1g
        rr2DA = finalState[0:3]
        vv2Sol = finalState[3:6]
        success = True

        return MultiRevolutionPerturbedLambertSolver.output_mrplp_J2_analytic(
            vv1Sol, vv2Sol, rr2DA, res, iter, elapsed_time, success
        )

    # propagate in J2 dynamics and obtain the STM w.r.t. time
    @staticmethod
    def propagatePerturbedKeplerSTM(
        rr1: np.array,
        vv1: np.array,
        tof: float,
        Npoints: float,
        params: mrplp_J2_analytic_parameters,
    ):
        """This function is used to propagate the initial state and derive the State Transition Matrix.
        Use this script for plot purposes only."""

        # initialise DA variables --> DA.init(order, num_variables)
        DA.init(params.order, 6)

        # scaling
        Lsc = params.rE
        Vsc = np.sqrt(params.mu / params.rE)
        Tsc = Lsc / Vsc
        muSc = params.mu / Lsc / Lsc / Lsc * Tsc * Tsc

        scl2 = 1.0
        cont = params.cont
        scl = params.scl / Vsc

        # DA expansion around the initial state
        x0DA = array(
            [
                rr1[0] + DA(1),
                rr1[1] + DA(2),
                rr1[2] + DA(3),
                vv1[0] + DA(4),
                vv1[1] + DA(5),
                vv1[2] + DA(6),
            ]
        )

        # initialise the list with the states (trajectory)
        listState = []
        listState.append(x0DA.cons())

        # scaling
        x0DA[0:3] = x0DA[0:3] / Lsc
        x0DA[3:6] = x0DA[3:6] / Vsc

        # time scaling
        tof = tof / Tsc
        dt = 10.0 / Tsc  # step size for the propagation (s)

        vecttof = np.append(np.arange(dt, tof, tof / Npoints), tof)

        # initialise the STM
        STM = np.zeros((len(vecttof) + 1, 6, 6))
        for j in range(6):
            for k in range(6):
                if j == k:
                    STM[0, j, k] = 1.0
                else:
                    STM[0, j, k] = 0.0

        for indtf in range(len(vecttof)):

            # propagate
            xfDA = MultiRevolutionPerturbedLambertSolver.analyticJ2propHill(
                x0DA,
                vecttof[indtf],
                muSc,
                params.rE / Lsc,
                params.J2,
                cont + scl2 * DA(4),
            )

            # scale back
            xfDA[0:3] = xfDA[0:3] * Lsc
            xfDA[3:6] = xfDA[3:6] * Vsc

            # state transition matrix (STM) - derivate w.r.t. time
            for j in range(6):
                for k in range(6):
                    STM[indtf + 1, j, k] = DA.deriv(xfDA[j], k + 1).cons()

            # save the propagated trajectory
            listState.append(xfDA.cons())
            states = np.vstack(listState)

        return states, np.insert(vecttof, 0, 0.0) * Tsc, STM

    # class of OUTPUT for primer vector initial and final conditions
    class output_primer_vector_initial_final_conditions:
        def __init__(
            self,
            pp0: np.array,
            ppf: np.array,
            pp0dot: np.array,
            ppfdot: np.array,
            p0: float,
            pf: float,
            p0dot: float,
            pfdot,
        ):
            self.pp0 = pp0
            self.ppf = ppf
            self.pp0dot = pp0dot
            self.ppfdot = ppfdot

            self.p0 = p0
            self.pf = pf
            self.p0dot = p0dot
            self.pfdot = pfdot

    # class of OUTPUT for primer vector propagation
    class output_primer_vector_propagation:
        def __init__(
            self,
            pp: np.array,
            p: np.array,
            pd: np.array,
            states: np.array,
            vecttof: np.array,
            STM: np.array,
        ):
            self.pp = pp
            self.p = p
            self.pd = pd

            self.states = states
            self.vecttof = vecttof
            self.STM = STM

    # primer vector between t1 and t2 (manoeuvres dvv1 and dvv2 located at t1 and t2)
    @staticmethod
    def primerVectorInitialAndFinalConditions(
        dvv1: np.array, dvv2: np.array, STM12: np.array
    ):

        # primer vector and derivatives
        pp0 = dvv1 / np.linalg.norm(dvv1)  # initial primer vector
        ppf = dvv2 / np.linalg.norm(dvv2)  # final primer vector
        pp0dot = np.linalg.inv(STM12[0:3, 3:6]) @ (
            ppf - STM12[0:3, 0:3] @ pp0
        )  # initial primer vector derivative
        ppfdot = STM12 @ np.append(pp0, pp0dot)
        ppfdot = ppfdot[3:6]  # final primer vector derivative

        # magnitude of primer vector and derivative
        p0 = np.linalg.norm(pp0)  # initial primer vector magnitude
        pf = np.linalg.norm(ppf)  # final primer vector magnitude
        p0dot = (pp0dot @ pp0) / p0  # initial primer vector derivative
        pfdot = (ppfdot @ ppf) / pf  # final primer vector derivative

        # output primer vector initial and final conditions
        return MultiRevolutionPerturbedLambertSolver.output_primer_vector_initial_final_conditions(
            pp0, ppf, pp0dot, ppfdot, p0, pf, p0dot, pfdot
        )

    # obtain STM between t1 and t2 --> tof = t2 - t1
    @staticmethod
    def stateTransitionMatrix(
        rr1: np.array, vv1: np.array, tof: float, params: mrplp_J2_analytic_parameters
    ):

        # initialise DA variables --> DA.init(order, num_variables)
        DA.init(params.order, 6)

        # scaling
        Lsc = params.rE
        Vsc = np.sqrt(params.mu / params.rE)
        Tsc = Lsc / Vsc
        muSc = params.mu / Lsc / Lsc / Lsc * Tsc * Tsc
        scl2 = 1.0
        cont = params.cont

        # DA expansion around the initial state
        x0DA = array(
            [
                rr1[0] + DA(1),
                rr1[1] + DA(2),
                rr1[2] + DA(3),
                vv1[0] + DA(4),
                vv1[1] + DA(5),
                vv1[2] + DA(6),
            ]
        )

        # scaling
        x0DA[0:3] = x0DA[0:3] / Lsc
        x0DA[3:6] = x0DA[3:6] / Vsc

        # propagate with analytic J2 dynamics
        xfDA = MultiRevolutionPerturbedLambertSolver.analyticJ2propHill(
            x0DA, tof / Tsc, muSc, params.rE / Lsc, params.J2, cont + scl2 * DA(4)
        )

        # scale back
        xfDA[0:3] = xfDA[0:3] * Lsc
        xfDA[3:6] = xfDA[3:6] * Vsc

        # state transition matrix (STM)
        STM = np.zeros((6, 6))  # initialise the STM
        for j in range(6):
            for k in range(6):
                STM[j, k] = DA.deriv(xfDA[j], k + 1).cons()  # derivate

        # output
        return STM

    # propagate the primer vector
    @staticmethod
    def propagatePrimerVector(
        rr1: np.array,
        vv1: np.array,
        tof: float,
        dvv1: np.array,
        dvv2: np.array,
        Npoints: float,
        params: mrplp_J2_analytic_parameters,
    ):

        # propagate and obtain the STM
        states, vecttof, STM = (
            MultiRevolutionPerturbedLambertSolver.propagatePerturbedKeplerSTM(
                rr1, vv1, tof, Npoints, params
            )
        )

        # initial and final conditions for the primer vector
        primerVectorInitialFinalCond = (
            MultiRevolutionPerturbedLambertSolver.primerVectorInitialAndFinalConditions(
                dvv1, dvv2, STM[-1, :, :]
            )
        )

        pp0 = np.append(
            primerVectorInitialFinalCond.pp0, primerVectorInitialFinalCond.pp0dot
        )

        listpp = []
        listpp.append(pp0)

        listp = []
        listp.append(primerVectorInitialFinalCond.p0)

        listpd = []
        listpd.append(primerVectorInitialFinalCond.p0dot)

        for ind in range(1, states.shape[0]):

            ppcurr = STM[ind, :, :] @ pp0
            pcurr = np.linalg.norm(ppcurr[0:3])
            pdcurr = (ppcurr[3:6] @ ppcurr[0:3]) / pcurr

            listpp.append(ppcurr)
            pp = np.vstack(listpp)

            listp.append(pcurr)
            p = np.vstack(listp)

            listpd.append(pdcurr)
            pd = np.vstack(listpd)

        return MultiRevolutionPerturbedLambertSolver.output_primer_vector_propagation(
            pp, p.flatten(), pd.flatten(), states, vecttof, STM
        )
