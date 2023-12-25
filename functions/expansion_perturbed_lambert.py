from typing import Callable, Type

import numpy as np
from daceypy import DA, array
from scipy.optimize import fsolve
from functions.MRPLP_J2_analytic import MultiRevolutionPerturbedLambertSolver

# this class contains the expansion of the perturbed Lambert problem solution
class ExpansionPerturbedLambert:
    def __init__(self):
        pass

    # analytic J2 propagation --> in this case also the time of flight (tof) is a DA variable
    @staticmethod
    def analyticJ2propHill( x0: array, tof: array, mu: float, rE: float, J2: float, cont: float ):
        
        # cartesian to keplerian elements
        kep0 = MultiRevolutionPerturbedLambertSolver.cart2kep(x0, mu)

        # keplerian elements to Hill
        hill0 = MultiRevolutionPerturbedLambertSolver.kep2Hill(kep0, mu)

        # osculating to mean
        hill0Mean = MultiRevolutionPerturbedLambertSolver.osculating2meanHill(hill0, mu, J2, rE, cont)

        # hill to kep
        kep0Mean = MultiRevolutionPerturbedLambertSolver.hill2kep(hill0Mean, mu)
        kep0Mean[5] = MultiRevolutionPerturbedLambertSolver.true2meanAnomaly(kep0Mean[5], kep0Mean[1])

        del0Mean = MultiRevolutionPerturbedLambertSolver.kep2delaunay(kep0Mean, mu)
        delfMean = MultiRevolutionPerturbedLambertSolver.averagedJ2rhs(del0Mean, mu, J2, rE, cont)
        delfMean = delfMean*tof+del0Mean

        kepfMean = MultiRevolutionPerturbedLambertSolver.delaunay2kep(delfMean, mu)
        kepfMean[5] = MultiRevolutionPerturbedLambertSolver.mean2trueAnomaly(kepfMean[5], kepfMean[1])
        hillfMean = MultiRevolutionPerturbedLambertSolver.kep2hill(kepfMean, mu)
        hillf = MultiRevolutionPerturbedLambertSolver.mean2osculatingHill(hillfMean, mu, J2, rE, cont)

        xxf = MultiRevolutionPerturbedLambertSolver.hill2cart(hillf, mu)

        return xxf

    # expansion of perturbed Lambert problem
    @staticmethod
    def expansionOfPerturbedLambert(rr1: np.array, vv1: np.array, tof: float, params):

        """ Perform the expansion of the pertubed dynamics around (rr1, rr2, tof). In this way, one can evaluate the effect of perturbations on (rr1, rr2, tof) on the initial and final velocity (vv1, vv2). The state (rr2, vv2) is obtained propagating forward the state (rr1, vv1) for the time of flight (tof).  """

        # initialise DA variables --> DA.init(order, num_variables)
        DA.init( params.order, 7 ) # (rr1, vv1, tof)

        # scaling
        Lsc = params.rE
        Vsc = np.sqrt(params.mu/params.rE)
        Tsc = Lsc/Vsc
        muSc = params.mu/Lsc/Lsc/Lsc*Tsc*Tsc
        sclT = 100.0

        # DA expansion around the initial state and scaling
        x0DA = array( [rr1[0]+DA(1), rr1[1]+DA(2), rr1[2]+DA(3), vv1[0]+DA(4), vv1[1]+DA(5), vv1[2]+DA(6)] )
        x0DA[0:3] = x0DA[0:3]/Lsc
        x0DA[3:6] = x0DA[3:6]/Vsc
        tf = (tof + sclT*DA(7))/Tsc

        # propagate
        xfDA = ExpansionPerturbedLambert.analyticJ2propHill( x0DA, tf, muSc, params.rE/Lsc, params.J2, 1.0 )

        # direct map
        mapD = array.zeros(7)
        mapD[0] = DA(1)
        mapD[1] = DA(2)
        mapD[2] = DA(3)
        mapD[3] = ( xfDA[0] - xfDA[0].cons() )*Lsc
        mapD[4] = ( xfDA[1] - xfDA[1].cons() )*Lsc
        mapD[5] = ( xfDA[2] - xfDA[2].cons() )*Lsc
        mapD[6] = DA(7)

        # invert the map and evaluate intial and final state in the inverse map
        mapI = mapD.invert()

        # evaluate the intial and final state in the inverse map
        x0DA = x0DA.eval(mapI)
        xfDA = xfDA.eval(mapI)

        # scale back
        x0DA[0:3] = x0DA[0:3]*Lsc
        x0DA[3:6] = x0DA[3:6]*Vsc
        xfDA[0:3] = xfDA[0:3]*Lsc
        xfDA[3:6] = xfDA[3:6]*Vsc

        st = 1

        return x0DA, xfDA