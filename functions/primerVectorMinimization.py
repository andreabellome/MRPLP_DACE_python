from typing import Callable, Type

from scipy.optimize import minimize, show_options
from functools import partial


from daceypy import DA, array
from functions.MRPLP_J2_analytic import MultiRevolutionPerturbedLambertSolver as MRPLPsolver
from functions.expansion_perturbed_lambert import ExpansionPerturbedLambert
import numpy as np

class minimizationPrimerVector():
    def __init__(self):
        pass

    # define the cost function --> only 1 impulse, then extend
    @staticmethod
    def objectiveFunction( dXX = np.array, rr1 = np.array, rr2 = np.array,
                          vv1 = np.array, vv2 = np.array, 
                          rrM = np.array, vvM = np.array,
                           tof = float, tofM = float, 
                           mu = float, rE = float, J2 = float, 
                           order = int ):

        """ The variables (dx,dt) is the variation on the positions and times at maximum primer vector (excluding initial and final points). These are the optimization variables. """

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

        print(f"New cost             : {dvtotNew} km/s")

        return dvtotNew
    
    # define the gradients of the cost function --> only one impulse, then extend
    def gradients(dXX = np.array, rr1 = np.array, rr2 = np.array,
                          vv1 = np.array, vv2 = np.array, 
                          rrM = np.array, vvM = np.array,
                           tof = float, tofM = float, 
                           mu = float, rE = float, J2 = float, 
                           order = int):
        
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

        STM1M = MRPLPsolver.stateTransitionMatrix( rr1, vv1_1M, tofM+dt, parameters )
        STMM2 = MRPLPsolver.stateTransitionMatrix( rrM+dx, vv1_M2, tof-(tofM+dt), parameters )

        dvv1 = vv1_1M - vv1
        dvv2 = vv2 - vv2_M2
        dvvM = vv1_M2 - vv2_1M

        out1M = MRPLPsolver.primerVectorInitialAndFinalConditions( dvv1, dvvM, STM1M )
        out2M = MRPLPsolver.primerVectorInitialAndFinalConditions( dvvM, dvv2, STMM2 )

        dJdrm = out2M.pp0dot - out1M.ppfdot
        dJdtm = np.dot(out1M.ppfdot, vv2_1M) - np.dot( out2M.pp0dot, vv1_M2 )

        grad = np.array( [dJdrm[0], dJdrm[1], dJdrm[2], dJdtm] )

        print(f"Max. of the gradient : {np.amax(grad)}")

        return grad

    # generate initial guess on the position perturbation at maximum of primer vector
    @staticmethod
    def generateInitialGuess(beta = float, STM1M = np.array, STMM2 = np.array,
                            rrm = np.array, ppm = np.array):

        # generate initial guess
        Mmf = STMM2[0:3,0:3]
        Nmf = STMM2[0:3,3:6]
        T0m = STM1M[3:6,3:6]
        N0m = STM1M[0:3,3:6]
        A = -( np.transpose(Mmf) @ np.linalg.inv(np.transpose(Nmf)) + T0m @ np.linalg.inv(N0m) )

        eps = beta*np.linalg.norm( rrm )/( abs( np.linalg.inv(A) @ ppm ) )
        drrm = eps * np.linalg.inv(A) @ ppm

        return drrm

    @staticmethod
    def minimizationWithPrimerVector(rr1 = np.array, rr2 = np.array,
                          vv1 = np.array, vv2 = np.array, 
                          rrM = np.array, vvM = np.array,
                          beta = float, STM1M = np.array, STMM2 = np.array, ppm = np.array,
                           tof = float, tofM = float, 
                           params = MRPLPsolver.mrplp_J2_analytic_parameters ):

        # generate initial guess
        drrm = minimizationPrimerVector.generateInitialGuess( beta, STM1M, STMM2, rrM, ppm )
        initial_guess = np.array([drrm[0], drrm[1], drrm[2], 0.0])
        initial_guess = np.array([ 0.0, 0.0, 0.0, 0.0])

        args = (rr1 , rr2 ,
                vv1 , vv2 , 
                rrM , vvM ,
                tof , tofM , 
                params.mu , params.rE , 
                params.J2 , params.order)

        result = minimize(minimizationPrimerVector.objectiveFunction, initial_guess, args, 
                          jac=minimizationPrimerVector.gradients,
                          method='L-BFGS-B', 
                          options={'ftol': 1e-11, 'gtol': 1e-11, 'eps': 1e-11})
        

        return result


    