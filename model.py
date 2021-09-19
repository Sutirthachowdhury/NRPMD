import numpy as np
import random
import math

class parameters():
    NSteps = 3000       # time step of the propagation
    NTraj = 1           # total number of trajectories
    dtN = 0.01          # nuclear time step
    EStep = 20          # mapping integration length
    dtE = dtN/EStep     # mapping timestep
    beta = 1.0          # inverse temperature of the system
    M = 1               # mass of the particle
    nstate = 2          # number of electronic states
    nb = 15             # this is number of beads, bead number has to be odd
    lb_n = -(nb-1)/2    #lowest bead index, useful for normal mode transformation
    ub_n = (nb-1)/2     # highest bead index, useful for normal mode transformation
    ndof = 1            # number of nuclear degrees of freedom
    initState = 0       # initially occupied electronic state
    fs_to_au = 41.341   # a.u./fs (unit conversion)
        
    #-------------- MODEL SPECIFIC--------------------------------------------------
    
    # -- this is 1D spin-boson model------
    ε = -0.5            # energy offset between two states
    Δ  = 1.0            # off-diagonal coupling elements/ non-adiabatic coupling
    kappa = 0.1         # system-bath interaction
#====================================================================================

def initModelParams():

    """
    model related parameters
    frequency of the osillator and shifts
    """
    
    ndof = parameters.ndof
    k = parameters.kappa
    ω = np.zeros((ndof))
    c = np.zeros((ndof))

    for i in range(ndof):
        ω[i] = 1.0
        c[i] = np.sqrt(2.0*ω[i])*k 
    
    return ω,c 
#======================state dep part of the Hamiltonian==================
def Hel(R):

    ε = parameters.ε
    Δ = parameters.Δ
    ω,c = initModelParams()

    Hel = np.zeros((2,2))


    # Harmonic part is state-independent and common
    Hel[0,0] = ε
    Hel[0,1] = Δ/2.0
    Hel[1,0] = Hel[0,1]
    Hel[1,1] = ε
        
    Hel[0,0]  +=  np.sum( c * R )  
    Hel[1,1]  -=  np.sum( c * R )

    return Hel
#===========================================================================

#========== state independent part of the Hamiltonian==========
def Hel0(R):
    ω,c = initModelParams()
    return  np.sum(0.5 * ω**2 * R**2.0)
#==============================================================

#============== gradient of state indep part======================
def dHel0(R):

    """
    dv_0/dR
    """
    ω,c = initModelParams()

    return ω**2 * R
#=============================================================

#=========== gradient of state dep part======================
def dHel(R):

    """
    dv_ij/dR
    """

    dHij = np.zeros((2,2,len(R)))
    ω,c = initModelParams()
    dHij[0,0,:] = c
    dHij[1,1,:] = -c
    
    return dHij





