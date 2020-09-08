import numpy as np

def calc_k_arr(T, nu ,Eact):
    """
    Calculate reaction rate constant for a surface reaction
    
    T       - Temperature in K
    nu      - Pre-exponential factor in s^-1
    Eact    - Activation energy in J/mol
    """
    kb = 1.38064852E-23 # boltzmann constant
    R = 8.3144598 # gas constant
    h = 6.62607004e-34  # planck constant
    #return nu* np.exp(-Eact / (R * T))# *kb/h*T 
    return kb/h*T* np.exp(-Eact / (R * T)) 

def calc_kads(T, P, A, m):
    """
    Reaction rate constant for adsorption
    
    T           - Temperature in K
    P           - Pressure in Pa
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    """
    kb = 1.38064852E-23 # boltzmann constant
    return P*A / np.sqrt(2 * np.pi * m * kb * T)

def calc_kdes(T, A, m, sigma, theta_rot, Edes):
    """
    Reaction rate constant for desorption
    
    T           - Temperature in K
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    sigma       - Symmetry number
    theta_rot   - Rotational temperature in K
    Edes        - Desorption energy in J/mol
    """
    kb = 1.38064852e-23 # boltzmann constant
    h = 6.62607004e-34  # planck constant
    R = 8.3144598       # gas constant
    #return kb * T**3 / h**3 * A * (2 * np.pi * m * kb) /(sigma * theta_rot) * np.exp(-Edes / (R*T))
    #return (kb*T/h)**(2.5)* A * (2 * np.pi * m * kb * T)/(sigma * h**2)* np.sqrt(np.pi/1856) * np.exp(-Edes / (R*T))
    return A / np.sqrt(2 * np.pi * m * kb * T)* np.exp(-Edes / (R*T))

# Derivative of Rate constants in relation to Temperature
#--------------------------------------------------------
def dksur_dT(T, nu, Eact):
    """
    Calculate reaction rate constant for a surface reaction
    
    T       - Temperature in K
    nu      - Pre-exponential factor in s^-1
    Eact    - Activation energy in J/mol
    """
    R = 8.3144598 # gas constant
    kb = 1.38064852E-23 # boltzmann constant
    R = 8.3144598 # gas constant
    h = 6.62607004e-34  # planck constant
    #return nu * np.exp(-Eact / (R * T))   *(Eact+R*T)/(R*T) 
    return kb/h * np.exp(-Eact / (R * T))   *(Eact+R*T)/(R*T) 

def dkads_dT(T, P, A, m):
    """
    Reaction rate constant for adsorption
    
    T           - Temperature in K
    P           - Pressure in Pa
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    """
    kb = 1.38064852E-23 # boltzmann constant
    return P*A / np.sqrt(2 * np.pi * m * kb * T)

def dkdes_dT(T, A, m, sigma, theta_rot, Edes):
    """
    Reaction rate constant for desorption
    
    T           - Temperature in K
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    sigma       - Symmetry number
    theta_rot   - Rotational temperature in K
    Edes        - Desorption energy in J/mol
    """
    kb = 1.38064852e-23 # boltzmann constant
    h = 6.62607004e-34  # planck constant
    R = 8.3144598       # gas constant    
    
    K = kb/ h**3 * A * (2 * np.pi * m * kb) / (sigma * theta_rot)
    #return K/R*T* np.exp(-Edes / (R*T))* (Edes+2*R*T)
    return A*kb *m* np.exp(-Edes / (R*T)) *(2*Edes - R*T)/(2*np.sqrt(2 * np.pi)*R*T*(kb*m*T)**1.5)