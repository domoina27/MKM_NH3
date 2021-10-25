import sympy as sym
import numpy as np
from decimal import Decimal
from sympy import diff, sin, cos, exp , Sum, Product, symbols, Indexed, lambdify
from scipy.misc import derivative as der

#sym.init_printing()

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
    return kb/h*T* np.exp(Decimal(-Eact / (R * T)) )

def calc_k_arr2(T, Eact):
    """
    Calculate reaction rate constant for a surface reaction
    
    T       - Temperature in K
    nu      - Pre-exponential factor in s^-1
    Eact    - Activation energy in J/mol
    """
    kb = 1.38064852E-23 # boltzmann constant
    R = 8.3144598 # gas constant
    h = 6.62607004e-34  # planck constant
    return (kb/h*T)* np.exp((-Eact) / (R * T)) 

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
    I= 1856
    return kb * T / h**3 * A * (2 * np.pi * m * kb) /(sigma * theta_rot) * np.exp(-Edes / (R*T))
    # return (kb*T/h)**(2.5)* A * (2 * np.pi * m * kb * T)/(sigma * h**2)* np.sqrt(np.pi/1856) * np.exp(-Edes / (R*T))
    #return A / np.sqrt(2 * np.pi * m * kb * T)* np.exp(-Edes / (R*T))

# def calc_kdesN2(T, Edes):
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
    I= 1856
    A = 1e20
    m = 14 * 1.66054e-27
    sigma = 2
    theta_rot = 2.88
    return kb * T**3 / h**3 * A * (2 * np.pi * m * kb) /(sigma * theta_rot) * np.exp(-Edes / (R*T))
    # return (kb*T/h)**(2.5)* A * (2 * np.pi * m * kb * T)/(sigma * h**2)* np.sqrt(np.pi/I) * np.exp(-Edes / (R*T))

# def calc_kdesH2(T, A, m, sigma, theta_rot, Edes):
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
    
    
    m = 2 * 1.66054e-27
    return kb * T / h**3 * A * (2 * np.pi * m * kb) /(sigma * theta_rot) * np.exp(-Edes / (R*T))


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
    I= 1856
    #K = kb/ h**3 * A * (2 * np.pi * m * kb) / (sigma * theta_rot)
    # return (2*A*Edes*T*kb**2*m*np.pi*np.exp(-Edes/(R*T))/(R*h**3*sigma*theta_rot)) + (6*A*T**2*kb**2*m*np.pi*np.exp(-Edes/(R*T))/(h**3*sigma*theta_rot))
    return 2*A*Edes*T*kb*m*np.pi*np.sqrt(np.pi/I)*(T*kb/h)**2.5*np.exp(-Edes/(R*T))/(R*T*h**2*sigma) + 7*A*kb*m*np.pi*np.sqrt(np.pi/I)*(T*kb/h)**2.5*np.exp(-Edes/(R*T))/(h**2*sigma)
    # return A*kb *m* np.exp(-Edes / (R*T)) *(2*Edes - R*T)/(2*np.sqrt(2 * np.pi)*R*T*(kb*m*T)**1.5)

# Vibrational partition functtion and derivative of the partition function

def Qvib(T, frequency):  
    R = 8.3144598       # kg⋅m2⋅s−2⋅K−1⋅mol−1   Ideal gas constant
    kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s                 speed of light
    Qvib= np.zeros(len(frequency))
    for i in range(len(frequency)):
        def q_vib(frequency):  
            den = (kb)/(h*c)*0.01 #!!!!convert m to cm #h*c/(kb*T)
            return 1/(1-np.exp(-frequency/(den*T)))
        Qvib[i] = np.array(q_vib(frequency[i]))
        # print((Qvib))
    return (np.prod(Qvib))

def Qtran(T,m):
    R = 8.3144598       # kg⋅m2⋅s−2⋅K−1⋅mol−1   Ideal gas constant
    kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s                 speed of light
    V= 15*7.46720*7.46720*1e-40
    q_translation = (np.sqrt(2*np.pi*m*kb*T)/h)**(3) *V#*1000*1e-6
    return (q_translation)



def Qrot(T,sigma,B):
    kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
    h = 6.62607004e-34  # m2 kg / s             planck constant
    q_rotation = 1/sigma *kb*T/(h*B)
    return (q_rotation)


def ZPE(frequency):
    # zpe = np.zeros(len(frequency))
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s 
    NA = 6.0221409e23   #Avogadro's number
    # for i in range(len(frequency)):
    #     zpe = 1*frequency[i]  #convert cm-1 to m-1
    return np.sum(frequency)*100*299792458*6.62607004e-34*6.0221409e23/2

def ZPE_eV(frequency):
    # zpe = np.zeros(len(frequency))
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s 
    NA = 6.0221409e23   #Avogadro's number
    # for i in range(len(frequency)):
    #     zpe = 1*frequency[i]  #convert cm-1 to m-1
    return np.sum(frequency)*100*299792458*6.62607004e-34*6.0221409e23/2 * 0.00001036427230133138

def U(frequency,T):
    R = 8.3144598       # kg⋅m2⋅s−2⋅K−1⋅mol−1   Ideal gas constant
    kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s                 speed of light
    NA = 6.0221409e23   #Avogadro's number
    Uvib= np.zeros(len(frequency))
    for i in range(len(frequency)):
        def U_vib(frequency):  
            return h*frequency*100*c/kb/(np.exp(h*frequency*100*c/(kb*T))-1)  #convert cm-1 to m-1
        Uvib[i] = np.array(U_vib(frequency[i]))
        # print((Qvib))
    return R*np.sum(Uvib)   

def U_eV(frequency,T):
    R = 8.3144598       # kg⋅m2⋅s−2⋅K−1⋅mol−1   Ideal gas constant
    kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s                 speed of light
    NA = 6.0221409e23   #Avogadro's number
    Uvib= np.zeros(len(frequency))
    for i in range(len(frequency)):
        def U_vib(frequency):  
            return h*frequency*100*c/kb/(np.exp(h*frequency*100*c/(kb*T))-1)  #convert cm-1 to m-1
        Uvib[i] = np.array(U_vib(frequency[i]))
        # print((Qvib))
    return R*np.sum(Uvib)  * 0.00001036427230133138 

def S(frequency,T):
    R = 8.3144598       # kg⋅m2⋅s−2⋅K−1⋅mol−1   Ideal gas constant
    kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s                 speed of light
    NA = 6.0221409e23   #Avogadro's number
    Svib= np.zeros(len(frequency))
    for i in range(len(frequency)):
        def S_vib(frequency):  
            return h*frequency*100*c/kb/(np.exp(h*frequency*100*c/(kb*T))-1) - np.log(1-np.exp(-h*frequency*100*c/(kb*T)))   #convert cm-1 to m-1; 1cm-1 = 100 m-1
        Svib[i] = np.array(S_vib(frequency[i]))
        # print((Qvib))
    return R*np.sum(Svib)

def S_eV(frequency,T):
    R = 8.3144598       # kg⋅m2⋅s−2⋅K−1⋅mol−1   Ideal gas constant
    kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s                 speed of light
    NA = 6.0221409e23   #Avogadro's number
    Svib= np.zeros(len(frequency))
    for i in range(len(frequency)):
        def S_vib(frequency):  
            return h*frequency*100*c/kb/(np.exp(h*frequency*100*c/(kb*T))-1) - np.log(1-np.exp(-h*frequency*100*c/(kb*T)))   #convert cm-1 to m-1; 1cm-1 = 100 m-1
        Svib[i] = np.array(S_vib(frequency[i]))
        # print((Qvib))
    return R*np.sum(Svib)* 0.00001036427230133138 




# def S2(frequency,T):
#     R = 8.3144598       # kg⋅m2⋅s−2⋅K−1⋅mol−1   Ideal gas constant
#     kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
#     h = 6.62607004e-34  # m2 kg / s             planck constant
#     c = 299792458       # m / s                 speed of light
#     NA = 6.0221409e23   #Avogadro's number
#     Svib= np.zeros(len(frequency))
#     Exp = np.zeros(len(frequency))
#     for i in range(len(frequency)):
#         def S_vib(frequency): 
#             den = (kb)/(h*c)*0.01 #!!!!convert m to cm #h*c/(kb*T)            
#             return np.log(1/(1-np.exp(-frequency/(den*T))))   #convert cm-1 to m-1; 1cm-1 = 100 m-1
#         Svib[i] = np.array(S_vib(frequency[i]))
#         def Q_vib(frequency): 
#             den = (kb)/(h*c)*0.01 #!!!!convert m to cm #h*c/(kb*T)            
#             return np.prod(1/(1-np.exp(-frequency/(den*T))))   #convert cm-1 to m-1; 1cm-1 = 100 m-1
#         # def diff(frequency):
#         #     return Exp[i]= 3#*np.exp(-vTs[i]/(beta*T))*np.prod(Num)/(beta*T*h*np.prod(Den)*(1-np.exp(-vTs[i]/(T*beta))))
#         # print((Qvib))
#     return R*np.sum(Svib) + R* Q_vib *#* 0.00001036427230133138 
# G= H-TS; G correction = H correction -T*S 





def dk(T,vTs,vIs,Eact):
    R = 8.3144598       # kg⋅m2⋅s−2⋅K−1⋅mol−1   Ideal gas constant
    kb = 1.38064852e-23 #J⋅K−1                  Boltzmann Constant
    h = 6.62607004e-34  # m2 kg / s             planck constant
    c = 299792458       # m / s                 speed of light
    beta = kb/(h*c)*0.01
    

    Num = np.zeros(len(vIs))
    Den = np.zeros(len(vTs))

    #expression 1
    for i in range(len(vIs)):
        Num[i] = (1-np.exp(-vIs[i]/(T*beta)))
        Den[i] = (1-np.exp(-vTs[i]/(T*beta)))

    # print(Num)
    # print(Den)

    Exp1 = kb*Eact*np.exp(-Eact/(R*T))*np.prod(Num)/(R*T*h*np.prod(Den))
    # print(Exp1)

    #expression 2
    Exp2 = kb*np.exp(-Eact/(R*T))*np.prod(Num)/(h*np.prod(Den))
    # print(Exp2)

    #expression 3
    Exp3 = np.zeros(len(vIs))
    for i in range(len(vTs)):
        Exp3[i]= kb*vTs[i]*np.exp(-Eact/(R*T))*np.exp(-vTs[i]/(beta*T))*np.prod(Num)/(beta*T*h*np.prod(Den)*(1-np.exp(-vTs[i]/(T*beta))))
    # print(np.sum(Exp3))

    #expression 4
    Exp4 = np.zeros(len(vIs))
    for i in range(len(vIs)):
        Exp4[i]= kb*vIs[i]*np.prod(Num)*np.exp(-Eact/(R*T))*np.exp(-vIs[i]/(beta*T))/(beta*T*h*np.prod(Den)*(1-np.exp(-vIs[i]/(T*beta))))
    # print(np.sum(Exp4))

    return Exp1 + Exp2 + np.sum(Exp3) -np.sum(Exp4)
