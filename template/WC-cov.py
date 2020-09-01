"""
Rejoice always. 1 Thessalonian 5:16
@author: Domoina Holiharimanana
"""
import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import solve_ivp, ode
from mpl_toolkits.mplot3d import Axes3D

ma = 16*1.66054e-27 #molecular weight
y0 = [0,0,1] #initial condition
#P = 0.1e-5 #pressure
T = 400
# P = 1
t_end = 1000  #how long the reaction will last

# =============================================================================
# Rate constant
# =============================================================================
def kads(T, Eads):
    """
    Reaction rate constant for adsorption
    
    T           - Temperature in K
    P           - Pressure in Pa
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    """
    kb = 1.38064852E-23 # boltzmann constant
    R = 8.3144598       # gas constant
    #return P*A / np.sqrt(2 * np.pi * m * kb * T) *np.exp(-Eads / (R*T))
    return 1e4 *np.exp(-Eads / (R*T))

def kdes(T, Edes):
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
    return 1e13 * np.exp(-(Edes) / (R*T))

# =============================================================================
# Variation of Temperature    
# =============================================================================
P_range = np.linspace(0,1e6,1000)
T_range = np.linspace(500,1200,700)
Th_CH3=[]
Th_H=[]
Th=[]
th = 0




for P in P_range:
    # for T in T_range:
    # =============================================================================
    # Coverage in funtion of temperature    
    # =============================================================================
    kf= kads(T,  83.942e3)
    kb= kdes(T, 229.64e3)
    K = kf/kb
    print(K, kf, kb)
    th_ch3 =  np.sqrt(K*P)/(1 + np.sqrt(K*P))
    th = 1-th_ch3  
    # Th_H.append(th_h)
    Th_CH3.append(th_ch3)
    Th.append(th)
# =============================================================================
# Plot coverage in function of temperature
# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(P, T, Th_CH3, c='r')
# ax.set_xlabel('pressure')
# ax.set_ylabel('temperature')
# ax.set_zlabel('Th_A')

# legend = ["CH3*",'H*',"*"]
# coverage = [Th_CH3,Th_H,Th]

plt.plot(P_range, Th_CH3, label="CH3*")
# plt.plot(P_range, Th, label="*")

# for i in range(3):
#     # plt.semilogx(T_range, coverage[i], label=legend[i])
#     plt.plot(T_range, coverage[i], label=legend[i])
plt.ylabel('Coverage')
plt.xlabel('P (Pa)')
plt.title('At Presure ='+ str(P) +'Pa')
# plt.title('At Temperature ='+ str(T) +'K')
plt.legend()
plt.show()
