#!/usr/bin/env python
"""
THIS IS A PRACTICE FOR MICRO-KINETIC MODELING OF NH3 DISSOCIATION AND N2 AND H2 FROMATION:
    REACTION: 
        NH3 (g) + * = NH3*
            NH3 dissociation is done in 3 elementarty steps
        NH3* + * = NH2* + H*
        NH2* + * = NH* + H*
        NH* + * = N* + H*
            N2 formation is done in 3 elementarty steps
        NH* + N* = *NH-N*
        2N* = *N-N*
        *NH-N* = *N-N* + H*
        *N-N* = N2* + *
        N2* = N2 (g) + *
            H2 formation is done in one step
        2H* = H2 (g) + 2*
           PLOT COVERAGE VS TIME
"""
import sys
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp , ode
from scipy.misc import derivative as der

from rate_constant_func2 import calc_kads as k_ads
from rate_constant_func2 import calc_kdes as k_des
from rate_constant_func2 import calc_k_arr2 as k_surf
from rate_constant_func2 import Qvib , Qtran, Qrot, ZPE

from rate_constant_func2 import dksur_dT
from rate_constant_func2 import dkdes_dT
# from rate_constant_func2 import dk
import pandas as pd
import sympy as sym
from sympy import diff, sin, cos, exp , Sum, Product, symbols, Indexed, lambdify


frequency = pd.read_csv('Ni_batch.csv', na_filter=True)
#------------------------------------------------------------------------------
# CoNi2 (111)
#Hydrogen coverage
alpa_H= -0.03*96.48530749926e3; intercept_H = 0#.6622*2*96.48530749926e3 #convert eV to KJ/mol

#Nitrogen coverage
alpha_N= -2.13*96.48530749926e3; intercept_N = 0#1.1521*96.48530749926e3*2 #convert eV to KJ/mol

#activation barriers
Eng = (72512.78, 125540.90, 113172.2678, 71470.35, 96038.39527, 136039.66, 100315.8504, 167340.88, 139748.1857, 169334.25 , 38673.20 , 15584.5, 67225.0236, 50192.97   , 113877.37 , 38836.92, 105539.25)          
#        Ef1   ,   Ef2    ,    Eb2    ,     Ef3  ,      Eb3   ,    Ef4    ,   Eb4     ,    Ef5   ,    Eb5     ,     Ef_5  ,   Eb_5   ,   Ef6  ,     Eb6   ,   Ef_6     ,   Eb_b   ,  Ef7   ,     Ef8  
# ============================================================================= 
# # 0/ Constants
# =============================================================================
# step 1 - NH3 + * <=> NH3*        Ammonia adsorption
frq_IS1 = frequency.IS1.dropna(how='any')
# frq_TS1 = frequency.TS1.dropna(how='any')
frq_FS1 = frequency.IS2.dropna(how='any')

# step 2 - NH3* + * <=> NH2* + H*   Dehydrogenation
frq_IS2 = frequency.IS2.dropna(how='any')
frq_TS2 = frequency.TS2.dropna(how='any')
frq_FS2 = frequency.FS2.dropna(how='any')

#  step 3 -NH2* + * <=> NH* + H*    Dehydrogenation
frq_IS3 = frequency.IS3.dropna(how='any') 
frq_TS3 = frequency.TS3.dropna(how='any')
frq_FS3 = frequency.FS3.dropna(how='any')

#  step 4 - NH* + * <=> NH* + H*    Dehydrogenation
frq_IS4 = frequency.IS4.dropna(how='any') 
frq_TS4 = frequency.TS4.dropna(how='any')
frq_FS4 = frequency.FS4.dropna(how='any')

#  step 5 - N* + N* <=> *N-N*       N-N recombination
frq_IS5 = frequency.IS5.dropna(how='any') 
frq_TS5 = frequency.TS5.dropna(how='any')
frq_FS5 = frequency.FS5.dropna(how='any')


#  step 5' - NH* + N* <=> *NH-N*    NH-N recombination
frq_IS_5 = frequency.IS_5.dropna(how='any') 
frq_TS_5 = frequency.TS_5.dropna(how='any')
frq_FS_5 = frequency.FS_5.dropna(how='any')

#  step 6 - *N-N* <=> N2*  +*       N-N rotation
frq_IS6 = frequency.FS5.dropna(how='any')
frq_TS6 = frequency.TS6.dropna(how='any')
frq_FS6 = frequency.FS6.dropna(how='any')

#  step 6' - *NH-N* <=> N2*  +H*       N-N rotation
frq_IS_6 = frequency.FS_5.dropna(how='any')
frq_TS_6 = frequency.TS_6.dropna(how='any')
frq_FS_6 = frequency.FS_6.dropna(how='any')

#  step 7 - N2* <=> N2 (gas)  +*       N2 desorbtion
frq_IS7 = frequency.FS6.dropna(how='any')
frq_FS7 = frequency.FS7.dropna(how='any')

#  step 8 - H* + H* <=> H2 (gas) + 2*       N-N recombination
frq_IS8 = frequency.IS8.dropna(how='any')
frq_FS8 = frequency.FS8.dropna(how='any')


#------------------------------------------------------------------------------

Ef1  = Eng[0] +(ZPE(frq_FS1)-ZPE(frq_IS1))
# Eb1  = Eng[1] +(ZPE(frq_IS1)-ZPE(frq_FS1))

Ef2 = Eng[1]+(ZPE(frq_TS2)-ZPE(frq_IS2))
Eb2 = Eng[2]+(ZPE(frq_TS2)-ZPE(frq_FS2))

Ef3 = Eng[3]+(ZPE(frq_TS3)-ZPE(frq_IS3))
Eb3 = Eng[4]+(ZPE(frq_TS3)-ZPE(frq_FS3))

Ef4 = Eng[5]+(ZPE(frq_TS4)-ZPE(frq_IS4))
Eb4 = Eng[6]+(ZPE(frq_TS4)-ZPE(frq_FS4)) 
 
Ef5 = Eng[7]+(ZPE(frq_TS5)-ZPE(frq_IS5)) 
Eb5 = Eng[8]+(ZPE(frq_TS5)-ZPE(frq_FS5))  

Ef_5 = Eng[9]+(ZPE(frq_TS_5)-ZPE(frq_IS_5)) 
Eb_5 = Eng[10]+(ZPE(frq_TS_5)-ZPE(frq_FS_5)) 
 
Ef6 = Eng[11]+(ZPE(frq_TS6)-ZPE(frq_IS6))    
Eb6 = Eng[12]+(ZPE(frq_TS6)-ZPE(frq_FS6))

Ef_6 = Eng[13]+(ZPE(frq_TS_6)-ZPE(frq_IS_6))
Eb_6 = Eng[14]+(ZPE(frq_TS_6)-ZPE(frq_FS_6)) 
    
Ef7 = Eng[15]+(ZPE(frq_FS7)-ZPE(frq_IS7)) 
# Eb7 = Eng[17]+(ZPE(frq_IS7)-ZPE(frq_FS7))   

Ef8 = Eng[16]+(ZPE(frq_FS8)-ZPE(frq_FS8))
# Eb8 = Eng[19]+(ZPE(frq_IS8)-ZPE(frq_FS8))

#arguments =    (Ef1    ,   Ef2    ,    Eb2   ,     Ef3,    Eb3   ,    Ef4    ,   Eb4   ,    Ef5   ,    Eb5   ,   Ef_5,   Eb_5  ,   Ef6  ,   Eb6  ,  Ef7  , Eb7  ,    Ef8  , Eb8 )
arguments =    (Ef1    ,   Ef2    ,    Eb2   ,     Ef3,    Eb3   ,    Ef4    ,   Eb4   ,    Ef5   ,    Eb5   ,   Ef_5,   Eb_5  ,   Ef6  ,   Eb6  ,  Ef7  ,    Ef8  )

    # coverage correction
def Exp(x):
    # y2 = 1- 1/np.exp(1e48*x**100)    
    # y2 = 1 - 1/np.exp(0.8e10*x**21) #FOr allows
    # y2 = 1 - 1/np.exp(90*(x+0.08)**6)
    y2 =  1 - 1/np.exp(900*(x-0.02)**6)#y2 =  1 - 1/np.exp(1100*(x-0.09)**6) #
    return y2
   
    # mass of relevant reactants
m_NH3 = 17.03 * 1.66054e-27
m_N2 = 14 * 1.66054e-27
m_H2 = 2 * 1.66054e-27
m_H = 1.66054e-27

A= 1e-20 #Surface in m^2

Beta = 0.00025 #Beta  
T0 = 150 # Initial Temperature

t_end = 4e6 # Time end
th_NH3_0 = 0.0 #0.35,     0.25, 0.15, 0.05)
theta = 1-th_NH3_0 #np.ones(len(th_NH3_0)) -th_NH3_0
P = 1 #pressure in Pa
# graph style
col=        ('r','b','g','black',   'c', 'magenta','purple','orange','brown','darkgoldenrod')
linestyle = ('-','--','-.',':'   ,'-', '-', '-','-','-', '-')


# =============================================================================
# # 1/ Define function
# =============================================================================

def dydt(t,y, Ef1, Ef2, Eb2, Ef3, Eb3, Ef4, Eb4, Ef5, Eb5, Ef_5, Eb_5, Ef6, Eb6, Ef7, Ef8):
    
    dydt = np.zeros(12)
    T = Beta * t +T0
    
    #  0     1       2       3      4      5      6     7     8     9     10      11
    Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_NNH, Th_N2, Th_H,  Th,  P_N2,  P_H2, P_NH3 = y 
        
    # calculate all reaction rate constants        
        #Dissociation of NH3
    k_ads_1 = k_ads(T, P_NH3, A, m_NH3)/Beta *Qrot(T,3,9.444**2*6.196)*Qvib(T,frq_FS1)/Qvib(T,frq_IS1) #adsorption and desorption of NH3 calc_kads
    k_des_1 = k_surf(T, Ef1 )/Beta*Qrot(T,3,9.444**2*6.196)*Qvib(T,frq_IS1)/Qvib(T,frq_FS1)#k_des(T, 1e-20, m_NH3, 1, 8.92, Ef1)/Beta 
    # print(T, k_ads_1, k_des_1 )       
    
    k_f_2 = k_surf(T,  Ef2)/Beta * Qvib(T, frq_TS2) /Qvib(T, frq_IS2)
    k_b_2 = k_surf(T,  Eb2)/Beta * Qvib(T, frq_TS2) /Qvib(T, frq_FS2)  # dissociation of NH3
    
    k_f_3 = k_surf(T,  Ef3)/Beta * Qvib(T, frq_TS3) /Qvib(T, frq_IS3)
    k_b_3 = k_surf(T,  Eb3)/Beta * Qvib(T, frq_TS3) /Qvib(T, frq_FS3)  # dissociation of NH3
        
    k_f_4 = k_surf(T,  Ef4)/Beta  * Qvib(T, frq_TS4) /Qvib(T, frq_IS4)
    k_b_4 = k_surf(T,  Eb4) /Beta * Qvib(T, frq_TS4) /Qvib(T, frq_FS4) # dissociation of NH
    
    #------------------------------------
    k_f_5 = k_surf(T, Ef5 + alpha_N*  (Th_N+Th_NH +Th_H -0.33)  *Exp(Th_N))/Beta * Qvib(T, frq_TS5) /Qvib(T, frq_IS5)
    k_b_5 = k_surf(T, Eb5)/Beta * Qvib(T, frq_TS5) /Qvib(T, frq_FS5)  # N-N coupling
    
    k_f5 = k_surf(T, Ef_5)/Beta * Qvib(T, frq_TS_5) /Qvib(T, frq_IS5) #!!!
    k_b5 = k_surf(T, Eb_5)/Beta * Qvib(T, frq_TS_5) /Qvib(T, frq_FS5) # NH-N coupling
    
    k_f_6 = k_surf(T, Ef6)/Beta * Qvib(T, frq_TS6) /Qvib(T, frq_IS6)
    k_b_6 = k_surf(T, Eb6)/Beta * Qvib(T, frq_TS6) /Qvib(T, frq_FS6)  # N-N coupling
    
    k_f6 = k_surf(T, Ef_6)/Beta * Qvib(T, frq_TS_6) /Qvib(T, frq_IS6) #!!!
    k_b6 = k_surf(T, Eb_6)/Beta * Qvib(T, frq_TS_6) /Qvib(T, frq_FS6)  #NH-H = NN + H
    
    # k_f_7 = k_des(T, 1e20, m_N2, 2, 2.88, Ef7)/Beta  
    k_f_7 = k_surf(T, Ef7 )/Beta*Qrot(T,2,1.9982)*Qvib(T,frq_FS7)/Qvib(T,frq_IS7)  *Qtran(T,m_N2)   # N2 desorption
    k_b_7 = k_ads(T, P_N2, A, m_N2)/Beta *Qrot(T,2,1.9982)*Qvib(T,frq_IS7)/Qvib(T,frq_FS7)  *Qtran(T,m_N2)  #N2 adsorption
    
    #------------------------------------    
    # k_f_8 = k_des(T, 1e20, m_H2, 2, 87.6, E8)/Beta  
    k_f_8 =k_surf(T, Ef8 + alpa_H*Th_H)/Beta  *Qrot(T,2,60.853)*Qvib(T,frq_FS8)/Qvib(T,frq_IS8) *Qtran(T,m_H2) # H-H coupling and desorption
    k_b_8 = k_ads(T, P_H2, A, m_H2)/Beta *Qrot(T,2,60.853)*Qvib(T,frq_IS8)/Qvib(T,frq_FS8) *Qtran(T,m_H2) # H2 dissociative adsorbtion


# Rotational constant B source: http://www.lifesci.sussex.ac.uk/research/fluorine/p5qsp3l/sw_teaching/f1177_html/rotlab/node15.html
    
    # collect similar terms in new variables    
    rf1 = k_ads_1 * P_NH3* Th  # adsorption and desorption of NH3
    rb1 = k_des_1 * Th_NH3 ; 
    r1 = rf1 - rb1
    
    rf2 = k_f_2 * Th_NH3 * Th #good
    rb2 = k_b_2 * Th_NH2 * Th_H ; r2 = rf2-rb2; # K2=rf2/rb2
    
    rf3 = k_f_3 * Th_NH2 * Th #good
    rb3 = k_b_3 * Th_NH * Th_H ; r3 = rf3-rb3; #K3=rf3/rb3
    
    rf4 = k_f_4 * Th_NH * Th
    rb4 = k_b_4 * Th_N * Th_H ; r4 = rf4-rb4; #K4=rf4/rb4
    
    rf5 = k_f_5 * Th_N**2   
    rb5 = k_b_5 * Th_NN  ; r5 = rf5-rb5; 
    
    rf_5 = k_f5 * Th_NH *Th_N  # N* + NH* <=> *N-NH*
    rb_5 = k_b5 * Th_NNH ; r_5 = rf_5 - rb_5
    
    rf6 = k_f_6 * Th_NN
    rb6 = k_b_6 * Th_N2 * Th  ; r6 = rf6-rb6; # K6=rf6/rb6
    
    rf_6 = k_f6 * Th_NNH**2  #  *N-NH* <=> N2* + H
    rb_6 = k_b6 * Th_N2 * Th_H ;  r_6 = rf_6 - rb_6
    
    rf7 = k_f_7 * Th_N2
    rb7 = k_b_7 * P_N2* Th  ; 
    r7 = rf7 - rb7
    
    rf8 = k_f_8 * Th_H**2 
    rb8 = k_b_8 * Th**2 * P_H2 ; 
    r8 = rf8 - rb8
    
    dydt[0] = r1 - r2               # dθ(NH3)/dt
    dydt[1] = r2 - r3               # dθ(NH2)/dt
    dydt[2] = r3 - r4 - r_5         # dθ(NH)/dt
    dydt[3] = r4 - 2*r5 - r_5       # dθ(N)/dt 2*r5 
    dydt[4] = r5 - r6 # = 0         # dθ(N-N)/dt
    dydt[5] = r_5 - r_6             # dθ(N-NH)/dt
    dydt[6] = r6 + r_6 - r7         # dθ(N2)/dt
    dydt[7] = r2 + r3 + r4 + r_6 - 2*r8   # dθ(H)/dt
    dydt[8] =  - (dydt[0] + dydt[1] + dydt[2]+dydt[3]+dydt[4]+ dydt[5]+ dydt[6]+dydt[7])#-r1 -r2 -r3 -r4 + r6 + r7 +2*r8  # dθ(*)/dt   - dydt[6] - dydt[3] 
    
    dydt[9] = r7  # P_N2
    dydt[10] = r8  # P_H2
    dydt[11] = -r1  # P_NH3
    
    #dydt[10] = Beta   # Temperature changes with time
    
    return dydt

# =============================================================================
# 2/ Solve the differntial equation
# ============================================================================

y0 = [th_NH3_0 , 0 , 0 , 0 , 0 , 0 , 0 , 0, theta , 0 , 0, P] #P
r = solve_ivp(dydt, (0,t_end), y0, method="BDF", t_eval=np.linspace(0,t_end,1000), rtol = 1e-13, atol=1e-16, args =arguments)
Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_NNH, Th_N2, Th_H,  Th,  P_N2,  P_H2, P_NH3 = r.y 
#  0     1       2       3      4      5      6     7     8    9     10     11
t = r.t
T = Beta * t +T0

# =============================================================================
# 3/ Plot result
# =============================================================================
# print(t,T)
# Coverage
fig, ax = plt.subplots(figsize=(8, 7))
plt.plot(T,Th_NH3, label= r"$\theta_{NH_3}$")
plt.plot(T,Th_NH2, label= r"$\theta_{NH_2}$", color ='r')
plt.plot(T,Th_NH, label= r"$\theta_{NH}$",  color ='g')
plt.plot(T,Th_N, label= r"$\theta_{N}$",  color ='b')
plt.plot(T,Th_NN, label= r"$\theta_{N-N}$",  color ='black')
plt.plot(T,Th_NNH, label= r"$\theta_{N-NH}$",  color ='c')
plt.plot(T,Th_N2, label= r"$\theta_{N_2}$",  color ='orange')
plt.plot(T,Th_H, label= r"$\theta_H$",  color ='magenta')
plt.plot(T,Th, label= r"$\theta$", ls = "--")

plt.title('P=1, Beta =0.00025', fontsize=18)
plt.ylabel("Coverage (ML)", fontsize=16)
plt.xlabel("Temperature (K)", fontsize=16)
plt.xlim(200, 1200)

plt.legend()
plt.show()

# ratio molecule : free site
fig, ax = plt.subplots(figsize=(8, 7))
plt.plot(T,P_NH3, label= "P NH3")
plt.plot(T,P_N2, label= "P N2")
plt.plot(T,P_H2, label= "P H2")
plt.xlabel("Temperature (K)", fontsize=16)
plt.ylabel("Molecule:free site", fontsize=16)
plt.xlim(200, 1200)
plt.title('P=1, Beta =0.00025', fontsize=18)

plt.legend()
plt.show()
    
print("We are DONE! :)")
