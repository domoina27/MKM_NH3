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
        2N* = *N-N*
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


frequency = pd.read_csv('Ru(0001).csv', na_filter=True)
# frequency = pd.read_csv('Co2Ni.csv', na_filter=True)


# =============================================================================
# # 0/ Constants
# =============================================================================
# step 2 - NH3* + * <=> NH2* + H*   Dehydrogenation
frq_IS2 = frequency.IS2.dropna(how='any')
frq_TS2 = frequency.TS2.dropna(how='any')
frq_FS2 = frequency.FS2.dropna(how='any')

#  step 3 -NH2* + * <=> NH* + H*    Dehydrogenation
frq_IS3 = frequency.FS3.dropna(how='any') 
frq_TS3 = frequency.TS3.dropna(how='any')
frq_FS3 = frequency.FS3.dropna(how='any')

#  step 4 - NH* + * <=> NH* + H*    Dehydrogenation
frq_IS4 = frequency.FS4.dropna(how='any') 
frq_TS4 = frequency.TS4.dropna(how='any')
frq_FS4 = frequency.FS4.dropna(how='any')

#  step 5 - N* + N* <=> *N-N*       N-N recombination
frq_IS5 = frequency.FS5.dropna(how='any') 
frq_TS5 = frequency.TS5.dropna(how='any')
frq_FS5 = frequency.FS5.dropna(how='any')

#  step 6 - *N-N* <=> N2*  +*       N-N rotation
frq_IS6 = frequency.FS5.dropna(how='any')
frq_TS6 = frequency.TS6.dropna(how='any')
frq_FS6 = frequency.FS6.dropna(how='any')

#  step 7 - N2* <=> N2 (gas)  +*       N2 desorbtion
frq_IS7 = frequency.FS6.dropna(how='any')
frq_FS7 = frequency.FS7.dropna(how='any')

#  step 8 - H* + H* <=> H2 (gas) + 2*       N-N recombination
frq_IS8 = frequency.IS8.dropna(how='any')
frq_FS8 = frequency.FS8.dropna(how='any')



# Ruthenium

# frequency = pd.read_csv('Ru(0001).csv', na_filter=True)
#Hydrogen coverage
# alpa_H= 0; intercept_H = 0.0 
alpa_H= -0.1246*96.48530749926e3; intercept_H = 0#.6622*2*96.48530749926e3 #convert eV to KJ/mol

#Nitrogen coverage
# alpa_N= 0; intercept_N = 0
alpa_N= -1.4951*96.48530749926e3; intercept_N = 0#1.1521*96.48530749926e3*2 #convert eV to KJ/mol

#ene =    E1   ,   Ef2,      Eb2,     Ef3,     Eb3,        Ef4,    Eb4,      Ef5,    Eb5,     Ef6,     Eb6,      Ef7,    E8       
Eng = (85.44e3, 109.99e3 ,113.85e3, 68.50e3, 119.93e3, 40.52e3, 40.52e3, 247.97e3, 65.61e3, 14.47e3, 84.48e3, 58.86e3, 117.24E3)     #Ru(0001)


#------------------------------------------------------------------------------
# Co2Ni (111)

# frequency = pd.read_csv('Co2Ni.csv', na_filter=True)
#Hydrogen coverage
# alpa_H= 0; intercept_H = 0.0 
# alpa_H= -0.07*96.48530749926e3; intercept_H = 0#.6622*2*96.48530749926e3 #convert eV to KJ/mol

#Nitrogen coverage
# alpa_N= 0; intercept_N = 0
# alpa_N= -1.95*96.48530749926e3; intercept_N = 0#1.1521*96.48530749926e3*2 #convert eV to KJ/mol

# Eng = (71180.58, 110886.82, 115082.9827, 35554.10, 67174.02725, 122610.57, 96010.14438, 165400.69, 104540.668, 23586.70, 74596.32934, 48593.48, 100539.09)     #Co2Ni     
#------------------------------------------------------------------------------
# CoNi2 (111)

# frequency = pd.read_csv('CoNi2.csv', na_filter=True)
#------------------------------------------------------------------------------
# Co (111)
# frequency = pd.read_csv('Co.csv', na_filter=True)

#------------------------------------------------------------------------------
# Ni (111)
#frequency = pd.read_csv('Ni.csv', na_filter=True)

#------------------------------------------------------------------------------

# E1  = Eng[0]

# Ef2 = Eng[1]+(ZPE(frq_TS2)-ZPE(frq_IS2))
# Eb2 = Eng[2]+(ZPE(frq_TS2)-ZPE(frq_FS2))

# Ef3 = Eng[3]+(ZPE(frq_TS3)-ZPE(frq_IS3))
# Eb3 = Eng[4]+(ZPE(frq_TS3)-ZPE(frq_FS3))

# Ef4 = Eng[5]+(ZPE(frq_TS4)-ZPE(frq_IS4))
# Eb4 = Eng[6]+(ZPE(frq_TS4)-ZPE(frq_FS4)) 
 
# Ef5 = Eng[7]+(ZPE(frq_TS5)-ZPE(frq_IS5)) 
# Eb5 = Eng[8]+(ZPE(frq_TS5)-ZPE(frq_FS5))  
 
# Ef6 = Eng[9]+(ZPE(frq_TS6)-ZPE(frq_IS6))    
# Eb6 = Eng[10]+(ZPE(frq_TS6)-ZPE(frq_FS6))
    
# Ef7 = Eng[11]+(ZPE(frq_FS7)-ZPE(frq_IS7))   

# E8  = Eng[12]+(ZPE(frq_FS8)-ZPE(frq_FS8))

# arguments =    (E1    ,   Ef2    ,    Eb2   ,     Ef3,    Eb3   ,    Ef4    ,   Eb4   ,    Ef5   ,    Eb5   ,   Ef6  ,   Eb6  ,  Ef7    ,    E8 )
# arguments =      (85.44e3, 109.99e3, 113.85e3 , 68.50e3, 119.93e3 , 40.52e3   , 40.52e3 , 247.97e3 , 65.61e3 , 14.47e3 , 84.48e3, 58.86e3, 117.24E3) 
arguments =    (85440.0, 92305.57, 111047.08, 65743.56, 117173.57, 34051.62, 34051.61, 246516.13, 64156.128, 12829.46, 79158.52, 53611.83, 117240.0) #WITH ZPE
    
    # mass of relevant reactants
m_NH3 = 17.03 * 1.66054e-27
m_N2 = 14 * 1.66054e-27
m_H2 = 2 * 1.66054e-27
m_H = 1.66054e-27

Beta = 4.0 #Beta  
# pt = 0#e-20 # Total pressure
T0 = 150 # Initial Temperature

t_end = 300 # Time end
th_NH3_0 = 0.15#1.35/9
theta = 1-th_NH3_0

# Initial value
# P = pt * 1e5
y0 = [th_NH3_0 , 0 , 0 , 0 , 0 , 0 , 0 , theta , 0 , 0 ]#, T0]


# if th_NH3_0 > 0.33:
#     alpa_N= -1.95*96.48530749926e3
# else:
#     alpa_N= 0

#arguments = (140787, 74285 ,115109, 118872, 92242, 165379, 104495)     #Co2Ni

# =============================================================================
# # 1/ Define function
# =============================================================================

def dydt(t,y, E1,  Ef2, Eb2,  Ef3,Eb3,  Ef4,Eb4,  Ef5,Eb5,  Ef6,Eb6,  Ef7,  E8):
    
    dydt = np.zeros(10)
    T = Beta * t +T0
    
    #  0     1       2       3      4      5      6    7     8    9       
    Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_N2, Th_H,  Th,  P_N2,  P_H2 = y 
        
    # calculate all reaction rate constants        
        #Dissociation of NH3
    k_des_1 = k_des(T, 1e-20, m_NH3, 1, 8.92, E1)/Beta #adsorption and desorption of NH3
           
    k_f_2 = k_surf(T,  Ef2)/Beta * Qvib(T, frq_TS2) /Qvib(T, frq_IS2)
    k_b_2 = k_surf(T,  Eb2)/Beta * Qvib(T, frq_TS2) /Qvib(T, frq_FS2)  # dissociation of NH3
    
    k_f_3 = k_surf(T,  Ef3)/Beta * Qvib(T, frq_TS3) /Qvib(T, frq_IS3)
    k_b_3 = k_surf(T,  Eb3)/Beta * Qvib(T, frq_TS3) /Qvib(T, frq_FS3)  # dissociation of NH3
        
    k_f_4 = k_surf(T,  Ef4)/Beta  * Qvib(T, frq_TS4) /Qvib(T, frq_IS4)
    k_b_4 = k_surf(T,  Eb4) /Beta * Qvib(T, frq_TS4) /Qvib(T, frq_FS4) # dissociation of NH
    
    #------------------------------------
    # if Th_N > 0.33:
    #     alpa_N= -1.95*96.48530749926e3
    # else:
    #     alpa_N= 0
    k_f_5 = k_surf(T, Ef5 + alpa_N*Th_N)/Beta * Qvib(T, frq_TS5) /Qvib(T, frq_IS5)
    k_b_5 = k_surf(T, Eb5)/Beta * Qvib(T, frq_TS5) /Qvib(T, frq_FS5)  # N-N coupling
    
    k_f_6 = k_surf(T, Ef6)/Beta * Qvib(T, frq_TS6) /Qvib(T, frq_IS6)
    k_b_6 = k_surf(T, Eb6)/Beta * Qvib(T, frq_TS6) /Qvib(T, frq_FS6)  # N-N coupling
    
    # k_f_7 = k_des(T, 1e20, m_N2, 2, 2.88, Ef7)/Beta  
    k_f_7 = k_surf(T, Ef7 )/Beta*Qrot(T,2,1.9982)*Qvib(T,frq_FS7)/Qvib(T,frq_IS7)  *Qtran(T,m_N2)   # N2 desorption
    
    #------------------------------------    
    # k_f_8 = k_des(T, 1e20, m_H2, 2, 87.6, E8)/Beta  
    k_f_8 =k_surf(T, E8 + alpa_H*Th_H)/Beta  *Qrot(T,2,60.853)*Qvib(T,frq_FS8)/Qvib(T,frq_IS8) *Qtran(T,m_H2) # H-H coupling and desorption

# Rotational constant B source: http://www.lifesci.sussex.ac.uk/research/fluorine/p5qsp3l/sw_teaching/f1177_html/rotlab/node15.html
    
    # collect similar terms in new variables    
    rb1 = k_des_1 * Th_NH3 ; r1 = 0
    
    rf2 = k_f_2 * Th_NH3 * Th #good
    rb2 = k_b_2 * Th_NH2 * Th_H ; r2 = rf2-rb2; # K2=rf2/rb2
    
    rf3 = k_f_3 * Th_NH2 * Th #good
    rb3 = k_b_3 * Th_NH * Th_H ; r3 = rf3-rb3; #K3=rf3/rb3
    
    rf4 = k_f_4 * Th_NH * Th
    rb4 = k_b_4 * Th_N * Th_H ; r4 = rf4-rb4; #K4=rf4/rb4
    
    rf5 = k_f_5 * Th_N**2   
    rb5 = k_b_5 * Th_NN  ; r5 = rf5-rb5; 
    
    rf6 = k_f_6 * Th_NN   
    rb6 = k_b_6 * Th_N2 * Th  ; r6 = rf6-rb6; # K6=rf6/rb6
    
    r7 = k_f_7 * Th_N2
    
    r8 = k_f_8 * Th_H**2   
    
    dydt[0] = -r2               # dθ(NH3)/dt
    dydt[1] = r2 - r3 # = 0    # dθ(NH2)/dt
    dydt[2] = r3 - r4          # dθ(NH)/dt
    dydt[3] = r4 - 2*r5        # dθ(N)/dt
    dydt[4] = r5 - r6 # = 0    # dθ(N-N)/dt
    dydt[5] = r6 - r7 # = 0    # dθ(N2)/dt
    dydt[6] = r2 + r3 + r4 - 2*r8   # dθ(H)/dt
    dydt[7] = -r2 -r3 -r4 + r6 + r7 +2*r8  # dθ(*)/dt   - dydt[6] - dydt[3] 
    
    dydt[8] = r7  # P_N2
    dydt[9] = r8  # P_H2
    
    #dydt[10] = Beta   # Temperature changes with time
    
    return dydt

# =============================================================================
# 2/ Solve the differntial equation
# =============================================================================

# Define the JACOBIAN matrix

def JACOB(t,x, E1,  Ef2, Eb2,  Ef3,Eb3,  Ef4,Eb4,  Ef5,Eb5,  Ef6,Eb6,  Ef7,  E8):
    #  0     1       2       3      4      5      6    7     8    9   
    Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_N2, Th_H,  Th,  P_N2,  P_H2 = x
    T = Beta * t + T0
    #Rate constants
        #formation of N2
       
    k_f2 = k_surf(T,  Ef2)/Beta * Qvib(T, frq_TS2) /Qvib(T, frq_IS2)
    k_b2 = k_surf(T,  Eb2)/Beta * Qvib(T, frq_TS2) /Qvib(T, frq_FS2) # dissociation of NH3
    
    k_f3 = k_surf(T,  Ef3)/Beta * Qvib(T, frq_TS3)/Qvib(T, frq_IS3)
    k_b3 = k_surf(T,  Eb3)/Beta * Qvib(T, frq_TS3)/Qvib(T, frq_FS3) # dissociation of NH3
        
    k_f4 = k_surf(T,  Ef4)/Beta  * Qvib(T, frq_TS4)/Qvib(T, frq_IS4)
    k_b4 = k_surf(T,  Eb4) /Beta * Qvib(T, frq_TS4)/Qvib(T, frq_FS4) # dissociation of NH
    
    #------------------------------------
    # if Th_N > 0.33:
    #     alpa_N= -1.95*96.48530749926e3
    # else:
    alpa_N= 0
        
    k_f5 = k_surf(T, Ef5+ alpa_N*Th_N)/Beta * Qvib(T, frq_TS5) /Qvib(T, frq_IS5)
    k_b5 = k_surf(T, Eb5)/Beta * Qvib(T, frq_TS5) /Qvib(T, frq_FS5)  # N-N coupling
    
    k_f6 = k_surf(T, Ef6)/Beta * Qvib(T, frq_TS6) /Qvib(T, frq_IS6)
    k_b6 = k_surf(T, Eb6)/Beta * Qvib(T, frq_TS6) /Qvib(T, frq_FS6)  # N-N coupling
    
    # k_f7 = k_des(T, 1e20, m_N2, 2, 2.88, Ef7)/Beta  
    k_f7 = k_surf(T, Ef7 )/Beta *Qrot(T,2,1.9982)*Qvib(T,frq_FS7)/Qvib(T,frq_IS7) *Qtran(T,m_N2)  # N2 desorption
    
    #------------------------------------    
    # k_f8 = k_des(T, 1e20, m_H2, 2, 87.6, E8)/Beta  
    k_f8 =k_surf(T, E8 + alpa_H*Th_H)/Beta  *Qrot(T,2,60.853)*Qvib(T,frq_FS8)/Qvib(T,frq_IS8) *Qtran(T,m_H2) # H-H coupling
        
    #------------------------------------    
    #             0       7              1       6
    r2 = k_f2 * Th_NH3 * Th - k_b2 * Th_NH2 * Th_H   
    dr2_dx0=  k_f2 * Th
    dr2_dx1= - k_b2 * Th_H
    dr2_dx6= - k_b2 * Th_NH2
    dr2_dx7=  k_f2 * Th_NH3
    
    # dkf2_dT = dk(T, frq_TS2, frq_IS2,Ef2) #; print(dkf2_dT)
    # dkb2_dT = dk(T, frq_TS2, frq_FS2,Eb2)
    # dr2_dx10 = dkf2_dT * Th_NH3 * Th  - dkb2_dT * Th_NH2 * Th_H
    
     #             1       7              2       6
    r3 = k_f3 * Th_NH2 * Th - k_b3 * Th_NH * Th_H 
    dr3_dx1=  k_f3 * Th
    dr3_dx2= - k_b3 * Th_H
    dr3_dx6= - k_b3 * Th_NH
    dr3_dx7=  k_f3 * Th_NH2
    
    # dkf3_dT = dk(T, frq_TS3, frq_IS3,Ef3)
    # dkb3_dT = dk(T, frq_TS3, frq_FS3,Eb3)
    # dr3_dx10 = dkf3_dT * Th_NH2 * Th  - dkb3_dT * Th_NH * Th_H
    
    #             2       7           3       6
    r4 = k_f4 * Th_NH * Th - k_b4 * Th_N * Th_H 
    dr4_dx2=  k_f4 * Th
    dr4_dx3= - k_b4 * Th_H
    dr4_dx6= - k_b4 * Th_N
    dr4_dx7=  k_f4 * Th_NH
    
    # dkf4_dT = dk(T, frq_TS4, frq_IS4,Ef4)
    # dkb4_dT = dk(T, frq_TS4, frq_FS4,Eb4)
    # dr4_dx10 = dkf4_dT * Th_NH * Th  - dkb4_dT * Th_N * Th_H
    
    #             3               4
    r5 = k_f5 * Th_N**2 - k_b5 * Th_NN 
    dr5_dx3=  2*k_f5 * Th_N
    dr5_dx4= - k_b5 
        
    # dkf5_dT = dk(T, frq_TS5, frq_IS5,Ef5)
    # dkb5_dT = dk(T, frq_TS5, frq_FS5,Eb5)
    # dr5_dx10 = dkf5_dT * Th_N**2 -  dkb5_dT* Th_NN
    
    #             4               5       7
    r6 = k_f6 * Th_NN  - k_b6 * Th_N2 * Th  
    dr6_dx4=  k_f6 
    dr6_dx5= - k_b6 * Th
    dr6_dx7= - k_b6 * Th_N2
    
    # dkf6_dT = dk(T, frq_TS6, frq_IS6,Ef6)
    # dkb6_dT = dk(T, frq_TS6, frq_FS6,Eb6)
    # dr6_dx10 = dkf6_dT * Th_NN - dkb6_dT * Th_N2 * Th 
    
    #             5 
    r7 = k_f7 * Th_N2
    dr7_dx5=  k_f7
    
    dkf7_dT = dkdes_dT(T, 1e20, m_N2, 2, 2.88, Ef7)
    dr7_dx10 = dkf7_dT * Th_N2
    
    #             6 
    r8 = k_f8 * Th_H**2
    dr8_dx6=  k_f8*2*Th_H
    
    dkf8_dT = dkdes_dT(T, 1e20, m_H2, 2, 87.6, E8)
    dr8_dx10 = dkf8_dT * Th_H**2  
    
        #--------------------------
    # Derivatives dfi/dxi
   
    #f0 = -r2
    df0_dx0 = -dr2_dx0
    df0_dx1 = -dr2_dx1
    df0_dx6 = -dr2_dx6
    df0_dx7 = -dr2_dx7
    # df0_dx10 = -dr2_dx10
    
        
    #f1 = r2-r3
    #               0       7              1       6               1       7            2       6
    # f2 = k_f2 * Th_NH3 * Th - k_b2 * Th_NH2 * Th_H - (k_f3 * Th_NH2 * Th - k_b3 * Th_NH * Th_H)
    df1_dx0 = dr2_dx0 - 0
    df1_dx1 = dr2_dx1 - dr3_dx1
    df1_dx2 =       0 - dr3_dx2
    df1_dx6 = dr2_dx6 - dr3_dx6
    df1_dx7 = dr2_dx7 - dr3_dx7
    # df1_dx10= dr2_dx10 - dr3_dx10
        #--------------------------
    # f2 = r3 - r4
    #               1       7            2       6              2      7           3       6
    # f2 = k_f3 * Th_NH2 * Th - k_b3 * Th_NH * Th_H - (k_f4 * Th_NH * Th - k_b4 * Th_N * Th_H)
    df2_dx1 = dr3_dx1 - 0
    df2_dx2 = dr3_dx2 - dr4_dx2
    df2_dx3 =       0 - dr4_dx3
    df2_dx6 = dr3_dx6 - dr4_dx6
    df2_dx7 = dr3_dx7 - dr4_dx7
    # df2_dx10= dr3_dx10 - dr4_dx10
        #--------------------------
    # f3 = r4 - 2*r5
    #                2      7           3     6                  3               4
    # f3 = k_f4 * Th_NH * Th - k_b4 * Th_N * Th_H - 2*(k_f5 * Th_N**2 - k_b5 * Th_NN)
    df3_dx2 = dr4_dx2 - 0
    df3_dx3 = dr4_dx3 - 2* dr5_dx3
    df3_dx4 =       0 - 2* dr5_dx4
    df3_dx6 = dr4_dx6 - 0
    df3_dx7 = dr4_dx7 - 2* 0
    # df3_dx10= dr4_dx10 - 2* dr5_dx10
    
        #--------------------------
    # f4 = r5 - r6
    #                3               4               4             5       7
    # f4 = k_f5 * Th_N**2 - k_b5 * Th_NN - k_f6 * Th_NN  - k_b6 * Th_N2 * Th 
    df4_dx3 = dr5_dx3 - 0
    df4_dx4 = dr5_dx4 - dr6_dx4
    df4_dx5 =       0 - dr6_dx5
    df4_dx7 =       0 - dr6_dx7
    # df4_dx10= dr5_dx10 - dr6_dx10 
    
        #--------------------------
    #f5 =  r6 - r7
    #f5 =  (k_f6 * Th_NN  - k_b6 * Th_N2 * Th ) -  k_f7 * Th_N2
    #                4             5       7             5 
    df5_dx4 = dr6_dx4 - 0
    df5_dx5 = dr6_dx5 - dr7_dx5
    df5_dx7 = dr6_dx7 - 0
    # df5_dx10= dr6_dx10 - dr7_dx10
   
    #--------------------------
    #f6 =  r2 + r3 + r4 - 2*r8
    #f6 =  (k_f2 * Th_NH3 * Th - k_b2 * Th_NH2 * Th_H) + (k_f3 * Th_NH2 * Th - k_b3 * Th_NH * Th_H) +  (k_f4 * Th_NH * Th - k_b4 * Th_N * Th_H) -2* (k_f8 * Th_H**2)
    #               0       7              1       6               1       7            2       6              2       7           3       6                   6 
    df6_dx0 = dr2_dx0 + 0
    df6_dx1 = dr2_dx1 + dr3_dx1 +   0   -  0
    df6_dx2 =    0    + dr3_dx2 +dr4_dx2-  0
    df6_dx3 =    0    +    0    +dr4_dx3-  0
    df6_dx6 = dr2_dx6 + dr3_dx6 +dr4_dx6-  2*dr8_dx6
    df6_dx7 = dr2_dx7 + dr3_dx7 +dr4_dx7-  0
    # df6_dx10= dr2_dx10 + dr3_dx10+ dr4_dx10 - 2*dr8_dx10
    
    #--------------------------
    # f7 = -r2 -r3 -r4 +2*r8 + r6 + r7
    # f7 = -(k_f2 * Th_NH3 * Th - k_b2 * Th_NH2 * Th_H) - (k_f3 * Th_NH2 * Th - k_b3 * Th_NH * Th_H) -  (k_f4 * Th_NH * Th - k_b4 * Th_N * Th_H) +2* (k_f8 * Th_H**2) + (k_f6 * Th_NN  - k_b6 * Th_N2 * Th ) + ( k_f7 * Th_N2)
    #               0       7              1       6               1       7            2       6              2       7           3       6                   6         #         4             5       7               5 
    df7_dx0 = -dr2_dx0 - 0
    df7_dx1 = -dr2_dx1 - dr3_dx1 -   0     +  0         +  0        + 0       
    df7_dx2 =    0    - dr3_dx2 -  dr4_dx2 +  0         +  0        + 0       
    df7_dx3 =    0    -    0    -  dr4_dx3 +  0         +  0        + 0       
    df7_dx4 =    0    -    0    -    0     +  0         +  dr6_dx4  + 0
    df7_dx5 =    0    -    0    -    0     +  0         +  dr6_dx5  + dr7_dx5       
    df7_dx6 = -dr2_dx6 - dr3_dx6 - dr4_dx6 +  2*dr8_dx6 + 0         + 0
    df7_dx7 = -dr2_dx7 - dr3_dx7 - dr4_dx7 +  0         +  dr6_dx7  + 0 
    # df7_dx10= -dr2_dx10 - dr3_dx10-dr4_dx10 + 2*dr8_dx10+ dr6_dx10  + dr7_dx10
    
    #f8 = r7
    # f8 = ( k_f7 * Th_N2) #5
    df8_dx5 = dr7_dx5
    df8_dx10 = dr7_dx10
    
    #f9 = r8  # P_H2
    # f9= (k_f8 * Th_H**2 ) #6
    df9_dx6  = dr8_dx6
    df9_dx10 = dr8_dx10
    
    #f10 = Beta   
    
   #         0         1        2       3         4       5      6       7        8      9       10
   #      Th_NH3  , Th_NH2,   Th_NH ,  Th_N  ,  Th_NN , Th_N2 , Th_H ,  Th    ,  P_N2 , P_H2 ,   T 
    J0 = [df0_dx0 ,df0_dx1 ,   0    ,   0    ,    0   ,  0    ,df0_dx6 ,df0_dx7,   0   ,  0   ]#, df0_dx10]
    J1 = [df1_dx0 ,df1_dx1 , df1_dx2,   0    ,    0   ,  0    ,df1_dx6 ,df1_dx7,   0   ,  0   ]#, df1_dx10] 
    J2 = [   0    ,df2_dx1 , df2_dx2, df2_dx3,    0   ,  0    ,df2_dx6 ,df2_dx7,   0   ,  0   ]#, df2_dx10] 
    J3 = [   0    ,   0    , df3_dx2, df3_dx3, df3_dx4,  0    ,df3_dx6 ,df3_dx7,   0   ,  0   ]#, df3_dx10]
    J4 = [   0    ,   0    ,    0   , df4_dx3, df4_dx4,df4_dx5,   0    ,df4_dx7,   0   ,  0   ]#, df4_dx10]
    J5=  [   0    ,   0    ,    0   ,  0     , df5_dx4,df5_dx5,   0    ,df5_dx7,   0   ,  0   ]#, df5_dx10]
    J6=  [df6_dx0 , df6_dx1, df6_dx2, df6_dx3,    0   ,  0    ,df6_dx6 ,df6_dx7,   0   ,  0   ]#, df6_dx10]
    J7=  [df7_dx0 , df7_dx1, df7_dx2, df7_dx3, df7_dx4,df7_dx5,df7_dx6 ,df7_dx7,   0   ,  0   ]#, df7_dx10]
    J8=  [   0    ,   0    ,    0   ,  0    ,     0   ,df8_dx5 ,   0   ,   0   ,   0   ,  0   ]#, df8_dx10]
    J9=  [   0    ,   0    ,    0   ,  0    ,     0   ,    0   ,df9_dx6,   0   ,   0   ,  0   ]#, df9_dx10]
    # J10= [   0    ,   0    ,    0   ,  0    ,     0   ,    0   ,  0    ,   0   ,   0   ,  0   ,     0   ]
    
    Jacobian = np.array([J0, J1, J2, J3, J4, J5, J6, J7, J8, J9])#, J10])
    return Jacobian


r = solve_ivp(dydt, (0,t_end), y0, method="BDF", t_eval=np.linspace(0,t_end,3000), rtol = 1e-12, atol=1e-16, args =arguments,  vectorized = True, dense_output=True,jac=JACOB)

Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_N2, Th_H,  Th,  P_N2,  P_H2 = r.y 
#  0     1       2       3      4      5      6     7     8    9    
t = r.t
T = Beta * t +T0

# =============================================================================
# 3/ Plot result
# =============================================================================

    # Coverage
# plt.plot(T,Th_NH3, label= r"$\theta_{NH_3}$")
# plt.plot(T,Th_NH2, label= r"$\theta_{NH_2}$")
# plt.plot(T,Th_NH, label= r"$\theta_{NH}$")
# plt.plot(T,Th_N, label= r"$\theta_{N}$")
# plt.plot(T,Th_NN, label= r"$\theta_{N-N}$")
# plt.plot(T,Th_N2, label= r"$\theta_{N_2}$")
# plt.plot(T,Th_H, label= r"$\theta_H$")
# plt.plot(T,Th, label= r"$\theta$", ls = "--")
# plt.ylabel("Coverage (ML)", fontsize=16)
# plt.xlabel("Temperature (K)", fontsize=16)
# plt.show()
# plt.legend()

#     # pressure
# plt.plot(T,P_NH3, label= "P NH3")
# plt.plot(T,P_N2, label= "P N2")
# plt.plot(T,P_H2, label= "P H2")
# plt.xlabel("Temperature (K)", fontsize=16)
# plt.ylabel("Pressure", fontsize=16)
# plt.show()

# =============================================================================
#  4/ Caculate Rate
# =============================================================================
print("Starting rate calculations")
#------------------------------------
# alpa_N = 0#np.zeros(len(T))
for i in range(len(T)):
    
    # if Th_N[i] > 0.33 :
    #     alpa_N= -1.95*96.48530749926e3#np.full((1,(len(T))),-1.95*96.48530749926e3)       
    # else:
    #     alpa_N = 0#np.full((1,(len(T))),0)
    
    # kf5 = k_surf(T, arguments[7] + alpa_N*Th_N)/Beta * Qvib(T[i], frq_TS5) /Qvib(T[i], frq_IS5)
    # kb5 = k_surf(T, arguments[8])/Beta * Qvib(T[i], frq_TS5) /Qvib(T[i], frq_FS5)
    
    kf5 = k_surf(T, arguments[7] + alpa_N*Th_N[i])/Beta * Qvib(T[i], frq_TS5) /Qvib(T[i], frq_IS5)
    kb5 = k_surf(T, arguments[8])/Beta * Qvib(T[i], frq_TS5) /Qvib(T[i], frq_FS5)
   
    rf5 = kf5* Th_N**2 
    rb5 = kb5*Th_NN ;  r5 = rf5 -rb5

# k7 = k_des(T, 1e20, m_N2, 2, 2.88, arguments[11])/Beta 
    k7 = k_surf(T,arguments[11] )/Beta *Qrot(T,2,1.9982)*Qvib(T[i],frq_FS7)/Qvib(T[i],frq_IS7) *Qtran(T,m_N2)
    r7 = k7*Th_N2

# plt.plot(T,r5, color = "r", ls = '-',linewidth=2, label= r"N-N recombination")
# plt.plot(T,r7, color = "g", ls = '--',linewidth=2, label= r"full MKM model")

#------------------------------------    
# k_f_8 = k_des(T, 1e20, m_H2, 2, 87.6, arguments[12])/Beta 
    k_f_8 = k_surf(T,arguments[12] + alpa_H*Th_H[i])/Beta  *Qrot(T,2,60.853)*Qvib(T[i],frq_FS8)/Qvib(T[i],frq_IS8) *Qtran(T,m_H2) # H-H coupling

    r8 = k_f_8 * Th_H**2   

print("done")


# =============================================================================
# 5/ Plot TPD

# =============================================================================
B =str(Beta)
def k_surf0(T, nu, Eact):
    R = 8.3144598 # gas constant
    return nu* np.exp(-Eact / (R * T))
k_ter = k_surf0(T, 1e12, 218e3 )/Beta  
# k_ter = k_surf0(T, 1e10, 150e3 )/Beta
r_ter = k_ter*Th_N**2


# =============================================================================
# Write a file that contains a table
# =============================================================================
# from astropy.io import ascii
# from astropy.table import Table

# data = Table()
# data['Temp'] = T
# data['N2'] = r7
# data['N-N recomb'] = r5
# data['H2'] = r8
# ascii.write(data, 'Co2Ni_TPD_Cov_0.37.dat', overwrite=True)  
# file = open("Myfilename", "w") #write a file that does not exist

# =============================================================================
# Plot figures
# =============================================================================

# plt.plot(T,r_ter, color = "b", ls = '-.',linewidth=2, label= r"Exp(Terrace)")

# plt.plot(T,r7, label= str(round(th_NH3_0,2)), color ="black", ls ="-")
plt.plot(T,r7, label= "N2 ", color ="b", ls ="-")
plt.plot(T,r5, label= "N-N recomb ", ls = '-', color ="orange")
plt.plot(T,r8, label= "H2 ", color ="red", ls ="-")

plt.title("TPD at "+ str(round(th_NH3_0,2))+" coverage", fontsize=15)
plt.xlabel("Temperature (K)")
plt.ylabel("Rate")

plt.rc('xtick', labelsize=18)
plt.title(r"TPD at $\beta$= " + B +r" K/s and "+str(round(th_NH3_0,2)) +r" coverage", fontsize=18)
plt.xlabel("Temperature (K)", fontsize=16)
plt.ylabel("Rate", fontsize=16)
plt.xlim(200,1300)

plt.legend()
plt.show()
