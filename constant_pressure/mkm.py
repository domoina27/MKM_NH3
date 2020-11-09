#!/usr/bin/env python
'''
Author: Domoina Holiharimanana
date: 7/14/2020
'''
# %matplotlib
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from rate_constant_func import calc_kads as k_ads
from rate_constant_func import calc_kdes as k_des
from rate_constant_func import calc_k_arr as k_surf

from rate_constant_func import dksur_dT
from rate_constant_func import dkdes_dT

    # mass of relevant reactants
m_NH3 = 17.03 * 1.66054e-27
m_N2 = 14 * 1.66054e-27
m_H2 = 2 * 1.66054e-27
m_H = 1.66054e-27

Beta = 10
pt = 1 # Total pressure
T0 = 20 # Initial Temperature

t_end = 1000 #/Beta # Time end
th_NH3_0 =1/9
theta = 1-th_NH3_0

# Initial value
P = pt * 1e5
y0 = [P,th_NH3_0 , 0 , 0 , 0 , 0 , 0 , 0 , theta , 0 , 0 , T0]

#Arguments

#           (  E1 ,   Ef2 ,   Eb2 ,  Ef4  , Eb4  ,  Ef5  ,  Eb5  )
arguments = (63787, 110285, 115109, 118872, 92242, 165379, 104495)     #Co2Ni
# arguments = (56927, 97452, 110960, 105171, 88865, 158239, 93785)     #Co
# arguments = (xx,xx,115e3, 118e3,92e3 ,165e3, xx)     #CoNi2
# arguments = (140e3,xx,xx, 118e3,xxx ,165e3, 104e3)     #Ni

def dydt(t,y,   E1, Ef2, Eb2, Ef4, Eb4, Ef5, Eb5):
    
    dydt = np.zeros(12)
    
    #  0     1       2     3      4      5      6     7     8      9     10   11
    P_NH3, Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_N2, Th_H, Th,  P_N2,  P_H2, T = y 
        
    # calculate all reaction rate constants        
        #Dissociation of NH3
    k_des_1 = k_des(T, 1e-20, m_NH3, 1, 8.92, E1)/Beta #adsorption and desorption of NH3
           
    k_f_2 = k_surf(T, 1e13, Ef2)/Beta
    k_b_2 = k_surf(T, 1e13, Eb2 )/Beta  # dissociation of NH3
        
    k_f_4 = k_surf(T, 1e13, Ef4)/Beta
    k_b_4 = k_surf(T, 1e13,  Eb4) /Beta  # dissociation of NH
    
    k_f_5 = k_surf(T, 1e13, Ef5)/Beta
    k_b_5 = k_surf(T, 1e13,  Eb5)/Beta   # N-N coupling
    
        #formation of H2
    k_des_8 = k_des(T, 1e-20, m_H2, 2, 87.6, 105.17e3)/Beta
    k_ads_8 = k_ads(T, P_H2, 1e-20, m_H2)/Beta
        
        
    # collect similar terms in new variables    
    rb1 = k_des_1 * Th_NH3 ; r1 = 0
    
    rf2 = k_f_2 * Th_NH3 * Th #good
    rb2 = k_b_2 * Th_NH2 * Th_H ; r2 = rf2-rb2
    
    
    rf4 = k_f_4 * Th_NH * Th
    rb4 = k_b_4 * Th_N * Th_H ; r4 = rf4-rb4
    
    rf5 = k_f_5 * Th_N**2   
    rb5 = k_b_5 * Th_NN  ; r5 = rf5-rb5
    
    rf8 = k_des_8 * Th_H**2 #good
    rb8 = k_ads_8 * Th**2  ; r8 = rf8#-rb8
       
    dydt[0] = 0 # -r1  #P_NH3
    
    #r8 = 0.5*(2*r2+r4)
    r7 = r5
    r6 = r5
    r3 = r2
    
    dydt[1] = r1 - r2          # dθ(NH3)/dt
    dydt[2] = r2 - r3 # = 0    # dθ(NH2)/dt
    dydt[3] = r3 - r4          # dθ(NH)/dt
    dydt[4] = r4 - 2*r5        # dθ(N)/dt
    dydt[5] = r5 - r6 # = 0    # dθ(N-N)/dt
    dydt[6] = r6 - r7 # = 0    # dθ(N2)/dt
    dydt[7] = r2 + r3 + r4 - 2*r8 #0  # dθ(H)/dt
    dydt[8] = 2*r5 + 2*r8 #r6 + r7 + 2*r8 - r1 - r2 - r3 -r4 # 2*r5-r1  # dθ(*)/dt
    
    dydt[9] = r5 #r7  # P_N2
    dydt[10] = r8 # P_H2
    
    dydt[11] = Beta   # Temperature changes with time
    
    return dydt


r = solve_ivp(dydt, (0,t_end), y0, method="BDF", t_eval=np.linspace(0,t_end,3000), args =arguments)#, jac=JACOB, rtol = 1e-5, atol=1e-7)
#  0     1       2     3      4      5      6       7   8    9     10   11
P_NH3, Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_N2, Th_H, Th, P_N2, P_H2, T = r.y
t = r.t



#Calculate rate
k_des_1 = k_des(T, 1e-20, m_NH3, 1, 8.92, arguments[0])/Beta

k_f_2 = k_surf(T, 1e13, arguments[1])/Beta # dissociation of NH3
k_b_2 = k_surf(T, 1e13,  arguments[2])/Beta

k_f_4 = k_surf(T, 1e13, arguments[3])/Beta
k_b_4 = k_surf(T, 1e13,  arguments[4])/Beta   # dissociation of NH

    # formation of N2 Th_N
k_f_5 = k_surf(T, 1e13, arguments[5])/Beta
k_b_5 = k_surf(T, 1e13,  arguments[6])/Beta   # N-N coupling

 #formation of H2
k_des_8 = k_des(T, 1e-20, m_H2, 2, 87.6, 105.17e3)/Beta
k_ads_8 = k_ads(T, P_H2, 1e-20, m_H2)/Beta

        # rate calc
rb1 = k_des_1 * Th_NH3 ; r1 = 0#rb1 #-rf1 #NH3 Ads and Des

rf2 = k_f_2 * Th_NH3 * Th 
rb2 = k_b_2 * Th_NH2 * Th_H ; r2 = rf2-rb2 # NH3* rate


rf4 = k_f_4 * Th_NH * Th
rb4 = k_b_4 * Th_N * Th_H ; r4 = rf4-rb4 # NH* rate

rf5 = k_f_5 * Th_N**2   
rb5 = k_b_5 * Th_NN  ; r5 = rf5-rb5 # N* rate

rf8 = k_des_8 * Th_H**2 
rb8 = k_ads_8 * Th**2  ; r8 = rf8#-rb8

r7 = r5
#r8 = 0.5*(2*r2+r4)

# =============================================================================
# 4/ Plot TPD
# ============================================================================= 
B =str(Beta)
    #Co2Ni
#plt.plot(T,r1, color = "b", ls = '-', label= r"$NH_3$")
plt.plot(T,r7, color = "g", ls = '-', label= r"$N_2$")
plt.plot(T,r8, color = "r", ls = '-', label= r"$H_2$")
    #Co
# plt.plot(T,r1, color = "b", ls = ':', label= r"$Co$")
# plt.plot(T,r7, color = "g", ls = ':')
# plt.plot(T,r8, color = "r", ls = ':')

plt.title(r"TPD of $H_2$ and $N_2$ on $Co_2Ni$"+"\n"+r" at $\beta$= " + B +r" K/s", fontsize=15)
plt.xlabel("Temperature (K)", fontsize=13)
plt.ylabel("Rate", fontsize=14)
plt.legend()
plt.xlim(100,900)
#plt.ylim(-0.01,0.03)
plt.show()
