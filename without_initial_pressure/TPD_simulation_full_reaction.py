"""
THIS IS A PRACTICE FOR MICRO-KINETIC MODELING OF NH3 DISSOCIATION AND N2 AND H2 FROMATION:
    REACTION: 
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

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from rate_constant_func import calc_kads as k_ads
from rate_constant_func import calc_kdes as k_des
from rate_constant_func import calc_k_arr as k_surf

m_NH3 = 17.03 * 1.66054e-27
m_N2 = 14 * 1.66054e-27
m_H2 = 2 * 1.66054e-27

# print("What is your temperature?")
#T = Temperature = 700 # float(input())
pt = 1     # total pressure in bar

# =============================================================================
# 1/  define the function
# =============================================================================
def dydt(t, y):   
    
    beta = 5
    
    dydt = np.zeros(12)
    Theta_NH3,Theta_NH2,Theta_NH,Theta_N,Theta_NN,Theta_N2,Theta_H,Theta,T,P_N2,P_H2,P_NH3= y
    # calculate all reaction rate constants        
        #Dissociation of NH3
    k_ads_1 = k_ads(T, P_NH3, 1e-20, m_NH3)           # adsorption of NH3
    k_des_1 = k_des(T, 1e-20, m_NH3, 1, 8.92, 60e3)  # desorption of NH3
           
    # print(k_ads_1,k_des_1)
    
    k_f_2 = k_surf(T, 1e13, 110e3)
    k_b_2 = k_surf(T, 1e13,  115e3)  # dissociation of NH3
    
    k_f_3 = k_surf(T, 1e13, 35e3)
    k_b_3 = k_surf(T, 1e13, 67e3)    # dissociation of NH2
    
    k_f_4 = k_surf(T, 1e13, 100e3)
    k_b_4 = k_surf(T, 1e13,  92e3)   # dissociation of NH
    
        #formation of N2
    k_f_5 = k_surf(T, 1e13, 100e3)
    k_b_5 = k_surf(T, 1e13,  90e3)   # N-N coupling
    
    k_f_6 = k_surf(T, 1e13, 23e3)
    k_b_6 = k_surf(T, 1e13, 74e3)    # N2* formation
    
    k_des_7 = k_des(T, 1e-20, m_N2, 2, 2.88, 16.6e3)  # N2 desorption and adsorption
    k_ads_7 = k_ads(T, P_N2, 1e-20, m_N2)            # N2 adsorption and adsorption
    
        #formation of H2
    k_des_8 = k_des(T, 1e-20, m_H2, 2, 87.6, 84e3)   # H2 desorption and adsorption
    k_ads_8 = k_ads(T, P_H2, 1e-20, m_H2)            # H2 adsorption and adsorption
    
    # print(k_f_5,k_b_5)
    
    # collect similar terms in new variables    
    rf1 = k_ads_1 * Theta #good
    rb1 = k_des_1 * Theta_NH3 ; r1 = rf1-rb1
    
    rf2 = k_f_2 * Theta_NH3 * Theta 
    rb2 = k_b_2 * Theta_NH2 * Theta_H ; r2 = rf2-rb2
    
    rf3 = k_f_3 * Theta_NH2 * Theta
    rb3 = k_b_3 * Theta_NH * Theta_H ; r3 = rf3-rb3
    
    rf4 = k_f_4 * Theta_NH * Theta
    rb4 = k_b_4 * Theta_N * Theta_H ; r4 = rf4-rb4
    
    rf5 = k_f_5 * Theta_N**2   
    rb5 = k_b_5 * Theta_NN  ; r5 = rf5-rb5
    
    rf6 = k_f_6 * Theta_NN
    rb6 = k_b_6 * Theta_N2 *Theta ; r6 = rf6-rb6
    
    rf7 = k_des_7 * Theta_N2
    rb7 = k_ads_7 * Theta  ; r7 = rf7-rb7
    
    rf8 = k_des_8 * Theta_H**2 
    rb8 = k_ads_8 * Theta**2  ; r8 = rf8-rb8
    
    dydt[0] = r1 - r2
    dydt[1] = r2 - r3
    dydt[2] = r3 - r4
    dydt[3] = r4 - 2*r5
    dydt[4] = r5 - r6
    dydt[5] = r6 - r7
    dydt[6] = r2 + r3 + r4 - 2*r8
    dydt[7] = r6 + r7 + 2*r8 - r1 - r2 - r3 -r4
    dydt[8] = beta   # Temperature changes with time
    dydt[9] = r7     # d P(N2)/dt
    dydt[10] = r8    # d P(H2)/dt
    dydt[11] = -r1   # d P(NH3)/dt
    
    return dydt

# =============================================================================
# 2/Solve the differential equation 
# =============================================================================
P_NH3 = pt  * 1e5   # pressure of NH3 in Pa
T_start = 200
t_end = 300
dt = 600
y0 = [0,0,0,0,0,0,0,1,T_start,0,0,P_NH3] #initial condittion    
t_span = np.linspace(0, t_end, dt)
r = solve_ivp(dydt, (0,t_end), y0, t_eval= t_span, method = "BDF")

Theta_NH3,Theta_NH2,Theta_NH,Theta_N,Theta_NN,Theta_N2,Theta_H,Theta,T,P_N2,P_H2,P_NH3= r.y

# =============================================================================
# 3/ Plot results
# =============================================================================

legend = ["Theta_NH3","Theta_NH2","Theta_NH","Theta_N", "Theta_N-N", "Theta_N2","Theta_H", "Theta*","Temperature", "P_N2","P_H2","P_NH3"]

# plt.plot(T,P_NH3, label= legend[11],ls = "-") 
# plt.xlim(0,1)    
#Coverage and Pressure Vs Temp

# plt.plot(T,P_H2, label= legend[0],ls = "-")
# plt.xlim(210,1000)

#-----PLOT Title--------------------------------------------------
#plt.title("Coverage in function of time at "+str(Temperature)+" K and " + str(pt)+" Bar")
#plt.title("Coverage in function of Temperature at " + str(pt)+" Bar")
# plt.title("Temperature in function of time")

#-----PLOT x and y lables--------------------------------------------------
# plt.xlabel("time(s)")
# plt.ylabel("Temperature (K)")
# plt.xlabel("Temperature (K)")
# plt.ylabel("coverage Theta")

# plt.legend()
# plt.show()

# =============================================================================
# Rate vs Temperature
# =============================================================================
k_ads_1 = k_ads(T, P_NH3, 1e-20, m_NH3)           # adsorption of NH3
k_des_1 = k_des(T, 1e-20, m_NH3, 1, 8.92, 140e3)  # desorption of NH3
       
# print(k_ads_1,k_des_1)

k_f_2 = k_surf(T, 1e13, 110e3)
k_b_2 = k_surf(T, 1e13,  115e3)  # dissociation of NH3

k_f_3 = k_surf(T, 1e13, 35e3)
k_b_3 = k_surf(T, 1e13, 67e3)    # dissociation of NH2

k_f_4 = k_surf(T, 1e13, 100e3)
k_b_4 = k_surf(T, 1e13,  92e3)   # dissociation of NH

    #formation of N2
k_f_5 = k_surf(T, 1e13, 100e3)
k_b_5 = k_surf(T, 1e13,  90e3)   # N-N coupling

k_f_6 = k_surf(T, 1e13, 23e3)
k_b_6 = k_surf(T, 1e13, 74e3)    # N2* formation

k_des_7 = k_des(T, 1e-20, m_N2, 2, 2.88, 16.6e3)  # N2 desorption and adsorption
k_ads_7 = k_ads(T, P_N2, 1e-20, m_N2)            # N2 adsorption and adsorption

    #formation of H2
k_des_8 = k_des(T, 1e-20, m_H2, 2, 87.6, 84e3)   # H2 desorption and adsorption
k_ads_8 = k_ads(T, P_H2, 1e-20, m_H2)            # H2 adsorption and adsorption
    
    # collect similar terms in new variables    
rf1 = k_ads_1 * Theta #good
rb1 = k_des_1 * Theta_NH3 ; r1 = rf1-rb1

rf2 = k_f_2 * Theta_NH3 * Theta 
rb2 = k_b_2 * Theta_NH2 * Theta_H ; r2 = rf2-rb2

rf3 = k_f_3 * Theta_NH2 * Theta
rb3 = k_b_3 * Theta_NH * Theta_H ; r3 = rf3-rb3

rf4 = k_f_4 * Theta_NH * Theta
rb4 = k_b_4 * Theta_N * Theta_H ; r4 = rf4-rb4

rf5 = k_f_5 * Theta_N**2   
rb5 = k_b_5 * Theta_NN  ; r5 = rf5-rb5

rf6 = k_f_6 * Theta_NN
rb6 = k_b_6 * Theta_N2 *Theta ; r6 = rf6-rb6

rf7 = k_des_7 * Theta_N2
rb7 = k_ads_7 * Theta  ; r7 = rf7-rb7

rf8 = k_des_8 * Theta_H**2 
rb8 = k_ads_8 * Theta**2  ; r8 = rf8-rb8


rate = np.zeros(dt)
r_NH3 = np.zeros(dt)
r_N2 = np.zeros(dt)
r_H2 = np.zeros(dt)

# for i in range(dt):
rdes_NH3= rb1 #k_des_1 * Theta_NH3  # desorption of NH3
rdes_N2 = rf7  #k_f_5 *Theta_N**2 - k_b_5*Theta_NN  #k_des_7 * Theta_N2 #+ k_ads(T, P_N2, 1e-20, m_N2)*Theta
rdes_H2 = rf8  # 0.5*(r2+r3+r4)#k_des_8 * Theta_H**2


plt.plot(T, rdes_NH3, "b", ls = "-", label= "NH3") 
plt.plot(T, rdes_N2, "green", label= "N2")
plt.plot(T, rdes_H2 , "r", label= "H2") 
# plt.show()
# plt.savefig("rate_NH3 Vs Temp.png")
  
# plt.xlim(600,800) 
plt.xlim(200,1100) 
  
# plt.ylabel('Rate of reaction')
# plt.xlabel("Temperature (K)")
# plt.legend(loc='best')

# #-----PLOT Title--------------------------------------------------
# plt.title("Rate of adsorption of NH3 and desorption \nof N2 and H2 Vs Temperature at " + str(pt)+" Bar")

# =============================================================================
# Rate constants k Vs Temperature
# =============================================================================
# kf = np.zeros(dt)
# kb = np.zeros(dt)

# for i in range(dt):
#     kf[i] = k_surf(Theta[i,8], 1e13, 100e3)
#     kb[i] = k_surf(Theta[i,8], 1e13,90e3)

# plt.plot(1/(Theta[:,8]), np.log(kf), "c", ls="--", label="k5 forward")
# # plt.plot(1/(Theta[:,8]), np.log(kb),"b", label="k5 backward")

# # plt.title("Rate constant of RLS Vs time ")
# plt.title("Log(rate constant) Vs 1/T of RLS\n N* +N* = *N-N* (reaction 5)")

# # plt.ylabel('Rate constant')
# plt.ylabel('log(k)')

# # plt.xlabel("time (s)")
# plt.xlabel("1/T (K-1)")
plt.legend(loc='best')
