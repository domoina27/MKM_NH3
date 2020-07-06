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

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from rate_constant_func import calc_kads as k_ads
from rate_constant_func import calc_kdes as k_des
from rate_constant_func import calc_k_arr as k_surf

# 0/ Constants

    # mass of relevant reactants
m_NH3 = 17.03 * 1.66054e-27
m_N2 = 14 * 1.66054e-27
m_H2 = 2 * 1.66054e-27
m_H = 1.66054e-27

Beta = 2 #Beta  
pt = 1e-5 # Total pressure
T0 = 50 # Initial Temperature

t_end = 400 # Time end
th_NH3_0 = 1
theta = 1-th_NH3_0

# Initial value
P = pt * 1e5
y0 = [P,th_NH3_0 , 0 , 0 , 0 , 0 , 0 , 0 , theta , 0 , 0 , T0]

# 1/ Define function

def dydt(t,y):
    
    dydt = np.zeros(12)
    
    #  0     1       2     3      4      5      6     7     8      9     10   11
    P_NH3, Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_N2, Th_H, Th,  P_N2,  P_H2, T = y 
        
    # calculate all reaction rate constants        
        #Dissociation of NH3
    k_ads_1 = k_ads(T, P_NH3, 1e-20, m_NH3)
    k_des_1 = k_des(T, 1e-20, m_NH3, 1, 8.92, 140e3) #adsorption and desorption of NH3
           
    
    k_f_2 = k_surf(T, 1e13, 110e3)
    k_b_2 = k_surf(T, 1e13,  115e3)  # dissociation of NH3
    
    k_f_3 = k_surf(T, 1e13, 35e3)
    k_b_3 = k_surf(T, 1e13, 67e3)   # dissociation of NH2
    
    k_f_4 = k_surf(T, 1e13, 118e3)
    k_b_4 = k_surf(T, 1e13,  92e3)   # dissociation of NH
    
        #formation of N2
    k_f_5 = k_surf(T, 1e13, 165e3)
    k_b_5 = k_surf(T, 1e13,  104e3)   # N-N coupling
    
    k_f_6 = k_surf(T, 1e13, 23e3)
    k_b_6 = k_surf(T, 1e13, 74e3)   #N2* formation
    
    k_des_7 = k_des(T, 1e-20, m_N2, 2, 2.88, 41.6e3)
    k_ads_7 = k_ads(T, P_N2, 1e-20, m_N2)    #N2 deorption and adsorption
    
        #formation of H2
    k_des_8 = k_des(T, 1e-20, m_H, 2, 87.6, 20e3)
    k_ads_8 = k_ads(T, P_H2, 1e-20, m_H2)
    
    # collect similar terms in new variables    
    rf1 = k_ads_1 * Th #good
    rb1 = k_des_1 * Th_NH3 ; r1 = rf1-rb1
    
    rf2 = k_f_2 * Th_NH3 * Th #good
    rb2 = k_b_2 * Th_NH2 * Th_H ; r2 = rf2-rb2
    
    rf3 = k_f_3 * Th_NH2 * Th
    rb3 = k_b_3 * Th_NH * Th_H ; r3 = rf3-rb3
    
    rf4 = k_f_4 * Th_NH * Th
    rb4 = k_b_4 * Th_N * Th_H ; r4 = rf4-rb4
    
    rf5 = k_f_5 * Th_N**2   
    rb5 = k_b_5 * Th_NN  ; r5 = rf5-rb5
    
    rf6 = k_f_6 * Th_NN
    rb6 = k_b_6 * Th_N2*Th ; r6 = rf6-rb6
    
    rf7 = k_des_7 * Th_N2
    rb7 = k_ads_7 * Th  ; r7 = rf7-rb7
    
    rf8 = k_des_8 * Th_H**2 #good
    rb8 = k_ads_8 * Th**2  ; r8 = rf8-rb8
    
    dydt[0] = -r1  #P_NH3
    
    dydt[1] = r1 - r2    # dθ(NH3)/dt
    dydt[2] = r2 - r3    # dθ(NH2)/dt
    dydt[3] = r3 - r4    # dθ(NH)/dt
    dydt[4] = r4 - 2*r5  # dθ(N)/dt
    dydt[5] = r5 - r6    # dθ(N-N)/dt
    dydt[6] = r6 - r7    # dθ(N2)/dt
    dydt[7] = r2 + r3 + r4 - 2*r8  # dθ(H)/dt
    dydt[8] = r6 + r7 + 2*r8 - r1 - r2 - r3 -r4  # dθ(*)/dt
    
    dydt[9] = r5  # P_N2
    dydt[10] = r8 # P_H2
    
    dydt[11] = Beta   # Temperature changes with time
    
    return dydt

# 2/ Solve the differntial equation

r = solve_ivp(dydt, (0,t_end), y0, method="BDF", t_eval=np.linspace(0,t_end,5000))
P_NH3, Th_NH3, Th_NH2, Th_NH, Th_N, Th_NN, Th_N2, Th_H, Th, P_N2, P_H2, T = r.y 
#  0     1       2     3      4      5      6       7   8    9     10   11
t = r.t

# 3/ Plot result

    # Coverage
# plt.plot(T,Th_NH3, label= "Theta NH3")
# plt.plot(T,Th_NH2, label= "Theta NH2")
# plt.plot(T,Th_NH, label= "Theta NH")
# plt.plot(T,Th_N, label= "Theta N")
# plt.plot(T,Th_NN, label= "Theta N-N")
# plt.plot(T,Th_N2, label= "Theta N2")
# plt.plot(T,Th_H, label= "Theta H")
# plt.plot(T,Th, label= "Theta", ls = "--")

#     # pressure
# plt.plot(T,P_NH3, label= "P NH3")
# plt.plot(T,P_N2, label= "P N2")
# plt.plot(T,P_H2, label= "P H2")

# 4/ Caculate Rate
k_ads_1 = k_ads(T, P_NH3, 1e-20, m_NH3)           # adsorption of NH3
k_des_1 = k_des(T, 1e-20, m_NH3, 1, 8.92, 140e3)

k_f_2 = k_surf(T, 1e13, 110e3) # dissociation of NH3
k_b_2 = k_surf(T, 1e13,  115e3)

k_f_3 = k_surf(T, 1e13, 35e3)
k_b_3 = k_surf(T, 1e13, 67e3)    # dissociation of NH2

k_f_4 = k_surf(T, 1e13, 118e3)
k_b_4 = k_surf(T, 1e13,  92e3)   # dissociation of NH

    # formation of N2 Th_N
k_f_5 = k_surf(T, 1e13, 165e3)
k_b_5 = k_surf(T, 1e13,  104e3)   # N-N coupling

k_f_6 = k_surf(T, 1e13, 23e3)
k_b_6 = k_surf(T, 1e13, 74e3)    # N2* formation

k_des_7 = k_des(T, 1e-20, m_N2, 2, 2.88, 40e3)  # N2 desorption and adsorption
k_ads_7 = k_ads(T, P_N2, 1e-20, m_N2)            # N2 adsorption and adsorption

    # formation of H2
k_des_8 = k_des(T, 1e-20, m_H2, 2, 87.6, 20e3)   # H2 desorption and adsorption
k_ads_8 = k_ads(T, P_H2, 1e-20, m_H2)            # H2 adsorption and adsorption

        # rate calc
rf1 = k_ads_1 * Th
rb1 = k_des_1 * Th_NH3 ; r1 = rb1 #-rf1 #NH3 Ads and Des

rf2 = k_f_2 * Th_NH3 * Th 
rb2 = k_b_2 * Th_NH2 * Th_H ; r2 = rf2-rb2 # NH3* rate

rf3 = k_f_3 * Th_NH2 *Th
rb3 = k_b_3 * Th_NH * Th_H ; r3 = rf3-rb3 # NH2* rate

rf4 = k_f_4 * Th_NH * Th
rb4 = k_b_4 * Th_N * Th_H ; r4 = rf4-rb4 # NH* rate

rf5 = k_f_5 * Th_N**2   
rb5 = k_b_5 * Th_NN  ; r5 = rf5-rb5 # N* rate

rf6 = k_f_6 * Th_NN 
rb6 = k_b_6 * Th_N2 *Th ; r6 = rf6-rb6 # N-N* rate

rf7 = k_des_7 * Th_N2
rb7 = k_ads_7 * Th  ; r7 = r5 #+rb7  #rf7-rb7 # N2 Ads and Des

rf8 = k_des_8 * Th_H**2 
rb8 = k_ads_8 * Th**2  ; r8 = 1/2*(r2 + r3 + r4)   # + rb8#rf8-rb8 # H2 Ads and Des

# dydt[2] = r2 - r3   # Th_NH2
# dydt[3] = r3 - r4   # Th_NH
# dydt[4] = r4 - 2*r5 # Th_N
# dydt[5] = r5 - r6   # Th_N-N
# dydt[6] = r6 - r7   # Th_N2
# dydt[7] = r2 + r3 + r4 - 2*r8 # Th_H
# dydt[8] = r6 + r7 + 2*r8 - r1 - r2 - r3 -r4 # theta

# 5/ Plot rate Vs Temperature

# plt.plot(T,-r1, label= "NH3")
# plt.plot(T,-r1+r2, label= "NH3*")
# plt.plot(T,r2-r3, label= "NH2*")
# plt.plot(T,r3 - r4, label= "NH*")
# plt.plot(T,r4 - 2*r5, label= "N*")
# plt.plot(T,r5 - r6, label= "*NN*")
# plt.plot(T,r2 + r3 + r4 - 2*r8, label= "H*")
# plt.plot(T,r6 + r7 + 2*r8 - r1 - r2 - r3 -r4, label= "*")

# plt.plot(T,r8, label= 1-th_NH3_0)
# plt.plot(T,r8, label= "H2")

# plt.plot(T,rf8, label= "Th**2")

# #------------------------------------
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(T, abs(r1), "b", ls = "-", label= "NH3")
ax1.set_ylabel('NH3')

ax2 = ax1.twinx()
ax2.plot(T, abs(r5), "g", ls = "-", label= "N2")
ax2.set_ylabel('N2', color='g')

ax3 = ax1.twinx()
ax3.plot(T, abs(r8), "r", ls = "-", label= "H2")
ax3.set_ylabel('H2', color='g')

# for tl in ax2.get_yticklabels():
#     tl.set_color('g')

# plt.ylabel('Rate of reaction')
plt.xlabel("Temperature (K)")

plt.xlim(300,1000)
fig.legend(loc="upper right")
plt.show()
