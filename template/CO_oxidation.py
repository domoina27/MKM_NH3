"""
THIS IS A PRACTICE FOR MICRO-KINETIC MODELING OF A BIMOLECULAR REACTION:
 a/ PLOT RATE VS TEMPERATURE
 b/ PLOT COVERAGE VS TEMPERATURE           
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from rate_constant_func import calc_kads 
from rate_constant_func import calc_kdes 
from rate_constant_func import calc_k_arr

ma = 28 * 1.66054e-27
mb = 32 * 1.66054e-27
mc = 80 * 1.66054e-27

# print("What is your temperature?")
#Temperature = float(input())
    
# =============================================================================
# 1/  define the function
# =============================================================================
def dydt(t, y, params):
    
    T, pa, pb, pc= params 
    
    dydt = np.zeros(4)
    
    # calculate all reaction rate constants    
    k_ads_1 = calc_kads(T, pa, 1e-20, ma)
    k_des_1 = calc_kdes(T, 1e-20, ma, 1, 2.8, 80e3)
    k_ads_2 = calc_kads(T, pb, 1e-20, mb)
    k_des_2 = calc_kdes(T, 1e-20, mb, 2, 2.08, 40e3)   
    kf = calc_k_arr(T, 1e13, 120e3)
    kb = calc_k_arr(T, 1e13,  80e3)    
    k_ads_4 = calc_kads(T, pc, 1e-20, mc)
    k_des_4 = calc_kdes(T, 1e-20, mc, 1, 0.561, 10e3)

    # collect similar terms in new variables    
    r1f = k_ads_1 * y[3]
    r1b = k_des_1 * y[0]
    r2f = k_ads_2 * y[3]
    r2b = k_des_2 * y[1]**2   
    r3f = kf * y[0] * y[1]
    r3b = kb * y[2] * y[3]    
    r4f = k_ads_4 * y[3]
    r4b = k_des_4 * y[2]
    
    dydt[0] = r1f - r1b - r3f + r3b
    dydt[1] = 2.0 * r2f - 2.0 * r2b - r3f + r3b
    dydt[2] = r3f - r3b + r4f - r4b
    dydt[3] = -r1f + r1b - 2.0 * r2f + 2.0 * r2b + r3f - r3b - r4f + r4b
        
    return dydt

T_range = np.linspace(300,1000,100)

rate_CO=[]
rate_O2 =[]
rate_CO2 = []

Theta_CO = []
Theta_O = []
Theta_CO2 = []
Theta_vac = []

for T in T_range:   
    
    # =============================================================================
    # 2/Solve the differential equation 
    # =============================================================================
    y0 = [0,0,0,1] #initial condittion
    #T = Temperature    # temperature in K
    pt = 20     # total pressure in bar
    pa = 2.0/3.0 * pt * 1e5      # pressure of CO in Pa
    pb = 1.0/3.0 * pt * 1e5
    pc = 0
    # t = np.linspace(0,1,110)
    r = ode(dydt).set_integrator('vode', method='bdf',  atol=1e-8, rtol=1e-8, 
        nsteps=1000, with_jacobian=True) #Jacobian = true is very important
    r.set_initial_value(y0, 0).set_f_params([T, pa, pb, pc]) #CA_0 and t = 0
    
    t1 = 100
    # xx = np.linspace(-4,1,150)
    xx = np.linspace(-12, np.log10(t1), 500)
    yy = []
    tt = []
    for x in xx:
        tnew = 10.0**x
        tt.append(tnew)
        yy.append(r.integrate(tnew))
    
    Theta = np.matrix(yy)  #yy is a list of array! to make
                      #it easier to plot with legend, better transorm it into a matrix

# =============================================================================
# 3/ Plot results
# =============================================================================

# legend = ["Theta_CO", "Theta_O", "Theta_CO2", "Theta*"]

# for i in range(4):
#     plt.semilogx(tt,Theta[:,i], label= legend[i])
# plt.title("Coverage in function of time at "+str(Temperature)+" K")
# plt.xlabel("time(s)")
# plt.ylabel("coverage Theta")
# plt.legend()
# plt.show()

# =============================================================================
# 4/ Calculate rate in function of Temperature
# =============================================================================
    r_co = calc_kdes(T, 1e-20, ma, 1, 2.8, 80e3) * Theta[-1,0] - calc_kads(T, pa, 1e-20, ma)*Theta[-1,3] # 
    r_o2 = 0.5*(-calc_k_arr(T, 1e13, 120e3)*Theta[-1,0]*Theta[-1,1]+calc_k_arr(T, 1e13,  80e3)*Theta[-1,3]*Theta[-1,2] )
    r_co2 = calc_kdes(T, 1e-20, mc, 1, 0.561, 10e3) * Theta[-1,2] - calc_kads(T, pc, 1e-20, mc)*Theta[-1,3]
    #NEGATIVE INDEX: Theta[-1,:] -1 MEANS THE LAST ELEMENT OF THE MATRIX
    #print(r_co, r_o2, r_co2)
    
    rate_CO.append(r_co)
    rate_O2.append(r_o2)
    rate_CO2.append(r_co2)

    
# =============================================================================
# 5/ Plot rate in function of Temperature
# =============================================================================

# legend = ["CO", "O2", "CO2"]    
# rate = [rate_CO,rate_O2, rate_CO2]
# # plt.plot(T_range, rate[0])
# # plt.plot(T_range, rate[2])
# for i in range(3):
#     plt.plot(T_range, rate[i], label= legend[i])
# plt.title("Coverage in function of Temperature")
# plt.xlabel("Temperature (K)")
# plt.ylabel("Reaction rate")
# plt.legend()


# =============================================================================
# 7/ Expressions of coverage in function of time
# =============================================================================

    k_ads_1 = calc_kads(T, pa, 1e-20, ma)
    k_des_1 = calc_kdes(T, 1e-20, ma, 1, 2.8, 80e3)
    K1 = k_ads_1/k_des_1
    
    k_ads_2 = calc_kads(T, pb, 1e-20, mb)
    k_des_2 = calc_kdes(T, 1e-20, mb, 2, 2.08, 40e3)  
    K2 = k_ads_2/k_des_2
     
    kf = calc_k_arr(T, 1e13, 120e3)
    kb = calc_k_arr(T, 1e13,  80e3)    
    K3 = kf/kb
    
    k_ads_4 = calc_kads(T, pc, 1e-20, mc)
    k_des_4 = calc_kdes(T, 1e-20, mc, 1, 0.561, 10e3)
    K4 = k_des_4/k_ads_4
    
    cov_CO = K1 / (1+K1 + np.sqrt(K2) + K3)
    cov_O = np.sqrt(K2) / (1+K1 + np.sqrt(K2) + K3)
    cov_CO2 = K4 / (1+K1 + np.sqrt(K2) + K3)
    cov_vac = 1 / (1+K1 + np.sqrt(K2) + K3) 
    
    print(cov_CO, cov_O, cov_CO2, cov_vac )
    
    Theta_CO.append(cov_CO)
    Theta_O.append(cov_O)
    Theta_CO2.append(cov_CO2)
    Theta_vac.append(cov_vac)
    
# =============================================================================
# 7/ Plot coverage in function of Temperature
# =============================================================================

legend = ["CO*", "O2*", "CO2*", "*"]    
coverage = [Theta_CO, Theta_O, Theta_CO2, Theta_vac]

# plt.plot(T_range, Theta_CO)
# plt.plot(T_range, Theta_O)
# plt.plot(T_range, Theta_CO2)
# plt.plot(T_range, Theta_vac)

for i in range(4):
    plt.plot(T_range, coverage[i], label= legend[i])
plt.title("Coverage in function of Temperature")
plt.xlabel("Temperature (K)")
plt.ylabel("Coverage")
plt.legend()
