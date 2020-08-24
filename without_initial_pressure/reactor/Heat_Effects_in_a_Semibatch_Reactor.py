#%%

#Heat Effects in a Semibatch Reactor 

#Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 13, 'lines.linewidth': 2.5})
from matplotlib.widgets import Slider, Button

#%%
#Explicit equations
UA=3000
mc=100
Cpc=18
Ta1=285
Vi=0.2
To=300
dH=-7.9076e7
Cw0=55
Cb0=1
v0=0.004
cp=75240
cpa=170700
def ODEfun(Yfuncvec,t,UA,mc,Cpc,Ta1,Vi,To,dH,Cw0,Cb0,v0,cp,cpa):
    Ca= Yfuncvec[0]
    Cb= Yfuncvec[1]
    Cc= Yfuncvec[2]
    T= Yfuncvec[3]
    Nw= Yfuncvec[4]
      #Explicit Equation Inline
    k = 0.39175 *np.exp(5472.7 * (1 / 273 - (1 / T)))
    Kc = 10**(3885.44 / T)
    Cd = Cc
    V = Vi + v0 * t
    Fb0 = Cb0 * v0
    Fw = Cw0 * v0
    ra = 0 - (k * (Ca * Cb - (Cc * Cd / Kc)))
    Na = V * Ca
    Nb = V * Cb
    Nc = V * Cc
    Nd = V * Cd
    rb = ra
    rc = 0 - ra
    rate = 0 - ra
    NCp = cp * (Nb + Nc + Nd + Nw) + cpa * Na
    Qr1=(Fb0 + Fw)* cp*(T - To)
    Qr2 = mc * Cpc * (T - Ta1) * (1 - np.exp(0 - (UA / mc / Cpc)))
    Qr=Qr1+Qr2
    Qg= ra * V * dH
    Ta2 = T - ((T - Ta1) * np.exp(0 - (UA / mc / Cpc)))
    # Differential equations
    dCadt = ra - (v0 * Ca / V) 
    dCbdt = rb + v0 * (Cb0 - Cb) / V 
    dCcdt = rc - (Cc * v0 / V)
    dTdt = (Qg-Qr) / NCp
    dNwdt = v0 * Cw0 
    return np.array([dCadt, dCbdt, dCcdt, dTdt,dNwdt])

tspan = np.linspace(0, 360, 500) # Range for the independent variable
y0 = np.array([5,0,0,300,6.14]) # Initial values for the dependent variables

#%%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle("""LEP-13-4: Heat Effects in a Semibatch Reactor""", fontweight='bold', x = 0.18, y=0.98)
plt.subplots_adjust(left=0.3)
fig.subplots_adjust(wspace=0.25,hspace=0.2)
sol =  odeint(ODEfun, y0, tspan, (UA,mc,Cpc,Ta1,Vi,To,dH,Cw0,Cb0,v0,cp,cpa))
Ca = sol[:, 0]
Cb= sol[:, 1]
Cc= sol[:, 2]
T= sol[:, 3]
Nw=sol[:, 4]
k = 0.39175 *np.exp(5472.7 * (1 / 273 - (1 / T)))
Kc = 10 **(3885.44 / T)
Cd = Cc
ra = 0 - (k * (Ca * Cb - (Cc * Cd / Kc)))
rb = ra
rc = 0 - ra
rate = 0 - ra
Fb0 = Cb0 * v0
Fw = Cw0 * v0
Qr1=(Fb0 + Fw)* cp*(T - To)
Qr2 = mc * Cpc * (T - Ta1) * (1 - np.exp(0 - (UA / mc / Cpc)))
Qr=Qr1+Qr2
V=Vi + v0 * tspan
Qg =ra*(V)*dH
    
Ta2 = T - ((T - Ta1) * np.exp(0 - (UA / mc / Cpc)))
p1,p2,p3= ax2.plot(tspan, Ca,tspan, Cb,tspan, Cc)
ax2.legend([r'$C_A$',r'$C_B$',r'$C_C$'], loc='upper right')
ax2.set_xlabel('time $(sec)$', fontsize='medium')
ax2.set_ylabel(r'$C_i$ (kmol/$m^{3}$)', fontsize='medium')
ax2.grid()
ax2.set_ylim(0, 5)
ax2.set_xlim(0, 360)

p4,p5 = ax3.plot(tspan, T,tspan,Ta2)
ax3.legend([r'$T$',r'$T_{a2}$'], loc='upper right')
ax3.set_xlabel('time $(sec)$', fontsize='medium')
ax3.set_ylabel(r'Temperature $(K)$', fontsize='medium')
ax3.grid()
ax3.set_ylim(280, 350)
ax3.set_xlim(0, 360)

p6,p7 = ax4.plot(tspan,Qg,tspan,Qr)
ax4.legend(['$Q_g$', '$Q_r$'], loc='upper left')
ax4.set_xlabel('time $(sec)$', fontsize='medium')
ax4.set_ylabel(r'Q $(kJ/s)$', fontsize='medium')
ax4.grid()
ax4.set_ylim(0, 8*10**5)
ax4.set_xlim(0, 360)
ax4.ticklabel_format(style='sci',scilimits=(3,4),axis='y')

ax1.text(-1.0,-1.3,'Differential Equations'
         '\n'
         r'$\dfrac{dC_A}{dt} =r_A -\dfrac{(C_A)}{V} v_0$'
         '\n'
         r'$\dfrac{dC_B}{dt} =r_B +\dfrac{(C_{B0}-C_B)}{V} v_0$'
         '\n'
         r'$\dfrac{dC_C}{dt} =r_C -\dfrac{C_C}{V} v_0$'
         '\n'
         r'$\dfrac{dN_W}{dt} =C_{W0} v_0$'
         '\n'
         r'$\dfrac{dT}{dt}=\dfrac{(Q_{gs}-Q_{rs})}{\sum_{i} N_iC_{P_i}}$'
         '\n'
         'Explicit Equations'
         '\n\n'
         r'$N_A=V*C_A$'
         '\n'
         r'$N_B=V*C_B$'
         '\n'
         r'$N_C=V*C_C$'
         '\n'
         r'$N_D=V*C_D$'
         '\n'
         r'$NCp=C_P*(N_B+N_C+N_D+N_W)+C_{P_A}*N_A$'
         '\n'
         r'$k=(0.39175)*exp\left(\left(5472.7\right)\left(\dfrac{1}{273} - \dfrac{1}{T}\right)\right)$'
         '\n'
         r'$K_{C}=10^{(3885.44/T)}$'
         '\n'
         r'$r_A=-k*\left(C_AC_B-\dfrac{C_C*C_D}{K_C}\right)$'
         '\n'
         r'$r_B=r_A$'
         '\n'
         r'$r_C=-r_A$'
         '\n'
         r'$V=V_0+ v_0* t$'
         '\n'
         r'$F_{B0}=C_{B0}* v_0 $'
         '\n'
         r'$F_{W0}=C_{W0}* v_0 $'
         '\n'
         r'$Q_{gs}=(r_A*V)(\Delta H_{Rx})$'
         '\n'
         r'$Q_{rs1}=(F_{B0}*C_{P_B}+F_{W0}*C_{P_W})*(T-T_0) $'
         '\n'
         r'$Q_{rs2}=m_c* C_{P_C}*(T-T_{a1})\left(1-exp\left(\dfrac{-UA}{m_c*C_{P_C}}\right)\right)$'
         '\n'
         r'$Q_{rs}=Q_{rs1}+Q_{rs2} $'
          , ha='left', wrap = True, fontsize=13,
        bbox=dict(facecolor='none', edgecolor='black', pad=10.0), fontweight='bold')

ax1.axis('off')
axcolor = 'black'
ax_UA = plt.axes([0.39, 0.86, 0.15, 0.015], facecolor=axcolor)
ax_mc = plt.axes([0.39, 0.82, 0.15, 0.015], facecolor=axcolor)
ax_CpC = plt.axes([0.39, 0.78, 0.15, 0.015], facecolor=axcolor)
ax_Ta1 = plt.axes([0.39, 0.74, 0.15, 0.015], facecolor=axcolor)
ax_To = plt.axes([0.39, 0.70, 0.15, 0.015], facecolor=axcolor)
ax_dH = plt.axes([0.39, 0.66, 0.15, 0.015], facecolor=axcolor)
ax_Cw0 = plt.axes([0.39, 0.62, 0.15, 0.015], facecolor=axcolor)
ax_Cb0 = plt.axes([0.39, 0.58, 0.15, 0.015], facecolor=axcolor)
ax_v0 = plt.axes([0.39, 0.54, 0.15, 0.015], facecolor=axcolor)

sUA = Slider(ax_UA, r'$UA$($\frac{J}{s.K}$)', 1000, 50000, valinit=3000,valfmt='%1.0f')
smc= Slider(ax_mc, r'$m_{C}$ ($\frac{kg}{s}$)',50, 500, valinit=100,valfmt='%1.0f')
sCpc = Slider(ax_CpC,r'$C_{p_C}$($\frac{J}{mol.K}$)',5, 100, valinit=18,valfmt='%1.1f')
sTa1 = Slider(ax_Ta1,r'$T_{a1}$($K$)', 275, 500, valinit= 285,valfmt='%1.0f')
sTo = Slider(ax_To,r'$T_{0}$($K$)', 275, 500, valinit=300,valfmt='%1.0f')
sdH = Slider(ax_dH,r'$\Delta H_{Rx}$ ($\frac{kJ}{kmol}$)', -9.9076e7, -3.9076e7, valinit= -7.9076e7,valfmt='%1.0E')
sCw0 = Slider(ax_Cw0,r'$C_{W0}$ ($\frac{kmol}{m^3}$)', 5, 200, valinit= 50,valfmt='%1.0f')
sCb0 = Slider(ax_Cb0,r'$C_{B0}$ ($\frac{kmol}{m^3}$)', 0.5, 5, valinit= 1,valfmt='%1.1f')
sv0 = Slider(ax_v0,r'$v_0$ ($\frac{m^3}{s}$)', 0.001, 0.008, valinit= 0.004,valfmt='%1.4f')


def update_plot2(val):
    UA = sUA.val
    mc =smc.val
    Cpc = sCpc.val
    Ta1 =sTa1.val
    To = sTo.val
    dH =sdH.val
    Cw0 = sCw0.val
    Cb0 =sCb0.val
    v0 =sv0.val
    sol = odeint(ODEfun, y0, tspan, (UA,mc,Cpc,Ta1,Vi,To,dH,Cw0,Cb0,v0,cp,cpa))
    Ca = sol[:, 0]
    Cb= sol[:, 1]
    Cc= sol[:, 2]
    T= sol[:, 3]
    Nw=sol[:,4]
    k = 0.39175 *np.exp(5472.7 * (1 / 273 - (1 / T)))
    Kc = 10**(3885.44 / T)
    Cd = Cc
    ra = 0 - (k * (Ca * Cb - (Cc * Cd / Kc)))
    rb = ra
    rc = 0 - ra
    rate = 0 - ra
    Ta2 = T - ((T - Ta1) * np.exp(0 - (UA / mc / Cpc)))
    Fb0 = Cb0 * v0
    Fw = Cw0 * v0
    Qr1=(Fb0 + Fw)* cp*(T - To)
    Qr2 = mc * Cpc * (T - Ta1) * (1 - np.exp(0 - (UA / mc / Cpc)))
    Qr=Qr1+Qr2
    V=Vi + v0 * tspan
    Qg =ra*(V)*dH
    p1.set_ydata(Ca)
    p2.set_ydata(Cb)
    p3.set_ydata(Cc)
    p4.set_ydata(T)
    p5.set_ydata(Ta2)
    p6.set_ydata(Qg)
    p7.set_ydata(Qr)
    fig.canvas.draw_idle()

sUA.on_changed(update_plot2)
smc.on_changed(update_plot2)
sCpc.on_changed(update_plot2)
sTa1.on_changed(update_plot2)
sTo.on_changed(update_plot2)
sdH.on_changed(update_plot2)
sCw0.on_changed(update_plot2)
sCb0.on_changed(update_plot2)
sv0.on_changed(update_plot2)

resetax = plt.axes([0.41, 0.91, 0.09, 0.04])
button = Button(resetax, 'Reset variables', color='cornflowerblue', hovercolor='0.975')

def reset(event):
    sUA.reset()
    smc.reset()
    sCpc.reset()
    sTa1.reset()
    sTo.reset()
    sdH.reset()
    sCw0.reset()
    sCb0.reset()
    sv0.reset()
button.on_clicked(reset)
