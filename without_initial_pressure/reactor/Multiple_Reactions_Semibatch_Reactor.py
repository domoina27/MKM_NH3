#%%
# Multiple_Reactions_Semibatch_Reactor

#Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 13, 'lines.linewidth': 2.5})
from matplotlib.widgets import Slider, Button

#%%
#Explicit equations
UA=35000
Ta=298
Ca0=4
vo=240
To=305
V0=100
E1A = 9500
E2B = 7000
CpA=30
CpB=60
CpC=20
CpS=35
dH1A=-6500
dH2B=8000
def ODEfun(Yfuncvec,t,UA,Ta,Ca0,vo,To,V0,E1A,E2B,CpA,CpB,CpC,CpS,dH1A,dH2B):
    Ca= Yfuncvec[0]
    Cb= Yfuncvec[1]
    Cc= Yfuncvec[2]
    T= Yfuncvec[3]
      #Explicit Equation Inline
    k1A = (1.25)* np.exp((E1A/1.987) * (1/320-1/T))
    k2B = (0.08)* np.exp((E2B/1.987) * (1/300-1/T))
    ra = 0 - (k1A * Ca); 
    V = V0 + vo * t; 
    rc = 3 * k2B * Cb; 
    rb = k1A * Ca / 2 - (k2B * Cb);
    Fa0=Ca0*vo
    NH2S04=1*V0
    # Differential equations
    dCadt = ra + (Ca0 - Ca) * vo / V; 
    dCbdt = rb - (Cb * vo / V); 
    dCcdt = rc - (Cc * vo / V); 
    Qg= (dH1A * (ra) + dH2B * (0 - (k2B * Cb))) * V
    Qr=UA*(T-Ta)+(Fa0 *CpA *(T - To)) 
    dTdt = (Qg-Qr) / ((Ca *CpA + Cb *CpB + Cc *CpC) * V + NH2S04 *CpS) 
    return np.array([dCadt, dCbdt, dCcdt, dTdt])

tspan = np.linspace(0, 1.5, 100) # Range for the independent variable
y0 = np.array([1,0,0,290]) # Initial values for the dependent variables

#%%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle("""LEP-13-5: Multiple Reactions in a Semibatch Reactor""", fontweight='bold', x = 0.18, y=0.98)
plt.subplots_adjust(left=0.5)
fig.subplots_adjust(wspace=0.35,hspace=0.3)
sol =  odeint(ODEfun, y0, tspan, (UA,Ta,Ca0,vo,To,V0,E1A,E2B,CpA,CpB,CpC,CpS,dH1A,dH2B))
Ca = sol[:, 0]
Cb= sol[:, 1]
Cc= sol[:, 2]
T= sol[:, 3]
k1A = (1.25)* np.exp((E1A/1.987) * (1/320-1/T))
ra = 0 - (k1A * Ca); 
Rate=-ra
Fa0=Ca0*vo
k2B = (0.08)* np.exp((E2B/1.987) * (1/300-1/T))
V = V0 + vo * tspan
Qg= (dH1A * (ra) + dH2B * (0 - (k2B * Cb))) * V
Qr=UA*(T-Ta)+(Fa0 *CpA *(T - To)) 


p1,p2,p3= ax1.plot(tspan, Ca,tspan, Cb,tspan, Cc)
ax1.legend([r'$C_A$',r'$C_B$',r'$C_C$'], loc='upper right')
ax1.set_xlabel('time $(hr)$', fontsize='medium')
ax1.set_ylabel(r'$C_i$ (mol/$dm^{3}$)', fontsize='medium')
ax1.grid()
ax1.set_ylim(0, 5)
ax1.set_xlim(0, 1.5)

p4 = ax2.plot(tspan, T)[0]
ax2.legend([r'$T$'], loc='upper right')
ax2.set_xlabel('time $(hr)$', fontsize='medium')
ax2.set_ylabel(r'Temperature $(K)$', fontsize='medium')
ax2.grid()
ax2.set_ylim(280, 500)
ax2.set_xlim(0, 1.5)

p5 = ax3.plot(tspan,Rate)[0]
ax3.legend([r'-$r_A$'], loc='upper right')
ax3.set_xlabel('time $(hr)$', fontsize='medium')
ax3.set_ylabel(r'Rate $(mol/dm^{3}.hr)$', fontsize='medium')
ax3.grid()
ax3.set_ylim(0, 35)
ax3.set_xlim(0, 1.5)

p6,p7 = ax4.plot(tspan,Qg,tspan,Qr)
ax4.legend(['$Q_g$', '$Q_r$'], loc='upper left')
ax4.set_xlabel('time $(hr)$', fontsize='medium')
ax4.set_ylabel(r'Q $(cal/hr)$', fontsize='medium')
ax4.grid()
ax4.set_ylim(-2*10**7, 7*10**7)
ax4.set_xlim(0, 1.5)


ax1.text(-4.3,-6.3,'Differential Equations'
         '\n'
         r'$\dfrac{dC_A}{dt} =r_A +\dfrac{(C_{A0}-C_A)}{V} v_0$'
         '\n\n'
         r'$\dfrac{dC_B}{dt} =r_B -\dfrac{C_B}{V} v_0$'
         '\n\n'
         r'$\dfrac{dC_C}{dt} =r_C -\dfrac{C_C}{V} v_0$'
         '\n\n'
         r'$\dfrac{dT}{dt}=\dfrac{(Q_g-Q_r)}{(C_A*C_{P_A}+C_B*C_{P_B}+C_C*C_{P_C})*V+N_{H2S04}*C_{P_{H2S04}}}$'
         '\n\n'
         'Explicit Equations'
         '\n\n'
         r'$Q_g=\{(\Delta H_{Rx1A})(r_{1A})+(\Delta H_{Rx2B})(r_{2B})\}*V$'
         '\n'
         r'$Q_r=UA*(T-T_a)+F_{A0}*C_{P_A}*(T-T_0)+$'
         '\n'
         r'$k_{1A}=(1.25)*exp\left(\left(\dfrac{E_{1A}}{1.987}\right)\left(\dfrac{1}{320} - \dfrac{1}{T}\right)\right)$'
         '\n'
         r'$k_{2B}=(0.08)*exp\left(\left(\dfrac{E_{2B}}{1.987})(\dfrac{1}{300} - \dfrac{1}{T}\right)\right)$'
         '\n'
         r'$r_A=-k_{1A}* C_A$'
         '\n'
         r'$r_B=\dfrac{k_{1A}*C_A}{2}- k_{2B}*C_B$'
         '\n'
         r'$r_C=3* k_{2B}* C_{B}$'
         '\n\n'
         r'$V=V_0+ v_0* t$'
         '\n\n'
         r'$F_{A0}=C_{A0}* v_0 $'
         '\n\n'
         r'$N_{H2S04}=C_{H2S04}* V_0$'
         '\n\n'
          , ha='left', wrap = True, fontsize=13,
        bbox=dict(facecolor='none', edgecolor='black', pad=10.0), fontweight='bold')

#ax1.axis('off')
axcolor = 'black'
ax_UA = plt.axes([0.33, 0.82, 0.1, 0.015], facecolor=axcolor)
ax_Ta = plt.axes([0.33, 0.78, 0.1, 0.015], facecolor=axcolor)
ax_Ca0 = plt.axes([0.33, 0.74, 0.1, 0.015], facecolor=axcolor)
ax_vo = plt.axes([0.33, 0.7, 0.1, 0.015], facecolor=axcolor)
ax_To = plt.axes([0.33, 0.66, 0.1, 0.015], facecolor=axcolor)
ax_V0 = plt.axes([0.33, 0.62, 0.1, 0.015], facecolor=axcolor)
ax_E1A = plt.axes([0.33, 0.58, 0.1, 0.015], facecolor=axcolor)
ax_E2B = plt.axes([0.33, 0.54, 0.1, 0.015], facecolor=axcolor)
ax_dH1A = plt.axes([0.33, 0.50, 0.1, 0.015], facecolor=axcolor)
ax_dH2B = plt.axes([0.33, 0.46, 0.1, 0.015], facecolor=axcolor)

sUA = Slider(ax_UA, r'$UA$($\frac{cal}{h.K}$)', 1000, 50000, valinit=35000,valfmt='%1.0f')
sTa= Slider(ax_Ta, r'$T_{a}$ ($K$)', 275, 325, valinit=298,valfmt='%1.0f')
sCa0 = Slider(ax_Ca0,r'$C_{A0}$($\frac{mol}{dm^3}$)',1, 40, valinit=4,valfmt='%1.1f')
svo = Slider(ax_vo,r'$v_{0}$($\frac{dm^3}{hr}$)', 10, 500, valinit= 240,valfmt='%1.0f')
sTo = Slider(ax_To,r'$T_{0}$($K$)', 280, 350, valinit=305,valfmt='%1.0f')
sV0 = Slider(ax_V0,r'$V_0$ ($dm^3$)', 30, 300, valinit= 100,valfmt='%1.0f')
sE1A = Slider(ax_E1A,r'$E_{1A}$ ($\frac{cal}{mol}$)', 2000, 15000, valinit= 9500,valfmt='%1.0f')
sE2B = Slider(ax_E2B,r'$E_{2B}$ ($\frac{cal}{mol}$)', 2000, 15000, valinit= 7000,valfmt='%1.0f')
sdH1A = Slider(ax_dH1A,r'$\Delta H_{Rx1A}$ ($\frac{cal}{mol A}$)', -20500, -1000, valinit= -6500,valfmt='%1.0f')
sdH2B = Slider(ax_dH2B,r'$\Delta H_{Rx2B}$ ($\frac{cal}{mol B}$)', 1000, 30000, valinit= 8000,valfmt='%1.0f')

def update_plot2(val):
    UA = sUA.val
    Ta =sTa.val
    Ca0 = sCa0.val
    vo =svo.val
    To = sTo.val
    V0 =sV0.val
    E1A = sE1A.val
    E2B =sE2B.val
    dH1A =sdH1A.val
    dH2B =sdH2B.val
    sol = odeint(ODEfun, y0, tspan, (UA,Ta,Ca0,vo,To,V0,E1A,E2B,CpA,CpB,CpC,CpS,dH1A,dH2B))
    Ca = sol[:, 0]
    Cb= sol[:, 1]
    Cc= sol[:, 2]
    T= sol[:, 3]
    k1A = (1.25)* np.exp((E1A/1.987) * (1/320-1/T))
    ra = 0 - (k1A * Ca); 
    Rate=-ra 
    Fa0=Ca0*vo
    k2B = (0.08)* np.exp((E2B/1.987) * (1/300-1/T))
    V = V0 + vo * tspan
    Qg= (dH1A * (ra) + dH2B * (0 - (k2B * Cb))) * V
    Qr=UA*(T-Ta)+ (Fa0 *CpA *(T - To)) 
    p1.set_ydata(Ca)
    p2.set_ydata(Cb)
    p3.set_ydata(Cc)
    p4.set_ydata(T)
    p5.set_ydata(Rate)
    p6.set_ydata(Qg)
    p7.set_ydata(Qr)
    fig.canvas.draw_idle()

sUA.on_changed(update_plot2)
sTa.on_changed(update_plot2)
sCa0.on_changed(update_plot2)
svo.on_changed(update_plot2)
sTo.on_changed(update_plot2)
sV0.on_changed(update_plot2)
sE1A.on_changed(update_plot2)
sE2B.on_changed(update_plot2)
sdH1A.on_changed(update_plot2)
sdH2B.on_changed(update_plot2)

resetax = plt.axes([0.34, 0.86, 0.09, 0.04])
button = Button(resetax, 'Reset variables', color='cornflowerblue', hovercolor='0.975')

def reset(event):
    sUA.reset()
    sTa.reset()
    sCa0.reset()
    svo.reset()
    sTo.reset()
    sV0.reset()
    sE1A.reset()
    sE2B.reset()
    sdH1A.reset()
    sdH2B.reset()
button.on_clicked(reset)
    
