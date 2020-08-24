#%%

# 1b -  Batch Reactor with Heat Exchange

#Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 13, 'lines.linewidth': 2.5})
from matplotlib.widgets import Slider, Button

#%%
#Explicit equations
Ea = 18000
dH = -20202
CpA=35
CpB=18
CpM=19.5
NA0=54.8
NB0=555
NM0=98.8
R=1.987
To=286
UA=10
Cpc=4.16
mc=10
Ta1=290
def ODEfun(Yfuncvec,t,Ea,dH,CpA,CpB,CpM,NA0,NB0,NM0,R,To,UA,Cpc,mc,Ta1):
    X= Yfuncvec[0]
    T= Yfuncvec[1]
      #Explicit Equation Inline
    CpS=CpA+CpB*(NB0/NA0)+CpM*(NM0/NA0)
    k = (0.000273)* np.exp((Ea/R) * (1/297-1/T))
    Qr = mc*Cpc*(T-Ta1)*(1-np.exp(-UA/mc/Cpc))
    Qg =NA0* k*(1-X)*(-dH); 
    # Differential equations
    dXdt = k*(1-X)
    dTdt = (Qg-Qr)/CpS/NA0;
    return np.array([dXdt,dTdt])

tspan = np.linspace(0, 4000, 1000) # Range for the independent variable
y0 = np.array([0,To]) # Initial values for the dependent variables

#%%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle("""LEP-13-1b: Batch Reactor (With Heat Exchange)""", fontweight='bold', x = 0.18, y=0.98)
plt.subplots_adjust(left=0.3)
fig.subplots_adjust(wspace=0.2,hspace=0.2)
sol =  odeint(ODEfun, y0, tspan, (Ea,dH,CpA,CpB,CpM,NA0,NB0,NM0,R,To,UA,Cpc,mc,Ta1))
X = sol[:, 0]
T = sol[:, 1]
CpS=CpA+CpB*(NB0/NA0)+CpM*(NM0/NA0)
k = (0.000273)* np.exp((Ea/R) *(1/297-1/T))
Ta2=T-(T-Ta1)*np.exp(-UA/(mc*Cpc))
Qr = mc*Cpc*(T-Ta1)*(1-np.exp(-UA/mc/Cpc))
Qg =NA0* k*(1-X)*(-dH); 
p1= ax2.plot(tspan, X)[0]
ax2.legend([r'$X$'], loc='upper right')
ax2.set_xlabel('time $(sec)$', fontsize='medium')
ax2.set_ylabel(r'Conversion', fontsize='medium')
ax2.grid()
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 4000)
p2,p3 = ax3.plot(tspan, T,tspan,Ta2)
ax3.legend([r'$T$',r'$T_{a2}$'], loc='upper right')
ax3.set_xlabel('time $(sec)$', fontsize='medium')
ax3.set_ylabel(r'Temperature $(K)$', fontsize='medium')
ax3.grid()
ax3.set_ylim(280, 400)
ax3.set_xlim(0, 4000)
p4,p5 = ax4.plot(tspan, Qg,tspan,Qr)
ax4.legend(['$Q_g$', '$Q_r$'], loc='upper right')
ax4.set_xlabel('time $(sec)$', fontsize='medium')
ax4.set_ylabel(r'Q $(J/s)$', fontsize='medium')
ax4.grid()
ax4.set_ylim(0, 2800)
ax4.set_xlim(0, 4000)
ax4.text(-8800, 2000,'Differential Equations'
         '\n'
         r'$\dfrac{dX}{dt} = k*(1-X)$'
         '\n'
         r'$\dfrac{dT}{dt} = \dfrac{(Q_{gb}-Q_{rb})}{N_{A0} C_{P_S}}$'
         '\n\n'
         'Explicit Equations'
         '\n'
         r'$\theta_B=\dfrac{N_{B_0}}{N_{A_0}}$'
         '\n'
         r'$\theta_M=\dfrac{N_{M_0}}{N_{A_0}}$'
         '\n'
         r'$C_{P_S} = C_{P_A} +\theta_BC_{P_B}+\theta_MC_{P_M}$'
                  '\n\n'
         r'$k=(2.73*10^{-4})*exp\left(\left(\dfrac{E}{1.987}\right)\left(\dfrac{1}{297} - \dfrac{1}{T}\right)\right)$'
         '\n'
         r'$T_{a2}=T-(T-T_{a1})*exp\left(\dfrac{-UA}{m_c* C_{P_C}}\right)$'
         '\n'
         r'$Q_{rb}=m_c* C_{P_C}\{\left(T-T_{a1}\right)*\left[1-exp\left(\dfrac{-UA}{m_c* C_{P_C}}\right)\right]\} $'
         '\n'
          r'$Q_{gb}=N_{A0}*k*(1-X) *(-\Delta H_{Rx})$'
          '\n'
          , ha='left', wrap = True, fontsize=13,
        bbox=dict(facecolor='none', edgecolor='black', pad=10.0), fontweight='bold')

ax1.axis('off')
axcolor = 'black'
ax_Ea = plt.axes([0.35, 0.86, 0.15, 0.015], facecolor=axcolor)
ax_NA0 = plt.axes([0.35, 0.82, 0.15, 0.015], facecolor=axcolor)
ax_UA = plt.axes([0.35, 0.78, 0.15, 0.015], facecolor=axcolor)
ax_mc = plt.axes([0.35, 0.74, 0.15, 0.015], facecolor=axcolor)
ax_Ta1 = plt.axes([0.35, 0.70, 0.15, 0.015], facecolor=axcolor)

sEa = Slider(ax_Ea, r'$E$($\frac{cal}{mol}$)', 10000, 30000, valinit=18000,valfmt='%1.0f')
sNA0 = Slider(ax_NA0,r'$N_{A0}$($mol$)',10, 200, valinit=54.8,valfmt='%1.1f')
sUA = Slider(ax_UA,r'$UA$ ($\frac{cal}{K.s}$)', 1, 20, valinit= 10,valfmt='%1.1f')
smc = Slider(ax_mc,r'$m_c$ ($\frac{g}{s}$)', 1, 30, valinit= 10,valfmt='%1.1f')
sTa1 = Slider(ax_Ta1,r'$T_{a1}$ ($K$)', 280, 400, valinit= 290,valfmt='%1.0f')

def update_plot2(val):
    Ea = sEa.val
    NA0 = sNA0.val
    UA =sUA.val
    mc =smc.val
    Ta1 =sTa1.val
    y0 = np.array([0,To])
    sol = odeint(ODEfun, y0, tspan, (Ea,dH,CpA,CpB,CpM,NA0,NB0,NM0,R,To,UA,Cpc,mc,Ta1))
    X = sol[:, 0]
    T = sol[:, 1]
    k = (0.000273)* np.exp((Ea/R) *(1/297-1/T))
    Ta2=T-(T-Ta1)*np.exp(-UA/(mc*Cpc))
    Qr = mc*Cpc*(T-Ta1)*(1-np.exp(-UA/mc/Cpc))
    Qg =NA0* k*(1-X)*(-dH);  
    p1.set_ydata(X)
    p2.set_ydata(T)
    p3.set_ydata(Ta2)
    p4.set_ydata(Qg)
    p5.set_ydata(Qr)
    fig.canvas.draw_idle()

sEa.on_changed(update_plot2)
sNA0.on_changed(update_plot2)
sUA.on_changed(update_plot2)
smc.on_changed(update_plot2)
sTa1.on_changed(update_plot2)

resetax = plt.axes([0.37, 0.91, 0.09, 0.04])
button = Button(resetax, 'Reset variables', color='cornflowerblue', hovercolor='0.975')

def reset(event):
    sEa.reset()
    sNA0.reset()
    sUA.reset()
    smc.reset()
    sTa1.reset()
button.on_clicked(reset)
    
