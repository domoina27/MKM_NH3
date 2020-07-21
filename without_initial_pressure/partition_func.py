import numpy as np

def q_tran(m,T):
    kb = 1.38064852e-23 # boltzmann constant
    h = 6.62607004e-34  # planck constant
    return (2*m*kb*T/h**2)**(3/2)

def q_rot(sigma,A,B,C,T):
    '''
    Parameters
    ----------
    sigma : Symmetry number
    A,B,C : TYPE
    T : TYPE
    '''    
    kb = 1.38064852e-23 # boltzmann constant
    h = 6.62607004e-34  # planck constant    
    return 1/sigma*(kb*T/h)**(3/2)*np.sqrt(np.pi/(A*B*C))

def logq_vib(nu, T):
    log_q= np.log10(1)
    kb = 1.38064852e-23 # boltzmann constant
    h = 6.62607004e-34  # planck constant
    for i in range(len(nu)):
        log_q = log_q + np.log10(1/(1-np.exp(-h*nu[i]/(kb*T))))
    return log_q
