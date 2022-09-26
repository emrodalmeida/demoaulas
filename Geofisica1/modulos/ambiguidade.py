import numpy as np
import matplotlib.pyplot as plt


def fwd(t, g, s0=0, v0=0):
    return s0 + v0*t + 0.5*g*t**2


def fwd_esfera(rho, rho_bg, R, z, x):
    return 27.9e-3 * (rho - rho_bg) * R**3 *  z / ((x**2 + z**2)**(3/2))
    

def d_observ(tt, d_obs):
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(tt, d_obs, 'ob')
    ax.invert_yaxis()
    ax.set_title('Dados observados', fontsize=14)
    ax.set_xlabel('Tempo (s)', fontsize=14)
    ax.set_ylabel('Posição (m)', fontsize=14)
    ax.grid(which='both')
    plt.show()


def d_observ_bf(tt, d_obs):
    
    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(tt, d_obs, 'ob')
    ax.invert_yaxis()
    ax.set_title('Dados observados', fontsize=14)
    ax.set_xlabel('Distância (m)', fontsize=14)
    ax.set_ylabel('Medida (U)', fontsize=14)
    ax.grid(which='both')
    ax.invert_yaxis()
    plt.show()
    

def ajuste_bf(xx, d_pred, d_obs):

    err_rms_pc = np.round(np.sqrt(np.mean(((d_obs - d_pred) / d_obs)**2, axis=0)) * 100, 2)
    
    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(xx, d_obs, 'ob', label='dados observados')
    ax.plot(xx, d_pred, '-r', label='dados preditos')
    ax.set_title('Erro RMS = '+ str(err_rms_pc) + ' %', fontsize=14)
    ax.set_xlabel('Distância (m)', fontsize=14)
    ax.set_ylabel('Medida (U)', fontsize=14)
    ax.grid(which='both')
    ax.legend()
    plt.show()
    
    
    
    
def ajuste(tt, d_pred, d_obs):

    epsilon = 1e-6  # para evitar divisão por zero
    
    err_rms_pc = np.round(np.sqrt(np.mean((((d_obs+epsilon) - (d_pred+epsilon)) / (d_obs+epsilon))**2, axis=0)) * 100, 2)
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(tt, d_obs, 'ob', label='dados observados')
    ax.plot(tt, d_pred, '-r', label='dados preditos')
    ax.invert_yaxis()
    ax.set_title('Erro RMS = '+ str(err_rms_pc) + ' %', fontsize=14)
    ax.set_xlabel('Tempo (s)', fontsize=14)
    ax.set_ylabel('Posição (m)', fontsize=14)
    ax.grid(which='both')
    ax.legend()
    plt.show()
    
t = np.linspace(0, 5, 11)
s_o = np.array(list(map(lambda x: 0.5*9.8*x**2, t)))

s_o2 = np.array([[0.000000000000000000e+00, 1.653999999999999915e+00],
                 [5.000000000000000000e-01, 3.487249999999999961e+00],
                 [1.000000000000000000e+00, 6.247999999999999332e+00],
                 [1.500000000000000000e+00, 9.936249999999999361e+00],
                 [2.000000000000000000e+00, 1.455199999999999960e+01],
                 [2.500000000000000000e+00, 2.009525000000000006e+01],
                 [3.000000000000000000e+00, 2.656599999999999895e+01],
                 [3.500000000000000000e+00, 3.396424999999999983e+01],
                 [4.000000000000000000e+00, 4.228999999999999915e+01],
                 [4.500000000000000000e+00, 5.154325000000000045e+01],
                 [5.000000000000000000e+00, 6.172400000000000375e+01]])

bf_obs = np.array([3.03572027e-05, 3.42160388e-05, 3.87527057e-05, 4.41215728e-05,
                   5.05207954e-05, 5.82073599e-05, 6.75182668e-05, 7.89007722e-05,
                   9.29562188e-05, 1.10504592e-04, 1.32681254e-04, 1.61084576e-04,
                   1.98005689e-04, 2.46793366e-04, 3.12445717e-04, 4.02589480e-04,
                   5.29130633e-04, 7.11071166e-04, 9.79309520e-04, 1.38454779e-03,
                   2.00880000e-03, 2.97535720e-03, 4.42809752e-03, 6.38866273e-03,
                   8.38940503e-03, 9.30000000e-03, 8.38940503e-03, 6.38866273e-03,
                   4.42809752e-03, 2.97535720e-03, 2.00880000e-03, 1.38454779e-03,
                   9.79309520e-04, 7.11071166e-04, 5.29130633e-04, 4.02589480e-04,
                   3.12445717e-04, 2.46793366e-04, 1.98005689e-04, 1.61084576e-04,
                   1.32681254e-04, 1.10504592e-04, 9.29562188e-05, 7.89007722e-05,
                   6.75182668e-05, 5.82073599e-05, 5.05207954e-05, 4.41215728e-05,
                   3.87527057e-05, 3.42160388e-05, 3.03572027e-05])

x = np.linspace(-200, 200, 51)