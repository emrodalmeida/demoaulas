import numpy as np
import matplotlib.pyplot as plt

FONTE_PEQUENA = 12
FONTE_MEDIA = 14
FONTE_GRANDE = 14

plt.rc('font', size=FONTE_PEQUENA)
plt.rc('axes', titlesize=FONTE_GRANDE)
plt.rc('axes', labelsize=FONTE_MEDIA)
plt.rc('legend', fontsize=FONTE_PEQUENA)
plt.rc('xtick', labelsize=FONTE_PEQUENA)
plt.rc('ytick', labelsize=FONTE_PEQUENA)


def fwd_queda_livre(t, g, s0, v0):
    s = s0 + v0*t + 0.5*g*t**2
    return s
    
    
def fwd_qlqr(rho, rho_bg, r, z, x):

    # Calculado segundo a equação 2.50 do Telford et al. (1990)
    k = 27.9e-3
    delta_rho = rho - rho_bg
    
    # evita divisão por zero quando z=0.0
    e = 1e-9
    
    # anomalia em mGal
    ug = k * delta_rho * r**3 *  z / (e + (x**2 + z**2)**(3/2))
    return ug


def calcula_erro_rms(do, dp):
    diferenca = np.abs((do) - (dp))
    media_quadratica = np.mean(diferenca / (do)**2, axis=0)
    err_rms_percent = np.round(np.sqrt(media_quadratica) * 100, 1)
    return err_rms_percent


def plota_ajuste(dominio, d_obs, d_pred, ax=None, label_kwargs={}):
    
    if ax is None:
        ax = plt.gca()        
    
    ax.plot(dominio, d_obs, 'ob', label='Dados observados')
    ax.plot(dominio, d_pred, '.-r', label='Dados preditos')
    ax.set(**label_kwargs)
    ax.grid(True)
    ax.set_xlim([dominio[0], dominio[-1]])
    ax.legend()
    return ax


def plota_modelo(x, rho, rho_bg, z, r, ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    ax.add_patch(plt.Rectangle((x[0], 0.0), x[-1] - x[0], z+2*r, 
                 edgecolor='white', facecolor='yellow'))
    ax.add_patch(plt.Circle((0, z), r, color='r'))
    ax.plot([x[0], x[-1]], [0.0, 0.0], 'k', linewidth=5.0)
    ax.text(x[0]+10.0, z/3, r'$\rho_{re}$ = ' + str(rho_bg) + r' g/cm$^3$')
    ax.text(r+5.0, z, r'$\rho_{c}$ = ' + str(rho) + r' g/cm$^3$')
    ax.set_xlabel('Distância (m)')
    ax.set_ylabel('Profundidade (m)')
    ax.set_title('Modelo')
    ax.set_ylim([-1.0, z+2*r])
    ax.set_xlim([x[0], x[-1]])
    ax.grid(True)
    ax.invert_yaxis()
    
    return ax


def executa_exercicio3(a, s0, v0):
    dados = np.loadtxt('dados/queda_livre.dat')
    tempo = dados[:,0]
    d_obs = dados[:,1]
    d_pred = fwd_queda_livre(tempo, a, s0, v0)
    erro_rms = calcula_erro_rms(d_obs, d_pred)
    labels = {'xlabel': 'Tempo (s)',
              'ylabel': 'Posição (m)',
              'title': 'Erro RMS = '+ str(erro_rms) + ' %'}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plota_ajuste(tempo, d_obs, d_pred, ax=ax, label_kwargs=labels)
    ax.invert_yaxis()
    plt.show()
    
    return None


def executa_exercicio4(rho_c, rho_re, z, r):
    dados = np.loadtxt('dados/qlqr.dat')
    distancia = dados[:,0]
    d_obs = dados[:,1] * 1e3
    d_pred = fwd_qlqr(rho_c, rho_re, r, z, distancia) * 1e3
    erro_rms = calcula_erro_rms(d_obs, d_pred)
    labels = {'xlabel': 'Distância (m)',
              'ylabel': 'Anomalia (UG)',
              'title': 'Erro RMS = '+ str(erro_rms) + ' %'}
    
    fig, ax = plt.subplots(2, 1, figsize=(14, 7))
    plota_ajuste(distancia, d_obs, d_pred, ax=ax[0], label_kwargs=labels)
    plota_modelo(distancia, rho_c, rho_re, z, r, ax=ax[1])
    plt.tight_layout()
    plt.show()
    return None
