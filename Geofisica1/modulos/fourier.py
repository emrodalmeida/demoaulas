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


import numpy as np
from matplotlib import pyplot as plt


def gera_onda_quadrada(A, T, t_w):
    t = np.arange(0, t_w, t_w/1000)
    s = A * np.sign(np.sin(2 * np.pi * (1/T) * t))
    return s, t
    

def calcula_serie(A, tt, T, nh):
    ww = 2 * np.pi * (1/T)
    
    # inicializa com valores de a0, que para esta onda vai ser zero
    y = np.zeros(np.shape(tt))          

    for n in range(1, nh+1):
        
        if n%2 != 0:
            bn = (4.0 * A) / (np.pi * n)
        else:
            bn = 0

        y = y + (bn * np.sin(n * ww * tt))

    return y
    
      
def plota_onda(tt, sw, ax=None, plt_kwargs={}):

    if ax is None:
        ax = plt.gca()

    ax.plot(tt, sw, **plt_kwargs)
    ax.set_xlim([0, tt[-1]])
    ax.set_ylim([np.min(sw)+0.25*np.min(sw),
                 np.max(sw)+0.25*np.max(sw)])
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Amplitude (ua)')
    ax.legend(loc='lower right')
    ax.grid(True)
    return ax


def executa_exercicio1(a, T, tw):
    onda_q, tempo = gera_onda_quadrada(a, T, tw)
        
    fig, ax = plt.subplots(figsize=(15,5))
    label_onda = 'Onda quadrada'
    plota_onda(tempo, onda_q, ax=ax, plt_kwargs={'label': label_onda,
                                                 'color': 'blue',
                                                 'linewidth': 2.0})
    plt.tight_layout()
    plt.show()
    
    return None
    

def executa_exercicio2(a, T, tw, n):
    onda_q, tempo = gera_onda_quadrada(a, T, tw)
    onda_aprox = calcula_serie(a, tempo, T, n)
    
    fig, ax = plt.subplots(figsize=(15,5))
    label_onda = 'Onda quadrada\nde referência'
    plota_onda(tempo, onda_q, ax=ax, plt_kwargs={'label': label_onda,
                                                 'color': 'blue',
                                                 'linestyle': '--',
                                                 'alpha': 0.75,
                                                 'linewidth': 1.5})
    label_aprox = 'Aproximação por\n' + str(n) + ' harmônicos'
    plota_onda(tempo, onda_aprox, ax=ax, plt_kwargs={'label': label_aprox,
                                                     'color': 'red',
                                                     'linestyle': '-',
                                                     'linewidth': 2.0})
    plt.tight_layout()
    plt.show()
    
    return None