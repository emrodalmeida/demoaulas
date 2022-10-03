import numpy as np
from matplotlib import pyplot as plt


# funções utilizadas pra os cálculos

def gera_onda_quadrada(t_w, T, A):
    """
    Gera a forma de onda quadrada de referência
    """
    
    t_w = np.arange(0, t_w, t_w/1000)
    s_w = A * np.sign(np.sin(2 * np.pi * (1/T) * t_w))
    
    return s_w, t_w
    

def calcula_serie(tt, nh, T, A):
    """
    Calcula o somatório da série de Fourier para a onda quadrada    
    """
    
    ww = 2 * np.pi * (1/T)              # frequência angular
    y = np.zeros(np.shape(tt))          # inicializa com valores de a0, que para esta onda vai ser zero

    for n in range(1, nh+1):
        
        if n%2 != 0:
            bn = (4.0 * A) / (np.pi * n)
        else:
            bn = 0

        y = y + (bn * np.sin(n * ww * tt))
        
    return y
    
    
# funções utilizadas para as figuras
    
def plota_onda_quadrada(tt, sw, T):
    """
    Plota a forma de onda quadrada dentro da janela temporal definida
    """
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(tt, sw, '-b')
    ax.set_title("Figura 1. Onda quadrada de período " + str(T) + " segundos", fontsize=14)
    ax.set_xlim([0, tt[-1]])
    ax.set_ylim([np.min(sw)+0.25*np.min(sw), np.max(sw)+0.25*np.max(sw)])
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.grid()

    
def plota_serie(tt, s_t, sw, nh):
    """
    Plota a representação da onda calculada pela série de Fourier sobreposta à onda quadrada original
    """
    
    if nh==1:
        plural = ''
    else:
        plural = 's'
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(tt, sw, '--b', label='Onda quadrada')
    ax.plot(tt, s_t, '-r', label='$f(t)$ calculada')
    ax.set_title("Figura 2. Comparação entre a onda quadrada e a aproximação" \
                 "pela série de Fourier com " + str(nh) + " harmônico" + plural, fontsize=14)
    ax.set_xlim([0, tt[-1]])
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.grid()
    ax.legend(loc='lower right', fontsize=12)
    