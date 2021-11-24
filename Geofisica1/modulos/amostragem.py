from scipy.interpolate import interp1d
from scipy.signal import resample
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


class SinalAnalogico():
    def __init__(self, y, t):
        self.y = y
        self.t = t


class SinalDigital():
    def __init__(self, y, t):
        self.y = y
        self.t = t
        self.dt = t[1] - t[0]
        self.f_nyquist = 1 / (2 * self.dt)


def gera_sinal_analogico(amplitude, frequencia, t_window):
    tempo = np.linspace(0.0, t_window, 10000, endpoint=True)
    w = 2 * np.pi * frequencia
    y = amplitude * np.cos(w * tempo)
    sinal = SinalAnalogico(y, tempo)
    return sinal
    
    
def digitaliza_sinal(s_analog, f_am):
    dt = 1 / f_am
    t_amostragem = [0.0]
    
    while t_amostragem[-1] < (s_analog.t[-1] - dt):
        t_amostragem.append(t_amostragem[-1] + dt)
    
    #t_window = s_analog.t[-1] - s_analog.t[0]
    #n_amostras = np.int(np.round(t_window / dt, 0)) + 1
    #t_amostragem = np.linspace(s_analog.t[0], s_analog.t[-1], n_amostras, 
    #                           endpoint=True)
    
    # usar o interp1d funciona melhor do que usando o resample
    f_digitaliza = interp1d(s_analog.t, s_analog.y, kind='linear')
    s_dig = f_digitaliza(t_amostragem)
    sinal = SinalDigital(s_dig, t_amostragem)
    return sinal


def plota_sinal(s, ax=None, plt_kwargs={}):

    if ax is None:
        ax = plt.gca()

    ax.plot(s.t, s.y, **plt_kwargs)
    ax.set(xlabel='Tempo (s)', ylabel='Amplitude (ua)')
    ax.legend(loc='upper right')
    ax.set_xlim([s.t[0], s.t[-1]])
    ax.grid(True)


def executa_exercício1(amplitude, frequencia, t_window):
    sinal_analog = gera_sinal_analogico(amplitude, frequencia, t_window)
    
    fig, ax = plt.subplots(figsize=(15, 4))
    plota_sinal(sinal_analog, ax=ax, 
                plt_kwargs={'color': 'blue',
                            'linewidth': 2.0,
                            'label': 'Sinal Analógico'})    
    plt.tight_layout()
    plt.show()
    return None


def executa_exercício2(amplitude, frequencia, t_window, fam):
    sinal_analog = gera_sinal_analogico(amplitude, frequencia, t_window)
    sinal_dig = digitaliza_sinal(sinal_analog, fam)
    
    fig, ax = plt.subplots(figsize=(15, 4))
    plota_sinal(sinal_analog, ax=ax, 
                plt_kwargs={'color': 'blue',
                            'linewidth': 2.0,
                            'label': 'Sinal Analógico'})
    plota_sinal(sinal_dig, ax=ax, 
                plt_kwargs={'markerfacecolor': 'red',
                            'markeredgecolor': 'none',
                            'marker': 'o',
                            'markersize': 8.0,
                            'linestyle': 'none',
                            'label': 'Amplitude Amostrada'})
    plt.tight_layout()
    plt.show()
    
    return None


def executa_exercício3(amplitude, frequencia, t_window, fam):
    sinal_analog = gera_sinal_analogico(amplitude, frequencia, t_window)
    sinal_dig = digitaliza_sinal(sinal_analog, fam)

    fig, ax = plt.subplots(figsize=(15, 4))
    plota_sinal(sinal_analog, ax=ax, 
                plt_kwargs={'color': 'blue',
                            'linewidth': 2.0,
                            'linestyle': '-',
                            'alpha': 0.2,
                            'label': 'Sinal Original'})
    plota_sinal(sinal_dig, ax=ax, 
                plt_kwargs={'markerfacecolor': 'red',
                            'markeredgecolor': 'none',
                            'marker': 'o',
                            'markersize': 8.0,
                            'linestyle': 'none',
                            'label': 'Amplitude Amostrada'})
    plota_sinal(sinal_dig, ax=ax, 
                plt_kwargs={'color': 'red',
                            'linestyle': '-',
                            'linewidth': 2.0,
                            'label': 'Sinal Recuperado'})
    plt.tight_layout()
    plt.show()
    
    return None


def executa_exercício4(amplitude, frequencia, t_window):
    y = []

    for a, f in zip(amplitude, frequencia):
        y.append(gera_sinal_analogico(a, f, t_window))

    sinal_analog = gera_sinal_analogico(0.0, 0.0, t_window)
    sinal_analog.y = y[0].y + y[1].y + y[2].y + y[3].y + y[4].y

    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    cores = ['red', 'black', 'green', 'cyan', 'magenta']
    labels = ['$y_1(t)$', '$y_2(t)$', '$y_3(t)$', '$y_4(t)$', '$y_5(t)$']
    
    for yn, c, l in zip(y, cores, labels):
        plota_sinal(yn, ax=ax[0], 
                    plt_kwargs={'color': c,
                                'linewidth': 2.0,
                                'label': l})
    ax[0].set(xlabel='')
    plota_sinal(sinal_analog, ax=ax[1], 
                plt_kwargs={'color': 'blue',
                            'linewidth': 2.0,
                            'label': 'Sinal Analógico $s(t)$'})    
    plt.tight_layout()
    plt.show()
    return None


def executa_exercício5(amplitude, frequencia, t_window, fam):
    sinal_analog = gera_sinal_analogico(amplitude[0], frequencia[0], t_window)
    
    for a,f in zip(amplitude[1:], frequencia[1:]):
        s = gera_sinal_analogico(a, f, t_window)
        sinal_analog.y = sinal_analog.y + s.y

    sinal_dig = digitaliza_sinal(sinal_analog, fam)

    fig, ax = plt.subplots(figsize=(15, 4))
    plota_sinal(sinal_analog, ax=ax, 
                plt_kwargs={'color': 'blue',
                            'linewidth': 2.0,
                            'linestyle': '-',
                            'alpha': 0.2,
                            'label': 'Sinal Original'})
    plota_sinal(sinal_dig, ax=ax, 
                plt_kwargs={'markerfacecolor': 'red',
                            'markeredgecolor': 'none',
                            'marker': 'o',
                            'markersize': 8.0,
                            'linestyle': 'none',
                            'label': 'Amplitude Amostrada'})
    plota_sinal(sinal_dig, ax=ax, 
                plt_kwargs={'color': 'red',
                            'linestyle': '-',
                            'linewidth': 2.0,
                            'label': 'Sinal Recuperado'})
    plt.tight_layout()
    plt.show()
    
    return None