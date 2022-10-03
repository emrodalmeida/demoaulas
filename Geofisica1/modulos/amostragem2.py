from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt


# funções para os cálculos

def gera_funcao_ref(a, f, t_max):
    """
    Gera uma função obtida a partir de uma sobreposição de funções cosseno, onde cada uma delas é caracterizada por
    uma amplitude A e frequência f na forma s(t) = A * cos(2 * pi * f * t). A sobreposição destas funções será o sinal
    analógico de referência. Esta função pode ser amostrada em qualquer instante de tempo que se queira, de forma 
    que esta é a melhor forma de representar um sinal contínuo para os objetivos desta demonstração.
    """

    dt = t_max/1000
    tt = np.arange(-t_max, (2*t_max) + dt, dt)                        # eixo de tempo estendido
    
    s = np.zeros(np.shape(tt))
    
    for i in range(len(a)):
        s = s + (a[i] * np.cos(2 * np.pi * f[i] * tt))
        
    return interp1d(tt, s)      # função que caracteriza o sinal analógico
    

def gera_sinal_analogico(t_max, f_sinal):
    """
    Gera a representação do sinal analógico. O que faz na verdade é calcular as amplitudes da função cosseno com inervalo
    de amostragem curto o suficiente para que ela possa ser visualizada como um sinal analógico contínuo.
    """
    
    dt = t_max/1000
    tt = np.arange(-t_max, (2*t_max), dt)

    return f_sinal(tt), tt
    
    
def amostra_sinal(t_max, dt, f_sinal):
    """
    Faz a amostragem do sinal calculando as amplitudes da função cosseno de acordo com o intervalo
    de amostragem definido.
    """
    
    t_amostrado = np.arange(-t_max, t_max*2, dt)
    sinal_amostrado = f_sinal(t_amostrado)      # representação aproximada do sinal "analógico"
    
    return sinal_amostrado, t_amostrado


def recupera_sinal(t_in, s_in):
    """
    Interpola as amplitudes que foram amostradas da função cosseno para demonstrar como seria o
    comportamento real do sinal recuperado a partir destas amostras.
    """
    
    s_out = interp1d(t_in, s_in, kind='cubic')
    
    t_max = np.max(t_in)
    dt = t_max/1000
    tt = np.arange(-t_max, (2*t_max) + dt, dt)
    i1 = np.abs(tt - t_in[0]).argmin()
    i2 = np.abs(tt - t_in[-1]).argmin()

    return s_out(tt[i1+1:i2]), tt[i1+1:i2]
    

# funções de figuras

def plota_amostragem(tt1, s1, tt2, s2, t_w, titulo):
    """
    Plota as amplitudes amostradas em relação ao sinal analógico original
    """
        
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(tt1, s1, '-r', label='Sinal original')
    ax.plot(tt2, s2, '.b', label='Amplitudes amostradas')
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title(titulo, fontsize=14)
    ax.set_xlim([0, t_w])
    ax.grid()
    ax.legend(loc='lower right', fontsize=12)

    
def plota_sinal(tt, a, t_w, titulo):
    """
    Plota a função cosseno calculada a intervalos de tempo pequenos o suficiente para que se possa 
    fazer uma representação do sinal analógico original.
    """
    
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(tt, a)
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title(titulo, fontsize=14)
    ax.set_xlim([0, t_w])
    ax.grid()
    
    
def plota_representacao(tt_am, s_am, tt_rec, s_rec, t_w, titulo):
    """
    Plota a interpolação do sinal feita a partir das amostras obtidas do sinal analógico.
    """
    
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(tt_rec, s_rec, '--b', label='Sinal recuperado', linewidth=1)
    ax.plot(tt_am, s_am, '.b', label='Amplitudes amostradas')
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title(titulo, fontsize=14)
    ax.set_xlim([0, t_w])
    ax.grid()
    ax.legend(loc='lower right', fontsize=12)
    
    
def plota_comparacao(tt1, s1, tt2, s2, t_w, titulo):
    """
    Plota a interpolação do sinal feita a partir das amostras obtidas do sinal analógico e a sobrepõe à 
    representação do sinal analógico original.
    """
    
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(tt1, s1, '-r', label='Sinal original')
    ax.plot(tt2, s2, '--b', label='Sinal recuperado')
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title(titulo, fontsize=14)
    ax.set_xlim([0, t_w])
    ax.grid()
    ax.legend(loc='lower right', fontsize=12)