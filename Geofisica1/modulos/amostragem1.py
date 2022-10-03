from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt


class sinal_monofreq():
    
    """
    Sinal composto por uma única frequência e caracterizado por uma função cosseno na forma 
    y(t) = A * cos(2 * pi * F * t)
    """
    
    def __init__(self, a, f, tw):
        self.amplitude = a
        self.frequencia = f
        self.janela_tempo = tw
        self.dt = tw/1000
        
        # eixo de tempo estendido, apenas para evitar artefatos nas extremidades das interpolações
        self.tt_analogico = np.arange(-self.janela_tempo, (2*self.janela_tempo) + self.dt, self.dt)
        
        # Função de referência que caracteriza o sinal analógico na forma y(t) = A * cos(2 * pi * F * t).
        # Pode ser amostrada em qualquer instante de tempo que se queira, de forma que esta é a melhor
        # forma de representar um sinal contínuo para os objetivos desta demonstração.
        self.funcao_cos = interp1d(self.tt_analogico, self.amplitude * \
                                   np.cos(2*np.pi*self.frequencia*self.tt_analogico), kind='linear')      
        
        # Gera uma aproximação do sinal analógico calculando as amplitudes da função cosseno com inervalo
        # de amostragem curto o suficiente para que ela possa ser visualizada como um sinal analógico contínuo.
        self.analogico = self.funcao_cos(self.tt_analogico)
        
        # Inicializa com valores nulos pois não foi feita a amostragem ainda
        self.dt_amostrado = None
        self.tt_amostrado = None
        self.amostrado = None
        
        # O sinal recuperado usa o mesmo dt do sinal analógico original, mas aqui inicializa com valor nulo.
        self.recuperado = None
        self.tt_recuperado = None

   
    def amostragem(self, f_am):
        """
        Faz a amostragem do sinal calculando as amplitudes da função cosseno de acordo com o intervalo
        de amostragem definido.
        """
        
        t_min = self.tt_analogico[0]
        t_max = self.tt_analogico[-1]
        self.dt_amostrado = 1/f_am
        self.tt_amostrado = np.arange(t_min, t_max, self.dt_amostrado)

        # amostragem das amplitudes do sinal analógico
        self.amostrado = self.funcao_cos(self.tt_amostrado)      


    def recupera(self):
        """
        Interpola as amplitudes que foram amostradas da função cosseno para demonstrar como seria o
        comportamento real do sinal recuperado a partir destas amostras.
        """
        
        t_min = np.min(self.tt_amostrado)
        t_max = np.max(self.tt_amostrado)
        self.tt_recuperado = np.arange(t_min, t_max - self.dt, self.dt)
        
        funcao_recuperado = interp1d(self.tt_amostrado, self.amostrado, kind='cubic')
        self.recuperado = funcao_recuperado(self.tt_recuperado)

        
# funções para as figuras

def plota_amostragem(sinal, n_fig='X'):
    """
    Plota as amplitudes amostradas em relação ao sinal analógico original
    """
        
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(sinal.tt_analogico, sinal.analogico, '-r', label='Sinal original')
    ax.plot(sinal.tt_amostrado, sinal.amostrado, 'ob', label='Amplitudes amostradas')
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title("Figura " + n_fig + ". Sinal amostrado a uma frequência de " + \
                 str(1/sinal.dt_amostrado) + " amostras por segundo", fontsize=14)
    ax.set_xlim([0, sinal.janela_tempo])
    ax.set_ylim([np.min(sinal.analogico)*1.25, np.max(sinal.analogico)*1.25])
    ax.grid()
    ax.legend(loc='lower right', fontsize=12)

    
def plota_analogico(sinal, n_fig='X'):
    """
    Plota a função cosseno calculada a intervalos de tempo pequenos o suficiente para que se possa 
    fazer uma representação do sinal analógico original.
    """
    
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(sinal.tt_analogico, sinal.analogico)
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title("Figura " + n_fig + ". Representação do sinal analógico original", fontsize=14)
    ax.set_xlim([0, sinal.janela_tempo])
    ax.set_ylim([np.min(sinal.analogico)*1.25, np.max(sinal.analogico)*1.25])
    ax.grid()
    
    
def plota_representacao(sinal, n_fig='X'):
    """
    Plota a interpolação do sinal feita a partir das amostras obtidas do sinal analógico.
    """
    
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(sinal.tt_recuperado, sinal.recuperado, '--b', label='Sinal recuperado', linewidth=1)
    ax.plot(sinal.tt_amostrado, sinal.amostrado, 'ob', label='Amplitudes amostradas')
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title("Figura " + n_fig + ". Sinal recuperado da amostragem", fontsize=14)
    ax.set_xlim([0, sinal.janela_tempo])
    ax.set_ylim([np.min(sinal.analogico)*1.25, np.max(sinal.analogico)*1.25])
    ax.grid()
    ax.legend(loc='lower right', fontsize=12)
    
    
def plota_comparacao(sinal, n_fig='X'):
    """
    Plota a interpolação do sinal feita a partir das amostras obtidas do sinal 
    analógico e a sobrepõe à representação do sinal analógico original.
    """
    
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(sinal.tt_analogico, sinal.analogico, '-r', label='Sinal original')
    ax.plot(sinal.tt_recuperado, sinal.recuperado, '--b', label='Sinal recuperado')
    ax.set_xlabel("Tempo (s)", fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title("Figura " + n_fig + ". Comparação entre o sinal original e o sinal"\
                 " recuperado da amostragem", fontsize=14)
    ax.set_xlim([0, sinal.janela_tempo])
    ax.set_ylim([np.min(sinal.analogico)*1.25, np.max(sinal.analogico)*1.25])
    ax.grid()
    ax.legend(loc='lower right', fontsize=12)