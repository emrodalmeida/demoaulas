from scipy.fftpack import fft, fftfreq, ifft
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt

# funções dos cálculos

def gera_sinal(a, f, t_max):
    """
    Gera uma função obtida a partir de uma sobreposição de funções seno, onde cada uma delas é caracterizada por
    uma amplitude A e frequência f na forma s(t) = A * cos(2 * pi * f * t). A sobreposição destas funções será o sinal
    analógico de referência. Esta função pode ser amostrada em qualquer instante de tempo que se queira, de forma 
    que esta é a melhor forma de representar um sinal contínuo para os objetivos desta demonstração.
    """
    
    dt = t_max/1000
    tt = np.arange(0, t_max + dt, dt)                        # eixo de tempo estendido
    
    s = np.zeros(np.shape(tt))
    
    for i in range(len(a)):
        s = s + (a[i] * np.sin(2 * np.pi * f[i] * tt))
        
    ruido = np.random.normal(loc=0.0, scale=1, size=tt.shape)   # ruído branco para estabilizar a filtragem
    
    return s, tt      # função que caracteriza o sinal analógico



def calcula_espectro(tt, ss):
    
    dt = tt[1] - tt[0]
    n_amostras = len(ss)

    espectro = fft(ss, axis=0)
    ff = fftfreq(len(ss), dt)
    
    return ff, espectro


def gera_filtro(f_c, forma, ff):
    i_max_freq = np.abs(ff-np.max(ff)).argmin()   # indice da máxima frequência positiva
    delta_rampa = ff[1] - ff[0]     # a largura das rampas é de apenas 1 delta_f

    # ajusta a caixa das palavras e troca a separação para um traço ao invés de um espaço
    forma = forma.lower()
    if forma in ['passa baixa', 'passa alta', 'passa banda', 'rejeita banda']:
        forma = '-'.join(forma.split(' '))
    
    if forma=='passa-baixa':
        amp_caixa = [1, 1, 0, 0]                           # amplitudes da caixa do filtro na parte positiva do espectro
        f_rampa = f_c[0] + delta_rampa                               # frequência no fim da rampa do filtro
        f_caixa_pos = np.array([ff[0], f_c[0], f_rampa, np.max(ff)]) # caixa para a parte positiva do espectro

    elif forma=='passa-alta':
        amp_caixa = [0, 0, 1, 1]                            # amplitudes da caixa do filtro na parte positiva do espectro
        f_rampa = f_c[0] - delta_rampa                               # frequência no início da rampa do filtro
        f_caixa_pos = np.array([ff[0], f_rampa, f_c[0], np.max(ff)]) # caixa para a parte positiva do espectro

    elif forma=='passa-banda':
        amp_caixa = [0, 0, 1, 1, 0, 0]                      # amplitudes da caixa do filtro na parte positiva do espectro
        f_rampa_sub = f_c[0] - delta_rampa                                # frequência no início da rampa do filtro
        f_rampa_desc = f_c[1] + delta_rampa                               # frequência no fim da rampa do filtro
        f_rampa_sub, f_rampa_desc, f_c = verifica_rampas(f_rampa_sub, f_rampa_desc, f_c, ff)
        f_caixa_pos = np.array([ff[0], f_rampa_sub, f_c[0], f_c[1], \
                                f_rampa_desc, np.max(ff)])                # caixa para a parte positiva do espectro

    elif forma=='rejeita-banda':
        amp_caixa = [1, 1, 0, 0, 1, 1]                      # amplitudes da caixa do filtro na parte positiva do espectro
        f_rampa_desc = f_c[0] - delta_rampa                               # frequência no fim da rampa do filtro
        f_rampa_sub = f_c[1] + delta_rampa                                # frequência no inicio da rampa do filtro
        f_rampa_desc, f_rampa_sub, f_c = verifica_rampas(f_rampa_desc, f_rampa_sub, f_c, ff)
        f_caixa_pos = np.array([ff[0], f_rampa_desc, f_c[0], f_c[1], \
                                f_rampa_sub, np.max(ff)])                # caixa para a parte positiva do espectro

    # espelhamnto da caixa na parte positiva do espectro
    f_caixa_neg = np.flip(-1 * f_caixa_pos)   # caixa para a parte negativa do espectro
    f_caixa_neg[-1] = ff[-1]
    f_caixa_neg[0] = ff[i_max_freq+1]

    # interpolação das funções caixa para as frequências do espectro
    caixa_pos = interp1d(f_caixa_pos, amp_caixa, kind='linear')
    caixa_neg = interp1d(f_caixa_neg, np.flip(amp_caixa), kind='linear')

    return np.concatenate([caixa_pos(ff[:i_max_freq+1]), caixa_neg(ff[i_max_freq+1:])], axis=0)


def verifica_rampas(f_rampa1, f_rampa2, f_c, ff):
    """
    Verifica se as rampas do filtro passa-banda e rejeita-banda estão dentro dos limites do espectro.
    Não funciona para passa-alta e passa-baixa porque como só precisa de uma frequência de corte não 
    faz sentido estabelecer esta frequência próximo dos limites do espectro.
    """
        
    # para evitar pegar frequências negativas no limite da rampa
    if f_rampa1 < 0:                          
        f_c[0] = f_c[0] - f_rampa1    # desloca a frequêcia de corte para cima no eixo de frequências
        f_rampa1 = ff[0]              # define o início da rampa na frequência inicial do espectro

    # para evitar pegar frequências acima da frequência máxima do espectro no limite da rampa
    if f_rampa2 > np.max(ff):    
        f_c[1] = np.max(frequencias) - (ff[1] - ff[0])   # desloca a frequêcia de corte para baixo no eixo de frequências
        f_rampa2 = np.max(frequencias)    # define o início da rampa na maior frequência positiva do espectro

    return f_rampa1, f_rampa2, f_c


def executa_filtragem(espec, f_c, forma, ff):
    filtro = gera_filtro(f_c, forma, ff)
    amp_filtrada = espec * filtro
    
    return np.real(ifft(amp_filtrada))


# funções das figuras

def ajusta_escala_tempo(tt):
    
    """
    Ajusta a escala de tempo a ser plotada nas figuras para não precisar mostrar os valores em notação científica.
    """
    
    if np.max(tt) < 1e-6 and np.max(tt) >= 1e-9:
        tt = tt * 1e9
        titulo_eixo_t = "Tempo (ns)"
    elif np.max(tt) < 1e-3 and np.max(tt) >= 1e-6:
        tt = tt * 1e6
        titulo_eixo_t = "Tempo ($\mu$s)"
    elif np.max(tt) < 1 and np.max(tt) >= 1e-3:
        tt = tt * 1e3
        titulo_eixo_t = "Tempo (ms)"
    else:
        titulo_eixo_t = "Tempo (s)"

    return tt, titulo_eixo_t



def ajusta_escala_frequencia(ff):
    """
    Ajusta a escala de tempo a ser plotada nas figuras para não precisar mostrar os valores em notação científica.
    """
    
    if np.max(ff) >= 1e9 and np.max(ff) < 1e12:
        ff = ff * 1e-9
        titulo_eixo_f = "Frequência (GHz)"
    elif np.max(ff) >= 1e6 and np.max(ff) < 1e9:
        ff = ff * 1e-6
        titulo_eixo_f = "Frequência (MHz)"
    elif np.max(ff) >= 1e3 and np.max(ff) < 1:
        ff = ff * 1e-3
        titulo_eixo_f = "Frequência (kHz)"
    else:
        titulo_eixo_f = "Frequência (Hz)"

    return ff, titulo_eixo_f



def plota_sinal(tt, ss, n_figura='X'):
    """
    Plota o sinal calculado pela sobreposição das funções seno.
    """
    
    tt, rotulo_x = ajusta_escala_tempo(tt)
    
    fig, ax = plt.subplots(figsize=(15,3))
    ax.plot(tt, ss)
    ax.set_xlabel(rotulo_x, fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.set_title("Figura " + n_figura + ". Sinal Original", fontsize=14)
    ax.set_xlim([tt[0], tt[-1]])
    ax.grid()

    
def plota_espectro(ff, espec, n_figura='X'):
    n_samples = len(espec)       # vai ser o mesmo número de amostras do sinal porque a fft não usou zeros adicionais

    nf_positivas = round(n_samples / 2) + 1     # número de frequências positivas
    amplitudes = (2 / n_samples) * np.abs(espec[:nf_positivas])
    frequencias = ff[:nf_positivas]
    
    frequencias, rotulo_x = ajusta_escala_frequencia(frequencias)
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.stem(frequencias, amplitudes)
    ax.set_xlabel(rotulo_x, fontsize=14)
    ax.set_ylabel('Amplitude (ua)', fontsize=14)
    ax.set_xlim([0, np.max(frequencias)])
    ax.set_ylim([0, np.max(amplitudes)*1.25])
    ax.set_title("Figura " + n_figura + ". Espectro de amplitudes do sinal original", fontsize=14)
    # ax.set_xticks(np.arange(0, frequencias[-1], 2))
    ax.grid()
    
    
def plota_filtragem(espec, ff, f_c, forma, n_figura='X'):
    """
    Apenas plota a representação da seleção de frequências com o filtro sobre a parte
    positiva do espectro, porém não executa a filtragem propriamente dita    
    """
    
    n_samples = len(espec)       # vai ser o mesmo número de amostras do sinal porque a fft não usou zeros adicionais
    nf_positivas = round(n_samples / 2)     # número de frequências positivas
    amplitudes = (2 / n_samples) * np.abs(espec[:nf_positivas])   # amplitudes das frequências positivas
    pos_ff = ff[:nf_positivas]

    filtro = gera_filtro(f_c, forma, ff)
    escala_filtro = np.max(amplitudes) + 0.10 * np.max(amplitudes)   # escala gráfica para plotar o contorno do filtro

    pos_ff, rotulo_x = ajusta_escala_frequencia(pos_ff)    
    
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(13,7))
    ax[0].stem(pos_ff, amplitudes)
    ax[0].plot(pos_ff, filtro[:nf_positivas] * escala_filtro, '--r')
    ax[0].set_xlabel(rotulo_x, fontsize=14)
    ax[0].set_ylabel('Amplitude (ua)', fontsize=14)
    ax[0].set_xlim([0, np.max(pos_ff)])
    ax[0].set_ylim([0, np.max(amplitudes)*1.25])
    ax[0].set_title("Figura " + n_figura + "a. Filtro " + forma + " sobre o espectro", fontsize=14)
    #ax[0].set_xticks(np.arange(0, ff[-1], 2))
    ax[0].grid()

    ax[1].stem(pos_ff, amplitudes * filtro[:nf_positivas])
    ax[1].set_xlabel(rotulo_x, fontsize=14)
    ax[1].set_ylabel('Amplitude (ua)', fontsize=14)
    ax[1].set_xlim([0, np.max(pos_ff)])
    ax[1].set_ylim([0, np.max(amplitudes)*1.25])
    ax[1].set_title("Figura " + n_figura + "b. Frequências remanescentes após a filtragem", fontsize=14)
    #ax[1].set_xticks(np.arange(0, ff[-1], 2))
    ax[1].grid()

    plt.tight_layout()
    
    
def plota_gabarito(tt, s_filtrado, s_limpo, n_figura='X'):

    tt, rotulo_x = ajusta_escala_tempo(tt)
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(tt, s_limpo, '-b', alpha=0.15, linewidth=3, label='Sinal original (gabarito)')
    ax.plot(tt, s_filtrado, '-r', label='Sinal após a filtragem')
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim([0, np.max(tt)])
    ax.set_ylim([np.min([s_filtrado, s_limpo])*1.25, np.max([s_filtrado, s_limpo])*1.25])
    ax.set_title("Figura " + n_figura + ". Comparação entre o sinal filtrado e o sinal original", fontsize=14)
    ax.set_xlabel(rotulo_x, fontsize=14)
    ax.set_ylabel("Amplitude (ua)", fontsize=14)
    ax.grid()
    # ax.arrow(0.3e-7, -15000, 0, 5000, length_includes_head=True)      # não consigo fazer esta linha funcionar
    
    
# funções de notificação do usuário

def verifica_filtro(f_c, forma):
    """
    Imprime na tela uma mensagem para informar ao usuário se há algum problema com os parâmetros escolhidos para
    a filtragem. Se os parâmetros estiverem corretos não imprime mensagem nenhuma.
    """
    
    sublinhado = '\033[4m'
    negrito = '\033[1m'
    vermelho = '\033[91m'
    normal = '\033[0m'
    
    atencao = negrito + vermelho + sublinhado + 'ATENÇÃO!' + normal
    
    if len(f_c) == 2 and f_c[1] < f_c[0]:
        print("\n\n" + atencao + " As frequências de corte devem ser informadas em ordem crescente!\n\n")
        
    if len(f_c) == 2 and f_c[0] == f_c[1]:
        print("\n\n" + atencao + " As frequências de corte devem ser diferentes!\n\n")
    
    if not type(f_c) == list:
        print("\n\n" + atencao + " Coloque a frequência de corte entre colchetes!\n\n")
        
    if type(f_c) == list and any([f<0 for f in f_c]):
        print("\n\n Atenção! As frequências de corte devem ser positivas!\n\n")

    if tipo.lower() not in ['passa-baixa', 'passa-alta', 'passa-banda', 'rejeita-banda', \
                    'passa baixa', 'passa alta', 'passa banda', 'rejeita banda']:
        forma = negrito + forma + normal
        print("\n\n" + atencao + " O filtro " + forma + " não é válido, verifique a digitação!\n\n")
        