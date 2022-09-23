import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy

from SimPEG import maps
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils import plot_1d_layer_model

warnings.filterwarnings('ignore')

FONTE_PEQUENA = 12
FONTE_MEDIA = 14
FONTE_GRANDE = 14

plt.rc('font', size=FONTE_PEQUENA)
plt.rc('axes', titlesize=FONTE_GRANDE)
plt.rc('axes', labelsize=FONTE_MEDIA)
plt.rc('legend', fontsize=FONTE_PEQUENA)
plt.rc('xtick', labelsize=FONTE_PEQUENA)
plt.rc('ytick', labelsize=FONTE_PEQUENA)


tendcurva = False  # Atua como uma "chave" ON/OFF dos componentes gráficos adicionais


def muda_estado(tend):
    
    """
    Muda o estado da variável tendcurva e exibe uma mensagem informando o estado atual.
    """
        
    tend = not tend

    if tend:
        print('\nComponentes gráficos adicionais ATIVADOS.\n')
    elif not tend:
        print('\nComponentes gráficos adicionais DESATIVADOS.\n')
    
    return tend


class sev():

    """
    Sondagem Elétrica Vertical configurada de acordo com
    os parâmetros necessários para uso com o SimPEG
    """

    def __init__(self, meioAB, meioMN, arranjo):

        if len(meioAB)>len(meioMN):
            print("\nERRO: número de posições de AB/2 é maior do número de "
                "posições de MN/2.\n")
            return None
          
        elif len(meioAB)<len(meioMN):
            print("\nERRO: número de posições de AB/2 é menor do número de "
                "posições de MN/2.\n")
            return None

        else:
            self.arranjo = arranjo
            self.meioAB = meioAB      # distâncias de AB/2
            self.meioMN = meioMN      # distâncias de MN/2
            source_list = []

            for ii in range(len(meioAB)):
                A = np.r_[-1 * meioAB[ii], 0.0,0.0]
                B = np.r_[meioAB[ii], 0.0,0.0]
                M = np.r_[-1 * meioMN[ii], 0.0,0.0]
                N = np.r_[meioMN[ii], 0.0,0.0]

                receiver_list = [dc.receivers.Dipole(M, N)]
                source_list.append(dc.sources.Dipole(receiver_list, A, B))

            self.survey = dc.Survey(source_list)

            # inicializa sem nada porque ainda não foi feita a modelagem
            self.rho_a = None


    def fwd(self, modelo):

        """
        Executa a modelagem com os parâmetros definidos para a SEV
        """

        simulation = dc.simulation_1d.Simulation1DLayers(
                      survey = self.survey,
                      rhoMap = modelo.model_map,
                      thicknesses = modelo.h,
                      data_type = "apparent_resistivity"
                      )

        self.rho_a = simulation.dpred(modelo.rho)


class modelo_geo():
    """
    Modelo Geoelétrico 1-D a ser modelado
    """
    
    def __init__(self, rho, h):

        self.rho = rho
        self.h = h
        self.z_max = np.sum(self.h) * 1.3

        # Define mapping from model to 1D layers
        self.model_map = maps.IdentityMap(nP=len(self.rho))


    def mostra_grafico(self, n_fig='X'):
        """
        Método para plotar o modelo
        """
        
        # Define a malha 1D
        #mesh = TensorMesh([np.r_[self.h, self.z_max - self.h.sum()]])

        #fig, ax1 = plt.subplots(figsize=(4,4))
        
        ax1 = plot_1d_layer_model(self.h, self.model_map * self.rho)
        # plot_layer(self.model_map * self.rho, mesh, xlim=[10, 1e4], ax=ax1, showlayers=False)

        ax1.set_xlabel('Resistividade Real (Ohm.m)')
        ax1.set_ylabel('Profundidade (m)')
        ax1.set_title('Figura ' + n_fig + '. Modelo de n=' + \
                      str(len(self.rho)) + ' camadas')
        ax1.grid(which='both')
        plt.show()


def calcula_tendencias(sondagem, modelo):
    """
    Calcula a tendência da curva de resistividade aparente para cada
    camada individualmente, como se a camada correspondesse ao semi-espaço
    homogêneo de resistividade correspondente à resistividade da camada
    sendo considerada.
    
    Utiliza as configurações da sondagem fornecida para executar o cálculo
    do problema direto N-1 vezes, onde N é o número de camadas do modelo
    fornecido. A execução começa com um modelo de uma única camada, e as
    camadas são consideradas na modelagem gradativamente, uma a uma.
    """
    
    rho_a_list = []
    sev_parcial = deepcopy(sondagem)

    for ii in range(0, len(modelo.rho)-1):

        if ii==0:
            # Faz um modelo de duas camadas de resistividades iguais
            # para simular um semi-espaço homogêneo.
            res = np.r_[modelo.rho[:ii+1], modelo.rho[:ii+1]]
            esp = modelo.h[:ii+1]

        else:
            # Cria modelos com aumento gradual do número de camadas 
            res = modelo.rho[:ii+1]
            esp = modelo.h[:ii]

        modelo_parcial = modelo_geo(res, esp)
        sev_parcial.fwd(modelo_parcial)
        rho_a_list.append(sev_parcial.rho_a)

    return rho_a_list


def plota_sondagem(sondagem, modelo, tendencias=False, n_fig='X'):
    
    """
    Exibe a curva de resistividade aparente resultante da modelagem
    """
  
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    # insere as curvas com as tendências da resistividade aparente
    if tendencias:
        curvas = calcula_tendencias(sondagem, modelo)
        cor = ['--r', '--y', '--c', '--k', '--m', '-.r', '-.y', \
               '-.c', '-.k', '-.m']
        
        for n in range(len(curvas)):
            ax[0].loglog(sondagem.meioAB, curvas[n], cor[n], \
                         label='n = ' + str(n+1))
          
        ax[0].loglog(sondagem.meioAB, sondagem.rho_a, '-b', \
                     label='n = ' + str(n+2))
        ax[0].legend(loc='upper right')
    
    else:
        ax[0].loglog(sondagem.meioAB, sondagem.rho_a, '-b')
    
    ax[0].plot(sondagem.meioAB, sondagem.rho_a, 'ok')
    ax[0].set_xlabel('AB/2 (m)', fontsize='14')
    ax[0].set_ylabel('Resistividade Aparente ($\Omega$.m)', fontsize='14')
    ax[0].set_title('Figura ' + n_fig + 'a. Sondagem ' + sondagem.arranjo, \
                    fontsize='14')
    ax[0].set_ylim([10, 1000])
    ax[0].set_xlim([1, 1000])
    ax[0].grid(which='both')
    
    #mesh = TensorMesh([np.r_[modelo.h, modelo.z_max - modelo.h.sum()]])
    # plot_layer(modelo.model_map * modelo.rho, mesh, xlim=[10, 1e4], ax=ax[1], showlayers=False)
    ax[1] = plot_1d_layer_model(modelo.h, modelo.model_map * modelo.rho)
    ax[1].set_xlabel('Resistividade Real (Ohm.m)')
    ax[1].set_ylabel('Profundidade (m)')
    ax[1].set_title('Figura ' + n_fig + 'b. Modelo de n=' + \
                    str(len(modelo.rho)) + ' camadas', fontsize='14')
    ax[1].grid(which='both')

    plt.show()


def compara_sevs(sev1, sev2, n_fig='X'):
    
    """
    Exibe uma figura para comparação entre duas sondagens diferentes
    """

    fig, ax1 = plt.subplots(figsize=(7, 7))
    
    ax1.loglog(sev1.meioAB, sev1.rho_a, '-ob', label=sev1.arranjo)
    ax1.loglog(sev2.meioAB, sev2.rho_a, '-or', label=sev2.arranjo)
    ax1.set_xlabel('AB/2 (m)', fontsize='14')
    ax1.set_ylabel('Resistividade Aparente ($\Omega$.m)', fontsize='14')
    ax1.set_title('Figura ' + n_fig + '. Comparação entre curvas de SEVs', \
                  fontsize='14')
    ax1.set_ylim([10, 1000])
    ax1.legend(loc='upper right')
    ax1.grid(which='both')
    plt.show()