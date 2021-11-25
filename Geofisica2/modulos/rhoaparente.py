import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from SimPEG import maps
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils import plot_1d_layer_model


def configura_ponto(A, B, M, N):

    coord_M = np.r_[M, 0.0, 0.0]
    coord_N = np.r_[N, 0.0, 0.0]
    MN = [dc.receivers.Dipole(coord_M, coord_N)]

    coord_A = np.r_[A, 0.0, 0.0]
    coord_B = np.r_[B, 0.0, 0.0]
    ABMN = [dc.sources.Dipole(MN, coord_A, coord_B)]

    ponto_sondagem = dc.Survey(ABMN)
    
    return ponto_sondagem


def calcula_ddp_1camada(esp, res, coord_ABMN):
    h = np.r_[esp]
    rho = np.r_[res, res]

    mapa_modelo = maps.IdentityMap(nP=len(rho))
    problema_direto = dc.simulation_1d.Simulation1DLayers(
                            survey=coord_ABMN,
                            rhoMap=mapa_modelo,
                            thicknesses=h,
                            data_type='volt')
    
    # Calculado com a corrente padrão de 1.0 A
    potencial = problema_direto.dpred(rho)
    potencial = np.round(potencial[0], 1)
    
    # considerando a simetria do arranjo, com M=potencial e N=-1*potencial
    ddp_calc = potencial * 2.0
    
    return(ddp_calc)


def calcula_ddp_2camadas(esp, res1, res2, coord_ABMN):
    h = np.r_[esp]
    rho = np.r_[res1, res2]

    mapa_modelo = maps.IdentityMap(nP=len(rho))
    problema_direto = dc.simulation_1d.Simulation1DLayers(
                            survey=coord_ABMN,
                            rhoMap=mapa_modelo,
                            thicknesses=h,
                            data_type='volt')
    
    # Calculado com a corrente padrão de 1.0 A
    potencial = problema_direto.dpred(rho)
    potencial = np.round(potencial[0], 1)
    
    # considerando a simetria do arranjo, com M=potencial e N=-1*potencial
    ddp_calc = potencial * 2.0
    
    return(ddp_calc)


# TODO: remover essa função na versão final
def calcula_rhoa(A, B, M, N, ddp_calc, corrente=1.0):
    AM = np.abs(M - A)
    AN = np.abs(N - A)
    BM = np.abs(M - B)
    BN = np.abs(N - B)
    
    k = 2*np.pi / (1/AM - 1/AN -1/BM + 1/BN)
    rho_a_calc = np.round((ddp_calc/corrente) * k, 1)
    print(rho_a_calc)
    return None


def plota_modelo(h1, rho1, rho2, limite_dist, ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    if rho1 < rho2:
        cor = ['gray', 'green']
        alpha = [0.25, 0.35]
    elif rho1 > rho2:
        cor = ['green', 'gray']
        alpha = [0.35, 0.25]
    elif rho1 == rho2:
        cor = ['gray', 'gray']
        alpha = [0.25, 0.25]
        
    limite_prof = 2 * h1
    camada1 = patches.Rectangle((-1*limite_dist, 0.0), 2*limite_dist, h1, 
                                linewidth=0.0, facecolor=cor[0], alpha=alpha[0])
    camada2 = patches.Rectangle((-1*limite_dist, h1), 2*limite_dist, limite_prof, 
                                linewidth=0.0, facecolor=cor[1], alpha=alpha[1])
    ax.add_patch(camada1)
    ax.add_patch(camada2)
    ax.plot([-1*limite_dist, limite_dist], [0.0, 0.0], '-k', linewidth=3.0)
    ax.text(-1*limite_dist + limite_dist/20, h1/3, s=(r'$\rho_1$ = ' + str(rho1) + r' $\Omega$.m'))
    
    if rho2 != rho1:
        ax.text(-1*limite_dist + limite_dist/20, h1+h1/3, s=(r'$\rho_2$ = ' + str(rho2) + r' $\Omega$.m'))
    
    ax.set_ylim([-limite_prof/5, limite_prof])
    ax.set_xlim([-1*limite_dist, limite_dist])
    ax.set_ylabel('Profundidade (m)')
    ax.set_xlabel('Distância (m)')
    ax.invert_yaxis()
    return ax


def plota_aquisicao(A, B, M, N, ax=None):
    if ax is None:
        ax = plt.gca()
    
    y_lim = ax.get_ylim()
    pos_y_eletrodo = y_lim[1] / 4.0
    pos_y_texto = y_lim[1] - y_lim[1] / 2.0
    ax.scatter(A, pos_y_eletrodo, s=200, c='blue', marker='v')
    ax.scatter(B, pos_y_eletrodo, s=200, c='blue', marker='v')
    ax.scatter(M, pos_y_eletrodo, s=200, c='red', marker='v')
    ax.scatter(N, pos_y_eletrodo, s=200, c='red', marker='v')
    ax.text(A, pos_y_texto, 'A', horizontalalignment='center')
    ax.text(B, pos_y_texto, 'B', horizontalalignment='center')
    ax.text(M, pos_y_texto, 'M', horizontalalignment='center')
    ax.text(N, pos_y_texto, 'N', horizontalalignment='center')
    return ax


def plota_ddp(ddp_calc, ax=None):
    if ax is None:
        ax = plt.gca()
    
    pos_y = np.abs(ax.get_ylim()[1] / 2)
    ax.text(0.0, pos_y, s=(r'$\Delta V_{MN}$ = %.2e V' % (ddp_calc)),
            horizontalalignment='center', verticalalignment='center')
    return ax


def executa_execicio(A, B, M, N, espessura, rho_baixa, rho_alta):
    ponto = configura_ponto(A, B, M, N)
    ddp1 = calcula_ddp_1camada(espessura, rho_baixa, ponto)
    ddp2 = calcula_ddp_1camada(espessura, rho_alta, ponto)
    ddp3 = calcula_ddp_2camadas(espessura, rho_baixa, rho_alta, ponto)
    ddp4 = calcula_ddp_2camadas(espessura, rho_alta, rho_baixa, ponto)

    max_dist_modelo = np.max(np.abs(np.r_[A, B, M, N, espessura]) * 1.5)

    calcula_rhoa(A, B, M, N, ddp1)
    calcula_rhoa(A, B, M, N, ddp2)
    calcula_rhoa(A, B, M, N, ddp3)
    calcula_rhoa(A, B, M, N, ddp4)

    fig, ax = plt.subplots(2, 2, figsize=(15, 7))

    plota_modelo(espessura, rho_baixa, rho_baixa, max_dist_modelo, ax=ax[0,0])
    plota_aquisicao(A, B, M, N, ax=ax[0,0])
    plota_ddp(ddp1, ax=ax[0,0])
    ax[0,0].set_title('Modelo 1')

    plota_modelo(espessura, rho_alta, rho_alta, max_dist_modelo, ax=ax[0,1])
    plota_aquisicao(A, B, M, N, ax=ax[0,1])
    plota_ddp(ddp2, ax=ax[0,1])
    ax[0,1].set_title('Modelo 2')

    plota_modelo(espessura, rho_baixa, rho_alta, max_dist_modelo, ax=ax[1,0])
    plota_aquisicao(A, B, M, N, ax=ax[1,0])
    plota_ddp(ddp3, ax=ax[1,0])
    ax[1,0].set_title('Modelo 3')

    plota_modelo(espessura, rho_alta, rho_baixa, max_dist_modelo, ax=ax[1,1])
    plota_aquisicao(A, B, M, N, ax=ax[1,1])
    plota_ddp(ddp4, ax=ax[1,1])
    ax[1,1].set_title('Modelo 4')

    plt.tight_layout()
    plt.show()
    
    return None