# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 08:30:12 2021

@author: emerson.almeida
"""

import warnings
warnings.filterwarnings('ignore')

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import model_builder, surface2ind_topo
from SimPEG.utils.io_utils.io_utils_electromagnetics import write_dcip2d_ubc
from SimPEG import maps, data
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
    plot_pseudosection,
)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def topografia_plana(x_max):

    # cria uma topografia plano-horizontal para o exemplo
    x_topo = np.linspace(-x_max*2, x_max*2, 501)
    z_topo = (x_topo * 0.0) + 0.01
    
    return np.c_[mkvc(x_topo), mkvc(z_topo)]


def cria_ce(arranjo, dx, n, ai_nf, topo):
    configuracao = {'dd': 'dipole-dipole', 'pd': 'pole-dipole', 'pp': 'pole-pole'}
    arranjo = arranjo.lower()
    
    if arranjo not in ['dd', 'pd', 'pp']:
        print('O arranjo deve ser \'dd\', \'dp\' ou \'pp\'.\n')
        return None
    
    # pontos de alocação dos eletrodos A, B, M, N
    pontos = generate_dcip_sources_line(configuracao[arranjo], 'volt', '2D', ai_nf, topo, n, dx)
    
    caminhamento = dc.survey.Survey(pontos, survey_type=configuracao[arranjo])
    
    return caminhamento


def cria_malha_2D(max_xz, ce, topo, dh, rho_bg=1e2, alvos=None, grade=False):

    xf, zf = [i*1.5 for i in max_xz]
    
    # número de células em cada direção
    n_cel_x = 2 ** int(np.round(np.log((2*xf) / dh) / np.log(2.0)))
    n_cel_z = 2 ** int(np.round(np.log(zf / dh) / np.log(2.0)))
    
    # Malha base. Manter a origem em CN para evitar efeitos de borda.
    malha = TreeMesh([[(dh, n_cel_x)],[(dh, n_cel_z)]], origin='CN')
    
    # refinamento da malha próximo à superfície
    malha = refine_tree_xyz(malha, topo, octree_levels=[4,4], method='surface', finalize=False)
    
    # refinamento da malha em torno dos eletrodos
    eletrodos = np.c_[ce.locations_a, ce.locations_b, ce.locations_m, ce.locations_n]
    eletrodos = np.unique(np.reshape(eletrodos, (4*ce.nD, 2)), axis=0)
    malha = refine_tree_xyz(malha, eletrodos, octree_levels=[4,4], method='radial', finalize=False)
    
    # refina a malha nas áreas onde serão posicionados os alvos
    # maxima discretização feita em uma borda igual a 1/5 da largura total do alvo 
    if alvos:
        for a in alvos.keys():
            coordenadas = alvos[a][0]
            x_c, z_c, r = coordenadas
            borda = np.round((2 * r) / 5, 1)
            x_min = x_c - r - borda
            x_max = x_c + r + borda
            z_min = z_c + r + borda
            z_max = z_c - r - borda

            xp, zp = np.meshgrid([x_min, x_max], [z_min, z_max])
            xz = np.c_[mkvc(xp), mkvc(zp)]
            malha = refine_tree_xyz(malha, xz, octree_levels=[4,4], method='box', finalize=False)

    malha.finalize()
    
    # inserção da condutividade na interface ar-solo no modelo
    # atenção porque o programa trabalha com CONDUTIVIDADES e a 
    # função cria_malha_2D está recebendo RESISTIVIDADES!    
    indices_solo = surface2ind_topo(malha, topo)
    nc = int(indices_solo.sum())
    
    condutividade_ar = 1e-8
    mapa_condutividade = maps.InjectActiveCells(malha, indices_solo, \
                                                condutividade_ar)
    modelo_condutividade = (1/rho_bg) * np.ones(nc)

    # inserção da condutividade dos alvos
    if alvos:
        for a in alvos.keys():
            coordenadas, rho = alvos[a]
            x_c, z_c, r = coordenadas
            indices_alvo = model_builder.getIndicesSphere(np.r_[x_c, z_c], r, \
                                                          malha.gridCC)
            indices_alvo = indices_alvo[indices_solo]
            modelo_condutividade[indices_alvo] = 1/rho
    
    # insere o modelo e o mapa de condutividade no objeto malha para ser 
    # usado pela função de modelagem
    malha.modelo_condutividade = modelo_condutividade
    malha.mapa_condutividade = mapa_condutividade
        
    # Figura do modelo
    mapa_figura = maps.InjectActiveCells(malha, indices_solo, np.nan)
    limites = LogNorm(vmin=1e1, vmax=1e3)
    ex = ce.electrode_locations[:, 0]
    ez = ce.electrode_locations[:, 1] + 0.10

    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
    malha.plot_image(mapa_figura * (1/modelo_condutividade), ax=ax1, grid=grade,
                    pcolor_opts={'norm': limites})
    ax1.scatter(ex, ez, marker='1', s=2**7, c='k')
    ax1.set_xlim(0.0, xf/1.5)
    ax1.set_ylim(-zf/1.5, 0.49)
    ax1.set_title('Posições dos eletrodos sobre o modelo', fontsize=14)
    ax1.set_xlabel('Distância (m)', fontsize=12)
    ax1.set_ylabel('Profundidade (m)', fontsize=12)
    
    ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=limites, orientation='vertical')
    cbar.set_label(r'$\rho$ ($\Omega$.m)', rotation=90, labelpad=12, size=12)

    return malha


def executa_modelagem(malha, ce, formato):
    
    mapa_condutividade = malha.mapa_condutividade
    modelo_condutividade = malha.modelo_condutividade
    
    # não precisa especificar o Solver na linha abaixo
    fwd = dc.simulation_2d.Simulation2DNodal(malha, survey=ce, sigmaMap=mapa_condutividade)
    
    # valores da ddp medida
    dpred_v = fwd.dpred(modelo_condutividade)
    
    # dados de resistividade aparente preditos (em forma de array)
    rho_a = apparent_resistivity_from_voltage(ce, dpred_v)

    arranjo = {'dipole-dipole': 'dipolo-dipolo',
               'pole-dipole': 'polo-dipolo',
               'pole-pole': 'polo-polo'}  
    
    # plota as pseudoseções
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.75])
    plot_pseudosection(ce, dobs=dpred_v, ax=ax1, scale='log', \
                       cbar_label='Tensão Normalizada (V/A)', \
                       contourf_opts={"levels": 75, "cmap": mpl.cm.gist_rainbow},
                       data_locations=True)
    ax1.set_title('Pseudo-seção de tensão normalizada obtida com arranjo ' + arranjo[ce.survey_type], fontsize=14)
    ax1.set_xlabel('Distância (m)', fontsize=12)
    ax1.set_ylabel('Profundidade (m)', fontsize=12)
    
    
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.75])
    plot_pseudosection(ce, dobs=rho_a, ax=ax1, scale='log', \
                       cbar_label=r'$\rho_{a}$ ($\Omega$.m)', \
                       contourf_opts={"levels": 75, "cmap": mpl.cm.gist_rainbow},
                       data_locations=True)
    ax1.set_title('Pseudo-seção de resistividade aparente obtida com arranjo ' + arranjo[ce.survey_type], fontsize=14)
    ax1.set_xlabel('Distância (m)', fontsize=12)
    ax1.set_ylabel('Profundidade (m)', fontsize=12)
    
    return rho_a, dpred_v


def roda_demo(lx, lz, tipo, dx, n, eletrodo_a, eletrodo_n, alvos=None, \
              arquivo=None, formato='volt'):
    dh = np.round(lx/100, 1)
    topografia = topografia_plana(lx)
    ce = cria_ce(tipo, dx, n, (eletrodo_a, eletrodo_n), topografia)
    modelo = cria_malha_2D((lx, lz), ce, topografia, dh=dh, alvos=alvos)
    
    # dados preditos
    dp_rho_a, dp_v = executa_modelagem(modelo, ce, formato=formato)
    dp = {'rho_a': dp_rho_a, 'volt': dp_v}
    
    # exporta os dados de resistividade aparente modelados
    if arquivo:
        
        if formato=='rho_a':
            print('\nATENÇÃO! se o arquivo de saída for usado posteriormente em', \
                  'uma inversão do SimPEG a saída precisa obrigatoriamente', \
                  'ser em \'volt\', caso contrário a inversão dá erro.\n')
        
        arquivo_topo = arquivo + '_' + formato + '.top'
        arquivo_dados = arquivo + '_' + formato + '.dat'
        
        dados_modelados = data.Data(ce, dp[formato])
        
        # data_type='volt' funciona tanto pra dados de tensão quanto 
        # de resistividade aparente
        write_dcip2d_ubc(arquivo_dados, dados_modelados, file_type='dobs', \
                         data_type='volt', format_type='simple')
        
        np.savetxt(arquivo_topo, topografia, fmt='%.2e')
    ''
    plt.show()
    
    return None
    

if __name__=='__main__':
    
    plt.close('all')
    
    # Roda um exemplo
    x_max = 14.0
    z_max = 3.0
    configuracao = 'dd'
    espacamento = 0.5
    n_niveis = 10
    posicao_a = 2.0
    posicao_n = 12.0
    prefixo_saida = 'dados_ce_exemplo'
    saida = 'rho_a'
    
    alvos = {'alvo 1': [(5.0, -1.0, 0.5), (10.0)],     # [(x, z, r), (rho)]
             'alvo 2': [(9.0, -1.0, 0.5), (1000.0)]}   # [(x, z, r), (rho)]
    
    roda_demo(lx=x_max, 
              lz=z_max, 
              tipo=configuracao, 
              dx=espacamento, 
              n=n_niveis, 
              eletrodo_a=posicao_a, 
              eletrodo_n=posicao_n, 
              alvos=alvos,
              #arquivo=prefixo_saida, 
              formato=saida)
    
else:
    mensagem = '\nMódulo de demonstração de caminhamento elétrico ' \
               'carregado com sucesso.\n'
    print(mensagem)























