# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 08:54:06 2021

@author: emerson.almeida


Elaborado para funcionar da forma mais parecida possível com os arquivos de 
dados que são obtidos em campo. Os dados devem ser separados por colunas, sendo
elas corresndentes às posições dos eletrodos A, B, M, N seguidas pelos valores
de resistividade aparente e de desvio padrão da medida (se não tiver, é 0.0).

Isso corresponde ao formato 'simple' do SimPEG (v0.15.0), que é o que mais se
aproxima do arquivo de dados do Elrec. Por isso, a primeira linha do arquivo
de dados deve conter a string:
    ! simple FORMAT

"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (LogNorm, Normalize)
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import surface2ind_topo
from SimPEG import (maps, data_misfit, regularization, optimization,
                    inverse_problem, inversion, directives)
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import plot_pseudosection
from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc


def cria_malha_2D(max_xz, dados, topo, dh=None, regioes=None):
    """
    Parameters
    ----------
    max_xz : tupla
        Valores máximos nas direções x e z do domínio a ser invertido.
    dados : objeto survey do SimPEG
        Contém as informações referentes à configuração dos eletrodos do
        caminhamento elétrico e os dados lidos do arquivo de entrada
    topo : array 2D
        Coordenadas de topografia com coordenadas x na primeira coluna e com
        coordenadas z na segunda. As coordenadas em x devem exceder as 
        coordenadas do espaço a ser invertido em pelo menos 1.5x em ambas
        as extremidades.
    dh : float, opcional
        tamanho em metros da menor célula da malha discretizada do modelo.
        O default é None. Se não for informado o valor de dh é considerado
        como 1/100 do tamanho máximo do modelo na direção x.
    regioes : dicionário, opcional
        Dicionário com as regiões que deveraão ser refinadas na malha do 
        modelo. Cada região deve ser informada como uma tupla de 4 valores
        contendo as coordenadas na ordem x_inferior_esquerdo, 
        z_inferior_esquerdo, x_superior_direito, z_superior_direito. O 
        default é None. Se não for informado é assumido o valor de dh para
        todas as células da malha.

    Returns
    -------
    tupla contendo a malha gerada, o mapa de condutividades do modelo e o
    mapa para criação da figura do modelo.

    """
    
    if not dh:
        dh = np.round(max_xz[0]/100, 1)
    
    xf, zf = [i*1.5 for i in max_xz]
     
    # número de células em cada direção
    n_cel_x = 2 ** int(np.round(np.log((2*xf) / dh) / np.log(2.0)))
    n_cel_z = 2 ** int(np.round(np.log(zf / dh) / np.log(2.0)))
    
    # Malha base. Manter a origem em CN para evitar efeitos de borda.
    malha = TreeMesh([[(dh, n_cel_x)],[(dh, n_cel_z)]], origin='CN')
    
    # refinamento da malha próximo à superfície
    malha = refine_tree_xyz(malha, topo, octree_levels=[2,4,6], \
                            method='surface', finalize=False)
    
    # refinamento da malha em torno dos eletrodos
    eletrodos = np.c_[dados.survey.locations_a, dados.survey.locations_b, \
                      dados.survey.locations_m, dados.survey.locations_n]
    eletrodos = np.unique(np.reshape(eletrodos, (4*dados.nD, 2)), axis=0)
    malha = refine_tree_xyz(malha, eletrodos, octree_levels=[2,4,6], \
                            method='radial', finalize=False)

    # refinamento da malha nas regiões de interesse, i.e., onde se presume que
    # estarão os alvos a serem observados após a inversão.
    if regioes:
        for r in regioes.keys():
            coordenadas = regioes[r]
            x_inf_esq, z_inf_esq, x_sup_dir, z_sup_dir = coordenadas
            
            xp, zp = np.meshgrid([x_inf_esq, x_sup_dir], \
                                 [z_sup_dir, z_inf_esq])
            xz = np.c_[mkvc(xp), mkvc(zp)]
            malha = refine_tree_xyz(malha, xz, octree_levels=[4,4], \
                                    method='box', finalize=False)
    
    else:
        xp, zp = np.meshgrid([0.0, xf], [-zf, 0.0])
        xz = np.c_[mkvc(xp), mkvc(zp)]
        malha = refine_tree_xyz(malha, xz, octree_levels=[4,4], \
                                method='box', finalize=False)
        
    malha.finalize()
    
    # Desloca os eletrodos para a interface ar-solo da malha
    topo_2d = np.unique(topo, axis=0)
    indices_ativos = surface2ind_topo(malha, topo_2d)
    dados.survey.drape_electrodes_on_topography(malha, indices_ativos)
    
    # inclui os indices na malha para carregar como auxiliar para 
    # fora da função
    malha.indices_ativos = indices_ativos
    
    # Define o mapa de condutividades da malha
    condutividade_ar = np.log(1e-8)
    mapa_ativo = maps.InjectActiveCells(malha, indices_ativos, \
                                        np.exp(condutividade_ar))
    mapa_condutividade = mapa_ativo * maps.ExpMap()
    mapa_figura = maps.InjectActiveCells(malha, indices_ativos, np.nan)
    # nc = int(indices_ativos.sum())
    
    # Figuras das pseudo-seções com a escala de profundidade correta.
    arranjo = {'dipole-dipole': 'dipolo-dipolo',
               'pole-dipole': 'polo-dipolo',
               'pole-pole': 'polo-polo'}
    
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.75])
    plot_pseudosection(dados, ax=ax1, scale='log', 
                       data_locations=True, plot_type='contourf', \
                       cbar_label='Tensão Normalizada (V/A)', \
                       contourf_opts={"levels": 75, \
                                      "cmap": mpl.cm.gist_rainbow})
    ax1.set_title('Pseudo-seção de tensão normalizada adquirida com arranjo ' \
                  + arranjo[dados.survey.survey_type], fontsize=14)
    ax1.set_xlabel('Distância (m)', fontsize=12)
    ax1.set_ylabel('Profundidade (m)', fontsize=12)
    
    
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.75])
    plot_pseudosection(dados, ax=ax1, scale='log', 
                       data_locations=True, plot_type='contourf', \
                       data_type='apparent resistivity', \
                       cbar_label=r'$\rho_{a}$ ($\Omega$.m)', \
                       contourf_opts={"levels": 75, \
                                      "cmap": mpl.cm.gist_rainbow})
    ax1.set_title('Pseudo-seção de resistividade aparente adquirida com ' \
                  'arranjo ' + arranjo[dados.survey.survey_type], fontsize=14)
    ax1.set_xlabel('Distância (m)', fontsize=12)
    ax1.set_ylabel('Profundidade (m)', fontsize=12)

    # Figura da malha
    ex = dados.survey.electrode_locations[:, 0]
    ez = dados.survey.electrode_locations[:, 1] + 0.15
    
    fig = plt.figure(figsize=(11, 3))
    ax = fig.add_axes([0.1, 0.15, 0.75, 0.75])
    malha.plotGrid(ax=ax)
    ax.scatter(ex, ez, marker='1', s=2**7, c='k')
    ax.set_title('Posições dos eletrodos sobre a malha que será usada ' \
                 'na inversão', fontsize=14)
    ax.set_xlim([0.0, xf/1.5])
    ax.set_ylim([-zf/1.5, 0.49])
    ax.set_xlabel('Distância (m)', fontsize=12)
    ax.set_ylabel('Profundidade (m)', fontsize=12)
    
    return (malha, mapa_condutividade, mapa_figura)


def carrega_dados(arq_dados, arq_topo):
    """
    Parameters
    ----------
    arq_dados : string
        Nome do arquivo ASCII contendo os dados a serem invertidos, sem a 
        extensão.
        O arquivo deve conter valores de tensão normalizada (V/A). Os dados 
        devem ser separados por colunas, sendo elas corresndentes às posições 
        dos eletrodos A, B, M, N seguidas pelos valores de resistividade 
        aparente e de desvio padrão da medida (se não tiver, pode ser 0.0).
        Isso corresponde ao formato 'simple' do SimPEG (v0.15.0), então a 
        primeira linha do arquivo deve conter a string:
            ! simple FORMAT
        -----------------------------
    
        Quando se faz o load do formato 'simple' as coordenadas dos eletrodos
        ficam com valor 9999 (valor padrão da variável dummy_elevation na 
        função read_dcip2d_ubc). É preciso posicioná-los na superfície para 
        ter as
        profundidades corretas, mas para isso precisa criar a malha antes.
        
        Para corrigir isso será preciso usar posteriormente o método 
        drape_electrodes_on_topography,
        que por sua vez precisa ter a malha construída para que seja 
        utilizado.
        
        Por causa disso o plot da pseudo-seção de resistividade aparente dos
        dados lidos é feito apenas depois da criação da malha na função
        cria_malha_2d
    arq_topo : string
        Nome do arquivo ASCII contendo as informações de topografia, sem a
        extensão e com 
        coordenadas x na primeira coluna e com coordenadas z na segunda. 
        As coordenadas em x devem exceder as coordenadas do espaço a ser 
        invertido em pelo menos 1.5x em ambas as extremidades.

    Returns
    -------
    dados : objeto survey do SimPEG
        Contém as informações referentes à configuração dos eletrodos do
        caminhamento elétrico e os dados lidos do arquivo de entrada
    topo : array 2D
        Coordenadas de topografia com coordenadas x na primeira coluna e com
        coordenadas z na segunda.
    """
    
    topo = np.loadtxt(arq_topo)
    dados = read_dcip2d_ubc(arq_dados, data_type='volt', format_type='simple')
    
    return (dados, topo)





def executa_inversao(max_xz, topo, ce, mapas_modelo, rho_bg, n_iter, \
                     desv_pad_dados, chifact, beta_i, taxa_red, n_iter_beta):
    """
    Esta versão só aceita um modelo inicial de resistividade homogênea. Isso 
    vai ser mudado futuramente.

    Parameters
    ----------
    max_xz : tupla
        Valores máximos nas direções x e z do domínio a ser invertido. Ambos
        os valores devem ser positivos.
    topo : topo : array 2D
        Coordenadas de topografia com coordenadas x na primeira coluna e com
        coordenadas z na segunda. As coordenadas em x devem exceder as 
        coordenadas do espaço a ser invertido em pelo menos 1.5x em ambas
        as extremidades.
    ce : objeto survey do SimPEG
        Contém as informações referentes à configuração dos eletrodos do
        caminhamento elétrico e os dados lidos do arquivo de entrada
    mapas_modelo : tupla
        Objetos de malha do SimPEG referentes à malha gerada, ao mapa de 
        condutividades do modelo e ao mapa para criação da figura do modelo.
    rho_bg : float
        Resistividade do background homogêneo do modelo inicial, em Ohm.m.
    n_iter : int
        Número máximo de iterações da inversão.
    desv_pad_dados : float
        Desvio padrão atribuído aos dados observados.
    chifact : float
        Norma mínima a ser atingida na inversão.
    beta_i : float
        Valor inicial do parâmetro beta de regularização.
    taxa_red : float
        Taxa de redução do parâmetro beta.
    n_iter_beta : int
        Número de iteraações consideradas na variação do parâmetro beta.

    Returns
    -------
    None.

    """
    
    malha, mapa_condutividade, mapa_figura = mapas_modelo
    xf, zf = [i*1.5 for i in max_xz]
    ce.standard_deviation = desv_pad_dados * np.abs(ce.dobs)
    
    # modelo inicial
    nc = mapa_condutividade.shape[0]     # número de células ativas
    modelo_inicial = np.log(1/rho_bg) * np.ones(nc)  # precisa ser o log!

    # Figura do modelo inicial
    limites = LogNorm(vmin=1e1, vmax=1e3)
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
    malha.plot_image(mapa_figura * (1/np.exp(modelo_inicial)), ax=ax1, \
                     grid=False, pcolor_opts={'norm': limites})
    ax1.set_xlim(0.0, xf/1.5)
    ax1.set_ylim(-zf/1.5, 0.49)
    ax1.set_title('Modelo Inicial', fontsize=14)
    ax1.set_xlabel('Distância (m)', fontsize=12)
    ax1.set_ylabel('Profundidade (m)', fontsize=12)
    ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=limites, orientation='vertical')
    cbar.set_label(r'$\rho$ ($\Omega$.m)', rotation=90, labelpad=12, size=12)

    # configura a solução do problema direto
    fwd = dc.simulation_2d.Simulation2DNodal(malha, survey=ce.survey,
                                             sigmaMap=mapa_condutividade)

    # configura o erro de ajuste
    misfit = data_misfit.L2DataMisfit(data=ce, simulation=fwd)

    # configura a suavidade da regularização em cada direção
    reg = regularization.Simple(malha, indActive=malha.indices_ativos, \
                                mref=modelo_inicial, alpha_s=0.01, \
                                alpha_x=1, alpha_y=1)
    
    # configura o modelo de referência em termos de suavidade (?)
    reg.mrefInSmooth = True
    
    # configura a otimização por aproximação de Gauss-Newton inexata
    otimizacao = optimization.InexactGaussNewton(maxIter=n_iter)

    # configura a solução do problema inverso
    inv = inverse_problem.BaseInvProblem(misfit, reg, otimizacao)


    # atualiza a sensibilidade conforme o modelo é modificado
    atualiza_sensibilidade = directives.UpdateSensitivityWeights()
    
    # define o valor inicial para o parâmetro beta de equilíbrio entre
    # o erro de ajuste e a regularização
    beta_inicial = directives.BetaEstimate_ByEig(beta0_ratio=beta_i)

    # define a taxa de redução do beta e o número de iterações para cada
    # valor do parâmetro
    beta_schedule = directives.BetaSchedule(coolingFactor=taxa_red, \
                                            coolingRate=n_iter_beta)

    # desabilita a gravação do arquivo de saída a cada iteração
    saida_iteracoes = directives.SaveOutputEveryIteration(save_txt=False)

    # configura a condição de parada da inversão de acordo com o erro 
    # de ajuste
    erro_minimo = directives.TargetMisfit(chifact=chifact)
    
    # Update preconditioner (?)
    update_jacobi = directives.UpdatePreconditioner()

    lista_diretrizes = [atualiza_sensibilidade, beta_inicial, beta_schedule, \
                        saida_iteracoes, erro_minimo, update_jacobi]


    # integra as diretrizes à configuração do problema inverso
    inversao_dc = inversion.BaseInversion(inv, directiveList=lista_diretrizes)
    
    # roda a inversão e calcula as condutividades
    modelo_cond_inv = inversao_dc.run(modelo_inicial)
    condutividade_inv = mapa_condutividade * modelo_cond_inv
    condutividade_inv[~malha.indices_ativos] = np.nan
    
    # calcula as resistividades obtidas na inversão
    rho_inv = 1 / condutividade_inv
    
    # plota as saídas
    rho_min, rho_max = (np.min(rho_inv), np.max(rho_inv))
    log_norm = LogNorm(rho_min, rho_max)
    lin_norm = Normalize(rho_min, rho_max)
    intervalos = [np.ceil(i) for i in np.linspace(rho_min, rho_max, 10)]
    
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
    malha.plot_image(rho_inv, normal='Y', ax=ax1,\
                     pcolorOpts={'norm': log_norm, 'cmap': mpl.cm.gist_rainbow})
    ax1.set_xlim(0.0, xf/1.5)
    ax1.set_ylim(-zf/1.5, 0.49)
    ax1.set_title('Modelo de resistividades obtido após a inversão')
    ax1.set_xlabel('Distância (m)')
    ax1.set_ylabel('Profundidade (m)')
    ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(ax2, ticks=intervalos, norm=lin_norm, \
                                     orientation='vertical', \
                                     cmap=mpl.cm.gist_rainbow)
    cbar.set_label(r'$\rho$ ($\Omega$.m)', rotation=90, labelpad=12, size=12)

    return None



def roda_demo(arq_dados, arq_topo, espaco, rho_bg, dh=None, regioes=None, \
              n_iter=50, desv_pad_dados=0.05, chifact=1, beta_i=1e1, 
              taxa_red=3, n_iter_beta=2):
    
    """
    Essa é a função principal, que vai executar todas as outras funções.
    
    Parameters
    ----------
    arq_dados : string
        Nome do arquivo ASCII contendo os dados a serem invertidos, sem a 
        extensão.
    arq_topo : string
        Nome do arquivo ASCII contendo as informações de topografia, sem a
        extensão
        DESCRIPTION.
    espaco : tupla
        Valores máximos das coordenadas x e z do espaço a ser invertido. OS
        valores mínimos são considerados 0.0 metros para ambas. Ambos os 
        valores devem ser positivos.
    rho_bg : float
        Resistividade do background homogêneo do modelo inicial, em Ohm.m.
    dh : float, opcional
        tamanho em metros da menor célula da malha discretizada do modelo.
        O default é None. Se não for informado o valor de dh é considerado
        como 1/100 do tamanho máximo do modelo na direção x.
    regioes : dicionário, opcional
        Dicionário com as regiões que deveraão ser refinadas na malha do 
        modelo. Cada região deve ser informada como uma tupla de 4 valores
        contendo as coordenadas na ordem x_inferior_esquerdo, 
        z_inferior_esquerdo, x_superior_direito, z_superior_direito. O 
        default é None. Se não for informado é assumido o valor de dh para
        todas as células da malha.
    n_iter : int, optional
        Número máximo de iterações da inversão. O default é 50.
    desv_pad_dados : float, optional
        Desvio padrão atribuído aos dados observados. O padrão é 0.05.
    chifact : float, optional
        Norma mínima a ser atingida na inversão. O default é 1.0.
    beta_i : float, optional
        Valor inicial do parâmetro beta de regularização. O default é 1e1.
    taxa_red : float, optional
        Taxa de redução do parâmetro beta. O default é 3.0.
    n_iter_beta : int, optional
        Número de iteraações consideradas na variação do parâmetro beta. O 
        default é 2.

    Returns
    -------
    None.

    """
    
    ce, topografia = carrega_dados(arq_dados=arq_dados, arq_topo=arq_topo)
    mapas = cria_malha_2D(espaco, ce, topografia, dh, regioes)
    executa_inversao(espaco, topografia, ce, mapas, rho_bg, n_iter=n_iter, \
                     desv_pad_dados=desv_pad_dados, chifact=chifact, \
                     beta_i=beta_i, taxa_red=taxa_red, n_iter_beta=n_iter_beta)
    
    print('\n\n')
    plt.show()    
    
    return None




if __name__=='__main__':
    
    plt.close('all')
    
    # Roda um exemplo
    arquivo = 'dados_ce_exemplo_volt'
    
    arquivo_dados = arquivo + '.dat'
    arquivo_topografia = arquivo + '.top'
    dominio = (14.0, 4.0)   # (x_total, z_total)
    dh = 0.1
    detalhe = {'regiao 1': (3.0, -3.0, 6.0, 3.0),
               'regiao 2': (8.0, -3.0, 11.0, 3.0)}
    n_max_iteracoes = 40
    resistividade_bg = 100.0
    desv_pad_dados = 0.05
    chi = 1e-1
    beta_i = 1e1
    taxa_red = 3
    n_iter_beta = 2
    
    
    roda_demo(arquivo_dados, 
              arquivo_topografia, 
              dominio, 
              resistividade_bg, 
              dh=dh, 
              #regioes=detalhe, 
              chifact=chi)
    
else:
    mensagem = '\nMódulo de demonstração de inversão de caminhamento ' \
               'elétrico carregado com sucesso.\n'
    print(mensagem)

