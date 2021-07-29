# Importa packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandapower as pp
import time
import random
from pandapower.networks import case14, case_ieee30, case118, case300, case4gs
import tabulate
import numba
from numba import njit
from gekko import GEKKO
from pandapower.plotting.plotly import pf_res_plotly
import pandapower.plotting as pplot

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"
font = {'size'   : 15}
plt.rc('font', **font)
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 

sep14 = case14()
sep30 = case_ieee30()
sep118 = case118()

def inicializa_sep(sep, algorithm, relatorio=False):
    
    """"
    
    Função Para Inicializar os Sistemas com Dados Retirados de Outros Trabalhos Para Efeitos de Comparação
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------   
    sep : sistema elétrico de potência carregado pelo pandapower.
    algorithm : algoritmo de solução do fluxo de potência pelo pandapower. Opções:
    
    

        “nr” Newton-Raphson (pypower implementation with numba accelerations)

        “iwamoto_nr” Newton-Raphson with Iwamoto multiplier (maybe slower than NR but more robust)

        “bfsw” backward/forward sweep (specially suited for radial and weakly-meshed networks)

        “gs” gauss-seidel (pypower implementation)

        “fdbx” fast-decoupled (pypower implementation)

        “fdxb” fast-decoupled (pypower implementation)
        
        https://pandapower.readthedocs.io/en/v2.4.0/powerflow/ac.html
        
        
        Dados:
        
        https://www.teses.usp.br/teses/disponiveis/18/18154/tde-08102019-153756/publico/Diego.pdf
        
        https://matpower.org/docs/ref/matpower5.0/case_ieee30.html
        
        https://matpower.org/docs/ref/matpower5.0/case14.html
        
        https://matpower.org/docs/ref/matpower5.0/case118.html
        
        


    relatorio : caso relatorio = True, retorna relatório informando que o sistema foi carregado e o tempo de execução do algoritmo.
                caso relatorio = False, não retorna nada.
    
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    """
    
    if len(sep.bus)==14:
        
        sep.bus['min_vm_pu'] = 0.95
        sep.bus['max_vm_pu'] = 1.05
        sep.ext_grid['vm_pu'] = 1.05
#         sep.ext_grid['min_q_mvar'] = 0
#         sep.ext_grid['max_q_mvar'] = 10
        sep.ext_grid['min_q_mvar'] = -99
        sep.ext_grid['max_q_mvar'] = 99
        
        
      
        
        if algorithm == 'nr':
        
            inicio = time.time()
            pp.runpp(sep,algorithm=algorithm,numba = True, init = 'results', tolerance_mva = 1e-5, trafo_model='pi')
            fim = time.time()
            
            tempo = fim-inicio
        
        
        else:
            
            inicio = time.time()
            pp.runpp(sep,algorithm=algorithm, init = 'results', tolerance_mva = 1e-5)
            fim = time.time()
        
            tempo = fim-inicio
       
        if relatorio == True:
            
            
            print('Algoritmo: ', algorithm)
            print('Tempo de Execução: ', tempo)

        
    if len(sep.bus)==30:
        
        sep.bus['min_vm_pu'] = 0.95
        sep.bus['max_vm_pu'] = 1.05
        sep.ext_grid['vm_pu'] = 1.05
        sep.gen['max_q_mvar']=np.array([50,40,40,24,24])
        sep.gen['min_q_mvar']=np.array([-40,-40,-10,-6,-6])
        sep.ext_grid['max_q_mvar'] = 10
        sep.ext_grid['min_q_mvar'] = 0
#         sep.ext_grid['min_q_mvar'] = -99
#         sep.ext_grid['max_q_mvar'] = 99
        
        if algorithm == 'nr':
        
            inicio = time.time()
            pp.runpp(sep,algorithm=algorithm,numba = True, init = 'results', tolerance_mva = 1e-5)
            fim = time.time()
            
            tempo = fim-inicio
        
        
        else:
            
            inicio = time.time()
            pp.runpp(sep,algorithm=algorithm, init = 'results', tolerance_mva = 1e-5)
            fim = time.time()
        
            tempo = fim-inicio
       
        if relatorio == True:
            
            
            print('Algoritmo: ', algorithm)
            print('Tempo de Execução: ', tempo)
                
    if len(sep.bus)==118:
        
        sep.bus['min_vm_pu'] = 0.94
        sep.bus['max_vm_pu'] = 1.06
        sep.ext_grid['vm_pu'] = 1.06
        
        if algorithm == 'nr':
        
            inicio = time.time()
            pp.runpp(sep,algorithm=algorithm,numba = True, init = 'results', tolerance_mva = 1e-6)
            fim = time.time()
            
            tempo = fim-inicio
        
        
        else:
            
            inicio = time.time()
            pp.runpp(sep,algorithm=algorithm, init = 'results', tolerance_mva = 1e-5)
            fim = time.time()
        
            tempo = fim-inicio
       
        if relatorio == True:
            
            
            print('Algoritmo: ', algorithm)
            print('Tempo de Execução: ', tempo)

                    
    if len(sep.bus)==300:
        
        sep.bus['min_vm_pu'] = 0.94
        sep.bus['max_vm_pu'] = 1.06
        sep.ext_grid['vm_pu'] = 1.06
    
        if algorithm == 'nr':
        
            inicio = time.time()
            pp.runpp(sep,algorithm=algorithm,numba = True, init = 'results', tolerance_mva = 1e-5)
            fim = time.time()
            
            tempo = fim-inicio
        
        
        else:
            
            inicio = time.time()
            pp.runpp(sep,algorithm=algorithm, init = 'flat', tolerance_mva = 1e-5)
            fim = time.time()
        
            tempo = fim-inicio
       
        if relatorio == True:
            
            
            print('Algoritmo: ', algorithm)
            print('Tempo de Execução: ', tempo)
            
    voltages_init = sep.gen['vm_pu'].to_numpy()
    tap_pos = sep.trafo[~pd.isnull(sep.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)
    tap_neutral = sep.trafo[~pd.isnull(sep.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)
    tap_step_percent = sep.trafo[~pd.isnull(sep.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)       
    valor_pu_tap = (tap_pos-tap_neutral)*(tap_step_percent/100) + 1
    valor_bshunt = (sep.shunt['q_mvar']/(-100)).to_numpy()
    zeros = np.array([0,0,0,0,0,0])
    valor_inicial = np.expand_dims(np.concatenate((voltages_init, valor_pu_tap, valor_bshunt,zeros), axis = None), 0)



    return valor_inicial


    ##################################################################################################################################################################################

def matriz_condutancia(sep,relatorio=True):
    
    '''
    
    Calcula a matriz de condutâncias de linha, retornando apenas a parte triangular superior.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------   
    sep : sistema elétrico de potência carregado pelo pandapower.
    relatorio : caso relatorio = True, retorna relatório informando barras de origem e destino das linhas, resistências (pu), reatâncias (pu).
                caso relatorio = False, retorna apenas a parte triangular superior da matriz de condutâncias.
    Retorno
    -------    
    matriz_g: matriz de condutâncias entre barras com triângulo inferior zerado.
    
    Observações:
    ------------
    
    Potência Aparente de Base = 100 MVA
    
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    
    '''
    
    sep.line=sep.line.sort_index()
   
    sep.bus=sep.bus.sort_index()
    
    vbus = sep.bus.vn_kv.to_numpy(dtype=np.float64)
    
    zbase = np.power(np.multiply(vbus,1000), 2)/100e6
    
    # Inicializa Matriz Zerada
    
    matriz_z = np.zeros((9,len(sep.line.index.ravel())),dtype=np.float64)
    
    matriz_g = np.zeros((sep.bus.name.count(),sep.bus.name.count()), dtype=np.float64)
    
    g = np.zeros(len(sep.line.index.ravel()),dtype=np.float64)
    
    # Pega Valores de Barra Origem e Destino das Linhas
        
    matriz_z[0,:]=sep.line.from_bus
    
    matriz_z[1,:]=sep.line.to_bus
    
    
    for i in range(len(sep.line.index.ravel())):    
    
        matriz_z[2,i] = sep.line.r_ohm_per_km[i]/zbase[int(matriz_z[0,i])]
        matriz_z[3,i] = sep.line.x_ohm_per_km[i]/zbase[int(matriz_z[0,i])]
    
    # Calcula Condutâncias
    
    g = np.array(np.divide(matriz_z[2,:], np.power(matriz_z[2,:],2)+np.power(matriz_z[3],2)))
    z = np.sqrt(np.power(matriz_z[2,:],2) + np.power(matriz_z[3,:],2))
    b = np.array(np.divide(matriz_z[3,:], np.power(matriz_z[2,:],2)+np.power(matriz_z[3],2)))
    matriz_z[4,:]=g
    
    vo = []
    vd = []
    to = []
    td = []


    for bus in matriz_z[0,:]:

        vo.append(sep.res_bus['vm_pu'][sep.res_bus.index==bus].to_numpy(dtype=np.float64))
        to.append(sep.res_bus['va_degree'][sep.res_bus.index==bus].to_numpy(dtype=np.float64))


    for bus in matriz_z[1,:]:

        vd.append(sep.res_bus['vm_pu'][sep.res_bus.index==bus].to_numpy(dtype=np.float64))
        td.append(sep.res_bus['va_degree'][sep.res_bus.index==bus].to_numpy(dtype=np.float64))
    
    matriz_z[5,:] = vo
    matriz_z[6,:] = to
    matriz_z[7,:] = vd
    matriz_z[8,:] = td
    
    # Gera Matriz
    
    for posicao in range(len(sep.line.index.ravel())):
        
        matriz_g[matriz_z[0,posicao].astype(np.int),matriz_z[1,posicao].astype(np.int)] = g[posicao]
        
    
    if relatorio==True:
    
        tabela = np.zeros((len(sep.line.index.ravel()),7))
        tabela[:,0] = matriz_z[0,:]
        tabela[:,1] = matriz_z[1,:]
        tabela[:,2] = matriz_z[2,:]
        tabela[:,3] = matriz_z[3,:]
        tabela[:,4] = z
        tabela[:,5] = g
        tabela[:,6] = b

        table = tabulate.tabulate(tabela, headers = ['Barra de Origem', 'Barra de Destino','R (pu)','Xl (pu)','Z (pu)', 'G (pu)','B (pu)'], tablefmt="psql")
        print(table)
        
        if len(sep.bus)==14:

            sns.heatmap(matriz_g+matriz_g.T,annot=True,fmt='.6g',cmap='jet')
            plt.xlabel('Barra Origem')
            plt.ylabel('Barra Destino')
            plt.title('Matriz de Condutâncias de Linha Completa')

        
    if relatorio==False:
        
        return matriz_g, matriz_z




def coleta_dados_vbus(sep,relatorio=True):
    
   
    '''
    
    Coleta os Dados de Tensões e Limites Superiores e Inferiores das Barras do Sistema.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------
    sep : sistema elétrico de potência carregado pelo pandapower.
    relatorio : caso relatorio = True, retorna relatório informando, tensões, ângulos e limites.
                caso relatorio = False, retorna apenas as tensões, ângulos e limtes
    
    Retorno
    ----------
    vbus : vetor de tensões [pu] das barras em ordem crescente do número da barra
    theta : vetor de ângulo de tensões [°]
    v_lim_superior : 
    
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        
    '''
    
    sep.res_bus=sep.res_bus.sort_index()
    
    sep.bus=sep.bus.sort_index()
      
    vbus = sep.res_bus['vm_pu'].to_numpy(dtype=np.float64)
    
    theta = sep.res_bus['va_degree'].to_numpy(dtype=np.float64)
    
    v_lim_superior = sep.bus["max_vm_pu"].to_numpy(dtype=np.float32)
    
    v_lim_inferior = sep.bus["min_vm_pu"].to_numpy(dtype=np.float32)

    
    
    if relatorio==True:
        
        tabela = np.zeros((len(vbus),4))
        tabela[:,0] = vbus
        tabela[:,1] = theta
        tabela[:,2] = v_lim_inferior
        tabela[:,3] = v_lim_superior

        table = tabulate.tabulate(tabela, headers = ['Tensões nas Barras (pu)', 'Ângulos das Barras (°)','Limites Inferiores','Limites Superiores'], tablefmt="psql")
        print(table)
    
        sns.scatterplot(x=np.arange(0,len(vbus),1),y=vbus,color='blue',label='Módulo da Tensão',s=75)
        sns.lineplot(x=np.arange(0,len(vbus),1),y=v_lim_superior,color='red',label='Limite Superior',alpha=0.5)
        sns.lineplot(x=np.arange(0,len(vbus),1),y=v_lim_inferior,color='orange',label='Limite Inferior',alpha=0.5)
        plt.title('Módulo da Tensão por Barra do Sistema')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Tensão [pu]')
        plt.legend(loc='best')
        plt.figure(figsize=(16,10))
        sns.scatterplot(x=np.arange(0,len(theta),1),y=theta,color='green',label='Ângulo da Tensão',s=75)
        plt.title('Ângulo da Tensão por Barra do Sistema')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Theta [°]')
        plt.legend(loc='best')
        
    
    if relatorio==False:
        
        return vbus, theta, v_lim_superior, v_lim_inferior
    
    
    ##################################################################################################################################################################################


def coleta_dados_gen(sep,relatorio=True):
       
    '''
    
    Coleta os Dados de Tensões, Potências Ativa e Reativa e Seus Respectivos Limites Superiores e Inferiores de geração.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------
    sep : sistema elétrico de potência carregado pelo pandapower.
    relatorio : caso relatorio = True, retorna relatório informando, limites, potências e gráficos.
                caso relatorio = False, retorna apenas as tensões, ângulos, potências e limites.
    
    Retorno
    ----------
    vgen : vetor de tensões [pu] das barras de geração
    theta : vetor de ângulo de tensões [°] das barras de geração
    p_lim_superior : Limite Superior de Potência Ativa (pu)
    p_lim_inferior : Limite Inferior de Potência Ativa (pu)
    q_lim_superior : Limite Superior de Potência Reativa (pu)
    q_lim_inferior : Limite Inferior de Potência Ativa (pu)
    
    Observações:
    - - - - - - -
    
    Potência Aparente de Base : 100 MVA
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        
    '''
    
    sep.res_gen=sep.res_gen.sort_index()
    
    sep.gen=sep.gen.sort_index()
      
    vgen = sep.res_gen['vm_pu'].to_numpy(dtype=np.float64)
    
    barra = sep.gen['bus'].to_numpy(dtype=np.float64)
    
    thetagen = sep.res_gen['va_degree'].to_numpy(dtype=np.float64)
    
    pgen = sep.res_gen['p_mw'].to_numpy(dtype=np.float64)/100
    
    qgen = sep.res_gen['q_mvar'].to_numpy(dtype=np.float64)/100
    
    p_lim_superior = sep.gen["max_p_mw"].to_numpy(dtype=np.float32)/100
    
    p_lim_inferior = sep.gen["min_p_mw"].to_numpy(dtype=np.float32)/100
    
    q_lim_superior = sep.gen["max_q_mvar"].to_numpy(dtype=np.float32)/100
    
    q_lim_inferior = sep.gen["min_q_mvar"].to_numpy(dtype=np.float32)/100

    
    if relatorio==True:
        
        tabela = np.zeros((len(vgen),6))
        tabela[:,0] = pgen
        tabela[:,1] = p_lim_superior
        tabela[:,2] = p_lim_inferior
        tabela[:,3] = qgen
        tabela[:,4] = q_lim_superior
        tabela[:,5] = q_lim_inferior


        table = tabulate.tabulate(tabela, headers = ['P (pu)','P Lim. Sup. (pu)','P Lim. Inf. (pu)','Q (pu)','Q Lim. Sup. (pu)','Q Lim. Inf. (pu)'], tablefmt="psql")
        print(table)
    

        sns.scatterplot(x=barra,y=qgen,color='blue',label='Potência Gerada',s=75)
        sns.lineplot(x=barra,y=q_lim_superior,color='red',label='Limite Superior',alpha=0.5)
        sns.lineplot(x=barra,y=q_lim_inferior,color='orange',label='Limite Inferior',alpha=0.5)
        plt.title('Potência Reativa Gerada')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Potência Reativa (pu)')
        plt.legend(loc='best')
        
    
    if relatorio==False:
        
        return vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra
    
    

    ################################################################################################################################################################################################


def func_objetivo(vbarra,theta,condutancias,matriz_z,relatorio=True):
    

    '''

    Calcula as perdas nas linhas de transmissão de acordo com as tensões, ângulos das barras e condutâncias de linha.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    Parâmetros
    ----------
    vbarra : tensão da barra.
    theta : ângulo da barra.
    condutancias : matriz de condutâncias de linha (triângulo superior)

    caso relatorio = True, retorna relatório informando a matriz de perdas de linha e as perdas totais.
    caso relatorio = False, retorna apenas as perda em pu.
    Retorno
    ----------

    perdas : perdas de potência ativa em pu.


    Observações:
    - - - - - - -

    Potência Aparente de Base : 100 MVA
     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    '''
    
    matriz_v = np.zeros((len(vbarra),len(vbarra)), dtype=np.float64)
    
    matriz_theta = np.zeros((len(theta),len(theta)), dtype=np.float64)
    
    for barra in range(len(vbarra)):
        
        matriz_v[:,barra]=vbarra
        matriz_theta[:,barra]=theta
        
    
    soma_v = np.power(matriz_v,2) + np.power(matriz_v.T,2)
    
    subtrai_theta = matriz_theta - matriz_theta.T
    
    cosenotheta=np.cos(np.radians(subtrai_theta))
    
    produto = 2 * np.multiply(np.multiply(matriz_v, matriz_v.T),cosenotheta)
    
    matriz_perdas = np.multiply(condutancias,soma_v-produto) 

    perdas = np.multiply(matriz_z[4,:], np.power(matriz_z[5,:],2)+np.power(matriz_z[7,:],2)-2*np.multiply(np.multiply(matriz_z[5,:],matriz_z[7,:]),np.cos(np.radians(matriz_z[8,:]-matriz_z[6,:]))))
    perdas = np.sum(perdas)
    
    if relatorio == True:
        
        tabela = np.zeros((1,2))
        tabela[:,0] = perdas
        tabela[:,1] = perdas*100
        table = tabulate.tabulate(tabela, headers = ['Perdas Totais Nas Linhas (pu)','Perdas Totais Nas Linhas (MW)'], tablefmt="psql")
        print(table)
        
        if len(vbarra) ==14:
            plt.figure(figsize=(18,10))
            sns.heatmap(100*(matriz_perdas+matriz_perdas.T),annot=True,cmap="jet")
            plt.xlabel('Barra Origem')
            plt.ylabel('Barra Destino')
            plt.title('Matriz de Perdas de Linha Completa [MW]')

     
    else:
    
        return perdas

    

    ################################################################################################################################################################################################


def pen_tensao(vbus, limite_sup, limite_inf,relatorio=True):
    
    """    
    Calcula a parcela de penalização pura (sem constante de multiplicação) referente a violação dos limites de tensão.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------   
    vbus : tensões das barras do sistema elétrico.
    limite_sup : limite superior das tensões das barras do sistema elétrico.
    limite_inf : limite inferior das tensões das barras do sistema elétrico.
    
    limite_sup : tensões
    relatorio : caso relatorio = True, retorna penalização e nº de violações 
                caso relatorio = False, retorna apenas o valor de penalização.
    Retorno
    -------    
    penalizacao: somatório da diferença ao quadradado das tensões que ultrapassaram os limites inferiores ou superiores.
    
    Observações:
    ------------
    
    ...
    
    """
    
    
    inferior = vbus - limite_inf
    superior = limite_sup - vbus
    penalizacao = np.sum(np.power(superior[superior<0],2))+np.sum(np.power(inferior[inferior<0],2))
    
    if relatorio == True:
        
        print('Penalização de Tensão:\n')
        print(penalizacao,'\n')
        print('Número de Violações:\n')
        print(len(inferior[inferior<0])+len(superior[superior<0]))
     
    else:
    
        return penalizacao


 ################################################################################################################################################################################################


def pen_ger_reativo(q, limite_sup, limite_inf,sep,relatorio=True):
    
    """    
    Calcula a parcela de penalização pura (sem constante de multiplicação) referente a violação dos limites de geração de reativos.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------   
    q : potências reativas das barras de controle de reativo do sistema elétrico.
    limite_sup : limite superior das potências reativas das barras de controle de reativo do sistema elétrico.
    limite_inf : limite superior das potências reativas das barras de controle de reativo do sistema elétrico.
    
    limite_sup : tensões
    relatorio : caso relatorio = True, retorna penalização e nº de violações 
                caso relatorio = False, retorna apenas o valor de penalização.
    Retorno
    -------    
    penalização: somatório da diferença ao quadradado das potências reativas que ultrapassaram os limites inferiores ou superiores.
    
    Observações:
    ------------
    
    ...
    
    """
    
    inferior = limite_inf - q
    superior = limite_sup - q
    
    ext_sup = sep.ext_grid['max_q_mvar'].to_numpy()
    ext_inf = sep.ext_grid['min_q_mvar'].to_numpy()
    
    qext = sep.res_ext_grid['q_mvar'].to_numpy()
    
    inferiorext = ext_inf - qext
    superiorext =  ext_sup - qext
    
    penalizacaoext = np.sum(np.power(superiorext[superiorext<0],2))+np.sum(np.power(inferiorext[inferiorext>0],2))
    penalizacao = np.sum(np.power(superior[superior<0],2))+np.sum(np.power(inferior[inferior>0],2))
    
    
    if relatorio == True:
        
        print('Penalização de Geração de Reativos:\n')
        print(penalizacao+penalizacaoext,'\n')
        print('Número de Violações:\n')
        print(len(inferior[inferior<0])+len(superior[superior<0]))
        
    else:
    
        return penalizacao+penalizacaoext


 ################################################################################################################################################################################################


def coleta_dados_trafo(sep, relatorio=True):
    
    '''    
    
    
    Valores dos TAPs Retirados de:
    
    - REFORMULAÇÃO DAS RESTRIÇÕESDE COMPLEMENTARIDADE EM PROBLEMAS DE _ DE POTÊNCIA ÓTIMO
      Marina Valença Alencar - Dissertação de Mestrado

    - FUNÇÕES PENALIDADE PARA O TRATAMENTO DAS VARIÁVEIS DISCRETAS DO PROBLEMA DE FLUXO DE POTÊNCIA ÓTIMO REATIVO
      Daisy Paes Silva - Dissertação de Mestrado
    

    ''' 
    """    
    Coleta dados de Carregamento, Taps e Informações dos Transformadores do Sistema.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------   
    sep : sistema elétrico a ser estudado.
    
    relatorio : caso relatorio = True, retorna gráficos dos taps e carregamento com informações
                caso relatorio = False, retorna valores referentes aos taps.
    Retorno
    -------    
    tap_pos : posição do tap.
    tap_neutral : valor do tap neutral.
    tap_step_percent : valor do percentual de passo.
    valores_taps : valores dos taps programados para esse sistema.
    
    
    Observações:
    ------------
    
    ...
    
    """

    sep.trafo.sort_index()
   
    sep.res_trafo.sort_index()
    
    sep.trafo['tap_pos']=sep.trafo['tap_pos']
    
    n_trafos_controlados = sep.trafo['tap_pos'].count()
    
    carregamento = sep.res_trafo['loading_percent'].to_numpy()/100
    
    tap_pos = sep.trafo[~pd.isnull(sep.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)
    
    tap_neutral = sep.trafo[~pd.isnull(sep.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)
    
    tap_step_percent = sep.trafo[~pd.isnull(sep.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)
        
    
    if len(sep.bus)==14:
        
        step = 0.0075
        valores_taps = np.arange(start = 0.88, stop = 1.12, step = step)
        
        
    if len(sep.bus)==30:
        
        step = 0.0075
        valores_taps = np.arange(start = 0.88, stop = 1.12, step = step)
        
                
    if len(sep.bus)==118:
        
        step = 0.0075
        valores_taps = np.arange(start = 0.88, stop = 1.12, step = step)

                    
    if len(sep.bus)==300:
        
        step = 0.0075
        valores_taps = np.arange(start = 0.88, stop = 1.12, step = step)
        
        
    if relatorio == True:
        
    
        tap_pos = sep.trafo[~pd.isnull(sep.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)

        tap_neutral = sep.trafo[~pd.isnull(sep.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)

        tap_step_percent = sep.trafo[~pd.isnull(sep.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)
        
        valor_percentual= (tap_pos-tap_neutral)*(tap_step_percent/100) + 1
        
      
        if len(sep.bus)==14:

            plt.figure(figsize=(20,10))
            sns.scatterplot(x=np.arange(0,len(tap_pos)),y=valor_percentual,label='Valor do TAP',color='b',s=75)
            sns.lineplot(x=np.arange(0,len(tap_pos)),y=np.tile([1.12], (len(tap_pos))),label='Limite Máximo do TAP',color='r')
            sns.lineplot(x=np.arange(0,len(tap_pos)),y=np.tile([0.88], (len(tap_pos))),label='Limite Mínimo do TAP',color='orange')
            plt.grid()


        if len(sep.bus)==30:
        
            plt.figure(figsize=(20,10))
            sns.scatterplot(x=np.arange(0,len(tap_pos)),y=valor_percentual,label='Valor do TAP',color='b',s=75)
            sns.lineplot(x=np.arange(0,len(tap_pos)),y=np.tile([1.12], (len(tap_pos))),label='Limite Máximo do TAP',color='r')
            sns.lineplot(x=np.arange(0,len(tap_pos)),y=np.tile([0.88], (len(tap_pos))),label='Limite Mínimo do TAP',color='orange')
            plt.grid()
        
        if len(sep.bus)==118:
        
            plt.figure(figsize=(20,10))
            sns.scatterplot(x=np.arange(0,len(tap_pos)),y=valor_percentual,label='Valor do TAP',color='b',s=75)
            sns.lineplot(x=np.arange(0,len(tap_pos)),y=np.tile([1.12], (len(tap_pos))),label='Limite Máximo do TAP',color='r')
            sns.lineplot(x=np.arange(0,len(tap_pos)),y=np.tile([0.88], (len(tap_pos))),label='Limite Mínimo do TAP',color='orange')
            plt.grid()
        
        print('Carregamento do Trafo (pu):\n')
        print(carregamento,'\n')
        print('Número de Trafos com TAP Controlado:\n')
        print(n_trafos_controlados,'\n')
        print('Valores dos TAPs:\n')
        print(valor_percentual,'\n')
        
        
    else:
        
        return tap_pos, tap_neutral, tap_step_percent,valores_taps



 ################################################################################################################################################################################################


def pen_trafo(linha,n_tap,n_vgen):
    
    '''    
    
    
    Valores dos TAPs Retirados de:
    
    - REFORMULAÇÃO DAS RESTRIÇÕESDE COMPLEMENTARIDADE EM PROBLEMAS DE FLUXO DE POTÊNCIA ÓTIMO
      Marina Valença Alencar - Dissertação de Mestrado

    - FUNÇÕES PENALIDADE PARA O TRATAMENTO DAS VARIÁVEIS DISCRETAS DO PROBLEMA DE FLUXO DE POTÊNCIA ÓTIMO REATIVO
      Daisy Paes Silva - Dissertação de Mestrado
    

    ''' 
    """    
   
    Calcula a penalização senoidal para taps não discretos.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    Parâmetros
    ----------   
    linha : linha da partícula.
    n_tap : número de taps.
    n_vgen : número de geradores.
    
    Retorno
    -------    
    linha : linha da partícula com os valores da penalização do trafo atualizados.
    
    
    Observações:
    ------------
    
    ...
    
    """
    
    step = 0.0075

    linha[-3] = np.sum(np.square(np.sin((linha[n_vgen:n_vgen+n_tap]*np.pi/step))))
    
    return linha


 ################################################################################################################################################################################################


def coleta_dados_bshunt(sep):
    
    '''    
    
    
    Valores dos Shunt Retirados de:
    
    - REFORMULAÇÃO DAS RESTRIÇÕESDE COMPLEMENTARIDADE EM PROBLEMAS DE FLUXO DE POTÊNCIA ÓTIMO
      Marina Valença Alencar - Dissertação de Mestrado

    - FUNÇÕES PENALIDADE PARA O TRATAMENTO DAS VARIÁVEIS DISCRETAS DO PROBLEMA DE FLUXO DE POTÊNCIA ÓTIMO REATIVO
      Daisy Paes Silva - Dissertação de Mestrado
          

    ''' 
    
    ieee14 = np.arange(0.00,0.45,0.001)
    ieee30 = np.arange(0.00,0.35,0.001)
    ieee118 = np.arange(-0.20,0.30,0.001)
    
    bus = sep.shunt['bus'].sort_values().to_numpy()
    sep.shunt=sep.shunt.sort_index()
  
    
    if len(sep.bus)==14:
        
        bsh = np.array([[0,0.19,0.34,0.39]],dtype=object)
        
        
    if len(sep.bus)==30:
        
        bsh = np.array([[0,0.19,0.34,0.39],[0, 0.05, 0.09]],dtype=object)
        
                
    if len(sep.bus)==118:
        
        bsh = np.array([[-0.40, 0],
                       [0, 0.06, 0.07, 0.13, 0.14, 0.20],
                       [-0.25, 0],
                       [0, 0.10],
                       [0, 0.10],
                        [0, 0.10],
                        [0, 0.15],
                        [0.08, 0.12, 0.20],
                        [0, 0.10, 0.20],
                        [0, 0.10, 0.20],
                        [0, 0.10, 0.20],
                        [0, 0.10, 0.20],
                        [0, 0.06, 0.07, 0.13, 0.14, 0.20],
                        [0, 0.06, 0.07, 0.13, 0.14, 0.20]],dtype=object)
        
#         bsh = np.array([ieee118,
#                        ieee118,
#                        ieee118,
#                        ieee118,
#                        ieee118,
#                         ieee118,
#                         ieee118,
#                         ieee118,
#                         ieee118,
#                         ieee118,
#                         ieee118,
#                         ieee118,
#                         ieee118,
#                         ieee118],dtype=object)
        
              

    if len(sep.bus)==300:
        
        bsh = np.array([[0,2,3.5,4.5], #95
                [0, 0.25, 0.44, 0.59], #98
                [0,0.19,0.34,0.39], #132
                [-4.5,0], #142
                [-4.5,0], #144
                [0, 0.25,0.44,0.59], #151
                [0, 0.25,0.44,0.59], #157
                [-2.5,0], #168
                [-4.5,0], #209
                [-4.5,0],#216
                [-1.5,0], #218
                [0, 0.25, 0.44, 0.59], #226
                [0, 0.15], #267
                [0], #274
                [0], #276
                [0], #277
                [0], #278
                [0], #279
                [0], #280
                [0], #281
                [0,0.15], #282
                [0], #283
                [0], #285
                [0], #286
                [0], #287
                [0], #288
                [0], #296
                [0], #297
                [0], #299
               ],dtype=object)
    
    
    
    return bsh, bus


 ################################################################################################################################################################################################


def converte_trafo(tap_pos, tap_neutral, tap_step_percent,valores_taps):
    
    '''
    Converte TAPS conforme equação disponibilizada pelo pandapower.
    
    https://pandapower.readthedocs.io/en/v2.1.0/elements/trafo.html
    
    '''
    
    taps_convertido = tap_neutral + ((valores_taps - 1.0)*(100/tap_step_percent))
    
    
    return taps_convertido


 ################################################################################################################################################################################################


def cria_alcateia(sep,n_lobos):
    
    """"
    
    Cria a alcatéia de lobos cinzentos.
    
    linhas = partículas
    
    colunas = tensões geradores, tap transformadores, susceptâncias shunt, perdas, penalização de tensão, penalização de reativo, penalização de trafo, penalização shunt, fitness
   
    
    """
    
    
    vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra = coleta_dados_gen(sep,relatorio=False)
    
    n_vgen=len(vgen)
    
    vbus, theta, v_lim_superior, v_lim_inferior = coleta_dados_vbus(sep,relatorio=False)
    
    tap_pos, tap_neutral, tap_step_percent,valores_taps=coleta_dados_trafo(sep,relatorio=False)
    
    n_taps = len(tap_pos)
    
    bshunt , bus = coleta_dados_bshunt(sep)
    
    n_bshunt = len(bus)
    
    dimensao = n_taps + n_vgen + n_bshunt + 6
    
    alcateia=np.zeros((n_lobos,dimensao),dtype=np.float64)
    
    alcateia[:,0:n_vgen] = np.random.uniform(np.max(v_lim_inferior), np.max(v_lim_superior), size=(n_lobos,n_vgen))
    
    alcateia[:,n_vgen:n_vgen+n_taps]=np.random.choice(valores_taps, size =(n_lobos, n_taps))
    
    i=1
    
    i=1
    
    for bsh in bshunt:
        
        alcateia[:,n_vgen+n_taps+i-1:n_vgen+n_taps+i] = np.random.choice(bsh, size =(n_lobos, 1))
        i=i+1

    return alcateia
    
    
    


 ################################################################################################################################################################################################


def cria_enxame(sep,n_particulas):
    
    """"
    
    Cria o enxame de partículas.
    
    
    linhas = partículas
    
    colunas = tensões geradores, tap transformadores, susceptâncias shunt, perdas, penalização de tensão, penalização de reativo, penalização de trafo, penalização shunt, fitness
    
    """
    
    
    vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra = coleta_dados_gen(sep,relatorio=False)
    
    n_vgen=len(vgen)+1
    
    vbus, theta, v_lim_superior, v_lim_inferior = coleta_dados_vbus(sep,relatorio=False)
    
    tap_pos, tap_neutral, tap_step_percent,valores_taps=coleta_dados_trafo(sep,relatorio=False)
    
    n_taps = len(tap_pos)
    
    bshunt , bus = coleta_dados_bshunt(sep)
    
    bshunt = np.array(bshunt)
    
    n_bshunt = len(bus)
    
    dimensao = n_taps + n_vgen + n_bshunt + 6 
    
    enxame=np.zeros((n_particulas,dimensao),dtype=np.float64)
    
    enxame[:,0:n_vgen] = np.random.uniform(np.max(v_lim_inferior), np.max(v_lim_superior), size=(n_particulas,n_vgen))
    
    enxame[:,n_vgen:n_vgen+n_taps]=np.random.choice(valores_taps, size =(n_particulas, n_taps))
    
    i=1
    
    for bsh in bshunt:
        
        enxame[:,n_vgen+n_taps+i-1:n_vgen+n_taps+i] = np.random.choice(bsh, size =(n_particulas, 1))
        i=i+1
        
    return enxame


def cria_enxame_v(sep,n_particulas):
    
    """"
    
    Cria o enxame de partículas.
    
    
    linhas = partículas
    
    colunas = tensões geradores, tap transformadores, susceptâncias shunt, perdas, penalização de tensão, penalização de reativo, penalização de trafo, penalização shunt, fitness
    
    """
    
    
    vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra = coleta_dados_gen(sep,relatorio=False)
    
    n_vgen=len(vgen)+1
    
    vbus, theta, v_lim_superior, v_lim_inferior = coleta_dados_vbus(sep,relatorio=False)
    tap_pos, tap_neutral, tap_step_percent,valores_taps=coleta_dados_trafo(sep,relatorio=False)
    
    n_taps = len(tap_pos)
    
    bshunt , bus = coleta_dados_bshunt(sep)
    
    bshunt = np.array(bshunt)
    
    n_bshunt = len(bus)
    
    dimensao = n_taps + n_vgen + n_bshunt + 6
    
    enxame=np.zeros((n_particulas,dimensao),dtype=np.float64)
    
    enxame[:,0:n_vgen] = np.random.uniform(-1*np.ones(n_vgen)*np.max(v_lim_superior), np.ones(n_vgen)*np.max(v_lim_superior), size=(n_particulas,n_vgen))
    
    enxame[:,n_vgen:n_vgen+n_taps]= np.random.uniform(-1*np.ones(n_taps)*np.max(valores_taps), np.ones(n_taps)*np.max(valores_taps), size =(n_particulas, n_taps))
    
    i=1
    
    for bsh in bshunt:
        
        enxame[:,n_vgen+n_taps+i-1:n_vgen+n_taps+i] = np.random.uniform(-1*np.ones(1)*np.max(bsh),np.ones(1)*np.max(bsh),size =(n_particulas, 1))
        i=i+1
        
    return enxame

 ################################################################################################################################################################################################

def pen_bshunt(grupo,n_tap,n_vgen,n_bshunt,sep):
    
    b = grupo[n_tap+n_vgen:n_tap+n_vgen+n_bshunt]
    
    bsh,bus=coleta_dados_bshunt(sep)
    
    penal = 0
    
    i=0

    bs=[]

    for i in range(len(bsh)):
    
        bs.append(np.array(bsh[i]))
    
    for i in range(len(bs)):
                
        if len(bs[i][bs[i]<=b[i]])==0:
            penal=1
            return penal
        if len(bs[i][bs[i]>=b[i]])==0:
            penal=1
            return penal
            
        anterior = bs[i][bs[i]<=b[i]][-1]
        posterior = bs[i][bs[i]>=b[i]][0]
        alfa = np.pi*(np.ceil((anterior/(0.001+posterior-anterior)))-(anterior/(0.001+posterior-anterior)))
        penal = penal + np.square(np.sin((b[i]/(0.001+posterior-anterior))*np.pi+alfa))

    
    return penal    
    



 ################################################################################################################################################################################################


def fluxo_de_pot(grupo, sep):
    
    n_bshunt = len(sep.shunt)
    n_vgen = len(sep.gen)+1
    n_tap = np.abs(sep.trafo['tap_pos']).count()
    
    matrizg = matriz_condutancia(sep,relatorio=False)
    
    for linha in range(grupo.shape[0]):
        
        sep.ext_grid['vm_pu']=grupo[linha,0]
        
        sep.gen['vm_pu']=grupo[linha,1:n_vgen]
        
        tap_pos, tap_neutral, tap_step_percent,valores_taps=coleta_dados_trafo(sep,relatorio=False)
        
        sep.trafo['tap_pos'][~pd.isnull(sep.trafo['tap_pos'])]=converte_trafo(tap_pos, tap_neutral, tap_step_percent,grupo[linha,n_vgen:n_vgen+n_tap])
        
        sep.shunt['q_mvar']=grupo[linha,n_vgen+n_tap:n_vgen+n_tap+n_bshunt]*-100
        
        if len(sep.bus)==300:
        
            pp.runpp(sep,algorithm='fdbx',numba=True, init = 'results', tolerance_mva = 1e-4,max_iteration=1000)
        
        else:
        
            pp.runpp(sep,algorithm='nr',numba=True, init = 'flat', tolerance_mva = 1e-5,max_iteration=100,enforce_q_lims=False,trafo_model='pi')
        
        vbus, theta, v_lim_superior, v_lim_inferior=coleta_dados_vbus(sep,relatorio=False)
        
        grupo[linha,-6] = sep.res_line['pl_mw'].sum()/100 + sep.res_trafo['pl_mw'].sum()/100 #func_objetivo(vbus,theta,matrizg,relatorio=False)

        grupo[linha,-5] = pen_tensao(vbus, v_lim_superior, v_lim_inferior,relatorio=False)
        
        vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra = coleta_dados_gen(sep,relatorio=False)
        
        grupo[linha,-4] = pen_ger_reativo(qgen, q_lim_superior, q_lim_inferior,sep,relatorio=False)
        
        grupo[linha,:] = pen_trafo(grupo[linha,:],n_tap,n_vgen)
        
        
        grupo[linha,-2] = pen_bshunt(grupo[linha,:],n_tap,n_vgen,n_bshunt,sep)
  
        
        
    
    return grupo
    

def fluxo_de_pot_q(grupo, sep):
    
    n_bshunt = len(sep.shunt)
    n_vgen = len(sep.gen)+1
    n_tap = np.abs(sep.trafo['tap_pos']).count()
    
    matrizg = matriz_condutancia(sep,relatorio=False)
    
    for linha in range(grupo.shape[0]):
        
        sep.ext_grid['vm_pu']=grupo[linha,0]
        
        sep.gen['vm_pu']=grupo[linha,1:n_vgen]
        
        tap_pos, tap_neutral, tap_step_percent,valores_taps=coleta_dados_trafo(sep,relatorio=False)
        
        sep.trafo['tap_pos'][~pd.isnull(sep.trafo['tap_pos'])]=converte_trafo(tap_pos, tap_neutral, tap_step_percent,grupo[linha,n_vgen:n_vgen+n_tap])
        
        sep.shunt['q_mvar']=grupo[linha,n_vgen+n_tap:n_vgen+n_tap+n_bshunt]*-100
        
        if len(sep.bus)==300:
        
            pp.runpp(sep,algorithm='fdbx',numba=True, init = 'results', tolerance_mva = 1e-4,max_iteration=1000)
        
        else:
        
            pp.runpp(sep,algorithm='nr',numba=True, init = 'flat', tolerance_mva = 1e-5,max_iteration=1000,enforce_q_lims=True,trafo_model='pi')
        
        vbus, theta, v_lim_superior, v_lim_inferior=coleta_dados_vbus(sep,relatorio=False)
        
        grupo[linha,-6] = sep.res_line['pl_mw'].sum()/100 + sep.res_trafo['pl_mw'].sum()/100 #func_objetivo(vbus,theta,matrizg,relatorio=False)

        grupo[linha,-5] = pen_tensao(vbus, v_lim_superior, v_lim_inferior,relatorio=False)
        
        vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra = coleta_dados_gen(sep,relatorio=False)
        
        grupo[linha,-4] = pen_ger_reativo(qgen, q_lim_superior, q_lim_inferior,sep,relatorio=False)
        
        grupo[linha,:] = pen_trafo(grupo[linha,:],n_tap,n_vgen)
        
        
        grupo[linha,-2] = pen_bshunt(grupo[linha,:],n_tap,n_vgen,n_bshunt,sep)
  
        
        
    
    return grupo
    



 ################################################################################################################################################################################################


def fitness (grupo,zeta,psi,sigma,omega):
    
# fitness J       perdas         pen tensão         pen q mvar          pen trafo           pen bshunt       
    grupo[:,-1]=(grupo[:,-6])+(zeta*grupo[:,-5])+(psi*grupo[:,-4])+(sigma*grupo[:,-3])+(omega*grupo[:,-2])

    return grupo


 ################################################################################################################################################################################################


def validacao (sep, best_solution,relatorio=True):
       
    valida = fluxo_de_pot(np.array([best_solution]), sep)
    
    if relatorio == True:
        print('Perdas de Potência Ativa [PU]:\n')
        print(valida[0][-6])
        print(' ')

        print('Penalização de Violação de Tensão [PU]:\n')
        print(valida[0][-5])
        print(' ')

        print('Penalização de Violação de Geração de Reativo [PU]:\n')
        print(valida[0][-4])
        print(' ')

        print('Penalização de Violação de TAP Discreto [PU]:\n')
        print(valida[0][-3])
        print(' ')

        print('Penalização de Violação de Bshunt Discreto [PU]:\n')
        print(valida[0][-2])
        print(' ')
        


def validacao_q (sep, best_solution,relatorio=True):
       
    valida = fluxo_de_pot_q(np.array([best_solution]), sep)
    
    if relatorio == True:
        print('Perdas de Potência Ativa [PU]:\n')
        print(valida[0][-6])
        print(' ')

        print('Penalização de Violação de Tensão [PU]:\n')
        print(valida[0][-5])
        print(' ')

        print('Penalização de Violação de Geração de Reativo [PU]:\n')
        print(valida[0][-4])
        print(' ')

        print('Penalização de Violação de TAP Discreto [PU]:\n')
        print(valida[0][-3])
        print(' ')

        print('Penalização de Violação de Bshunt Discreto [PU]:\n')
        print(valida[0][-2])
        print(' ')
        



 ################################################################################################################################################################################################


def otimizacao_gwo_continuo(sep, zeta, psi, sigma, omega, max_iter, n_lobos,valor_inicial,relatorio=True,inicial=True):
        
    alcateia_fit = cria_alcateia(sep,n_lobos)

    if inicial == True:
        
        alcateia_fit[0,:]=valor_inicial
    
    
    j = []
    
    perdas = []
    
    tempo = []
    
    pen_v = []
    
    pen_gq = []
    
    pen_tap = []
    
    pen_bsh = []

    
    v_lim_superior = np.repeat(sep.bus['max_vm_pu'][0], len(sep.gen))
    v_lim_inferior = np.repeat(sep.bus['min_vm_pu'][0], len(sep.gen))
    
    tap_pos, tap_neutral, tap_step_percent,valores_taps = coleta_dados_trafo(sep,relatorio=False)
    
    tap_max = np.repeat(valores_taps[-1], len(tap_pos))
    
    tap_min = np.repeat(valores_taps[0], len(tap_pos))
    
    bsh,b=coleta_dados_bshunt(sep)

    bsh_max=[]
    
    bsh_min=[]
    
    alcateias = []
    
    for bs in bsh:
        bsh_max.append([np.max(bs)])
        bsh_min.append([np.min(bs)])


    maximo = np.expand_dims(np.concatenate((v_lim_superior, tap_max, bsh_max), axis = None), 0)
    minimo = np.expand_dims(np.concatenate((v_lim_inferior, tap_min, bsh_min), axis = None), 0)
     
    
    lim_sup = np.tile(maximo, (n_lobos,1))
    lim_inf = np.tile(minimo, (n_lobos,1))
    
    
    for i in range(0,max_iter):

        start = time.time()
       
        alcateia_fit = fluxo_de_pot(alcateia_fit,sep)
        
        alcateia_fit = fitness(alcateia_fit,zeta,psi,sigma,omega)

        alcateia_fit = alcateia_fit[np.argsort(alcateia_fit[:, -1])]
        
        a = (2 - (i*(2/max_iter)))
        
        mu = 0.5
        sigma = 0.15
        
        r1 = np.random.normal(mu, sigma, size = (n_lobos,alcateia_fit[:,0:-6].shape[1]))
        r2 = np.random.normal(mu, sigma, size = (n_lobos,alcateia_fit[:,0:-6].shape[1]))
        
        #r1 = np.random.random_sample(size = (n_lobos,alcateia_fit[:,0:-6].shape[1]))
        
        #r2 = np.random.random_sample(size = (n_lobos,alcateia_fit[:,0:-6].shape[1]))

        A = (2*a*r1) - a
        
        C = 2*r2
        
        if (i == 0):
        
            lobo_alfa = alcateia_fit[0, :].copy()
            lobo_beta = alcateia_fit[1, :].copy()
            lobo_delta = alcateia_fit[2, :].copy()
            
            alfa = np.expand_dims(alcateia_fit[0,0:-6].copy(), 1)
            beta = np.expand_dims(alcateia_fit[1,0:-6].copy(), 1)
            delta = np.expand_dims(alcateia_fit[2,0:-6].copy(), 1)
            
        
        for t in range(3):

            if (alcateia_fit[t, -1] < lobo_alfa[-1]):

                lobo_alfa = alcateia_fit[0,:].copy()
                    
                alcateias.append(alcateia_fit)

                alfa = np.expand_dims(alcateia_fit[1,0:-6].copy(), 1)

            if (alcateia_fit[t,-1] > lobo_alfa[-1] and alcateia_fit[t,-1] < lobo_beta[-1]):

                lobo_beta = alcateia_fit[1,:].copy()

                beta = np.expand_dims(alcateia_fit[1,0:-6].copy(), 1)

            if (alcateia_fit[t,-1] > lobo_alfa[-1] and alcateia_fit[t,-1] > lobo_beta[-1] and alcateia_fit[t,-1] < lobo_delta[-1]):

                lobo_delta = alcateia_fit[2, :].copy()

                delta = np.expand_dims(alcateia_fit[2,0:-6].copy(), 1)         
        

        d_alfa = np.abs(np.multiply(C, alfa.T) - alcateia_fit[:, 0:-6])*0.1

        d_beta = np.abs(np.multiply(C, beta.T) - alcateia_fit[:, 0:-6])*0.1

        d_delta = np.abs(np.multiply(C, delta.T) - alcateia_fit[:, 0:-6])*0.1

        x_alfa = alfa.T - np.multiply(A, d_alfa)

        x_beta = beta.T - np.multiply(A, d_beta)

        x_delta = delta.T - np.multiply(A, d_delta)

        alcateia_fit[:,0:-6] = (x_alfa + x_beta + x_delta)/3

        alca_estat = alcateia_fit[:,-6:]

        alcateia_fit = np.concatenate(( np.clip(alcateia_fit[:,0:-6], a_min = lim_inf, a_max = lim_sup, out = alcateia_fit[:,0:-6]),alca_estat),axis=1)
        
        
        end = time.time()

        elapsed = end - start
        
        j.append(lobo_alfa[-1])

        perdas.append(lobo_alfa[-6])

        pen_v.append(lobo_alfa[-5])

        pen_gq.append(lobo_alfa[-4])

        pen_tap.append(lobo_alfa[-3])

        pen_bsh.append(lobo_alfa[-2])       
        
        
        tempo.append(elapsed)
        
        if relatorio == True:
            
            print(' ')

            print('Lobo Alfa da Iteração:',i)

            print('Perdas (pu):',lobo_alfa[-6])

            print('Penalização de Tensão:',lobo_alfa[-5])

            print('Penalização de Geração de Reativo:',lobo_alfa[-4])

            print('Penalização do Tap:',lobo_alfa[-3])

            print('Penalização do Bshunt:',lobo_alfa[-2])

            print('Fitness:',lobo_alfa[-1])
            
            print('Tempo: ',elapsed)

            print(' ')

            print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
            
    if relatorio == True:
        
            plt.figure(figsize=(18,10))
            plt.plot(perdas)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Perdas de Potência Ativa (pu)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(j)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Fitness (J)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_v)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Penalização de Tensão')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_gq)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Penalização de Geração de Reativo')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_tap)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Penalização do TAP')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_bsh)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Penalização do BShunt')
            plt.xlabel('Número da Iteração')

    return j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,alcateias,lobo_alfa, lobo_beta, lobo_delta, tempo

    


 ################################################################################################################################################################################################

def otimizacao_pso_continuo(sep, zeta, psi, sigma, omega, max_iter, n_particles,c1,c2,v_amp,valor_inicial,relatorio=True,inicial=True):
        
    enxame_fit = cria_enxame(sep,n_particles)
    
    if inicial == True:
        
        enxame_fit[0,:]=valor_inicial
        
    
    w_max=0.9
    w_min=0.4
    
    j = []
    
    tempo = []
        
    perdas = []
    
    pen_v = []
    
    pen_gq = []
    
    pen_tap = []
    
    pen_bsh = []

    
    v_lim_superior = np.repeat(sep.bus['max_vm_pu'][0], len(sep.gen))
    
    v_lim_inferior = np.repeat(sep.bus['min_vm_pu'][0], len(sep.gen))
    
    tap_pos, tap_neutral, tap_step_percent,valores_taps = coleta_dados_trafo(sep,relatorio=False)
    
    tap_max = np.repeat(valores_taps[-1], len(tap_pos))
    
    tap_min = np.repeat(valores_taps[0], len(tap_pos))
    
    bsh,b=coleta_dados_bshunt(sep)

    bsh_max=[]
    
    bsh_min=[]
    

    for bs in bsh:
        bsh_max.append([np.max(bs)])
        bsh_min.append([np.min(bs)])


    maximo = np.expand_dims(np.concatenate((v_lim_superior, tap_max, bsh_max), axis = None), 0)
    minimo = np.expand_dims(np.concatenate((v_lim_inferior, tap_min, bsh_min), axis = None), 0)
     
    
    lim_sup = np.tile(maximo, (n_particles,1))
    lim_inf = np.tile(minimo, (n_particles,1))
    
    v_anterior = v_amp*cria_enxame(sep,n_particles)

    for i in range(0,max_iter):
        
        start = time.time()
        
        mu, sigma = 0.5, 0.152 # mean and standard deviation

        #r1 = np.random.normal(mu, sigma, size = (n_particles,enxame_fit.shape[1]))
        #r2 = np.random.normal(mu, sigma, size = (n_particles,enxame_fit.shape[1]))
        
        r1 = np.random.random_sample(size = (n_particles,1))
        
        r2 = np.random.random_sample(size = (n_particles,1))
    
       
        enxame_fit = fluxo_de_pot(enxame_fit,sep)
        


        enxame_fit = fitness(enxame_fit,zeta,psi,sigma,omega)

        if i==0:
            
            best_particles = enxame_fit.copy()

            global_best = best_particles[np.argsort(best_particles[:, -1])][0,:].copy()
            

            
           
        for t in range(0,n_particles):
                
            if enxame_fit[t,-1] < best_particles[t,-1]:
                    
                best_particles[t,:] = enxame_fit[t,:].copy()
                    

        global_best = best_particles[np.argsort(best_particles[:, -1])][0,:].copy()
            
        global_matriz = np.tile(global_best, (n_particles,1))   
            
        
        enxame_fit_anterior = enxame_fit.copy()
        
        w_novo = w_max-(w_max-w_min)*(i+1)/max_iter
        v_novo = np.multiply(w_novo,v_anterior.copy()) + c1*np.multiply(r1,(best_particles.copy()-enxame_fit.copy())) + c2*np.multiply(r2,(global_matriz.copy()-enxame_fit.copy()))
        
        enxame_fit_novo = enxame_fit_anterior  + v_novo
        
        v_anterior = v_novo.copy()
        
        enxame_estat = enxame_fit_novo[:,-6:]

        enxame_fit = np.concatenate(( np.clip(enxame_fit_novo[:,0:-6], a_min = lim_inf, a_max = lim_sup, out = enxame_fit_novo[:,0:-6]),enxame_estat),axis=1)   


        
        end = time.time()

        elapsed = end - start

        j.append(global_best[-1])

        perdas.append(global_best[-6])

        pen_v.append(global_best[-5])

        pen_gq.append(global_best[-4])

        pen_tap.append(global_best[-3])

        pen_bsh.append(global_best[-2])
        
        tempo.append(elapsed)
      

        if relatorio == True:
            
            print(' ')

            print('Melhor Global da Iteração:',i)

            print('Perdas (pu):', global_best[-6])

            print('Penalização de Tensão:', global_best[-5])

            print('Penalização de Geração de Reativo:', global_best[-4])

            print('Penalização do Tap:', global_best[-3])

            print('Penalização do Bshunt:', global_best[-2])

            print('Fitness:', global_best[-1])
            
            print('Tempo: ', elapsed)

            print(' ')

            print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
            
            
    
    if relatorio == True:
        
            plt.figure(figsize=(18,10))
            plt.plot(perdas)
            plt.grid()
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Perdas de Potência Ativa (pu)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(j)
            plt.grid()
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Fitness (J)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_v)
            plt.grid()
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização de Tensão')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_gq)
            plt.grid()
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização de Geração de Reativo')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_tap)
            plt.grid()
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização do TAP')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_bsh)
            plt.grid()
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização do BShunt')
            plt.xlabel('Número da Iteração')
                       
            
    return j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,global_best, tempo

      



 ################################################################################################################################################################################################


def discreto_bshunt(grupo,n_tap,n_vgen,n_bshunt,sep):
    
    b = grupo[n_tap+n_vgen:n_tap+n_vgen+n_bshunt]
    
    bsh,bus=coleta_dados_bshunt(sep)
    
    penal = 0
    
    discretiza = []
    
    i = 0

    bs = []

    for i in range(len(bsh)):
    
        bs.append(np.array(bsh[i]))
    
    i = 0
    
    for c in bs:
                
        discretiza.append(c[np.argmin(np.abs(c-b[i]))])
        
        i=i+1    
        
    
    return discretiza    
    



 ################################################################################################################################################################################################



def discreto_tap(grupo,n_tap,n_vgen,n_bshunt,sep):
    
    b = grupo[n_vgen:n_vgen+n_tap]
    
    ref = np.arange(0.88,1.12,0.0075)
    
    discretizatap = np.zeros(len(b))
    
    i = 0

    
    for i in range(len(b)):
        
        discretizatap[i]=(ref[np.argmin(np.abs(ref-b[i]))])
                   
    return discretizatap
    
    




 ################################################################################################################################################################################################

def otimizacao_pso_discreto(sep, zeta, psi, sigma, omega, max_iter, n_particles,c1,c2,v_amp,valor_inicial,step, wmax,relatorio=True,inicial=True,):
        
    enxame_fit = cria_enxame(sep,n_particles)
    
        
    if len(sep.bus) == 14:
        
        n_vgen = 4+1
        n_tap = 3
        n_bshunt = 1
    
    if len(sep.bus) == 30:
        
        n_vgen = 5+1
        n_tap = 4
        n_bshunt = 2
        
    
    if len(sep.bus) == 118:
        
        n_vgen = 53+1
        n_tap = 9
        n_bshunt = 14
        
        
    if len(sep.bus) == 300:
        
        n_vgen = 68+1
        n_tap = 62
        n_bshunt = 29
    
    
#     for linha in range(enxame_fit.shape[0]):
        
#         enxame_fit[linha,0:len(sep14.gen)+1] = valor_inicial[0:len(sep14.gen)+1] + (1.05-0.95) *(np.random.randn(len(sep14.gen)+1)/3)*step

#         enxame_fit[linha,0:len(sep14.gen)+1][
#         enxame_fit[linha,0:len(sep14.gen)+1]>1.05] = 1.05

#         enxame_fit[linha,0:len(sep14.gen)+1][
#         enxame_fit[linha,0:len(sep14.gen)+1]<0.95] = 0.95

#         enxame_fit[linha,len(sep14.gen)+1:len(sep14.gen)+1+n_tap] = valor_inicial[len(sep14.gen)+1:len(sep14.gen)+1+n_tap] + (1.12-0.88) * (np.random.randn(n_tap)/3 )*step


#         enxame_fit[linha,len(sep14.gen)+1:len(sep14.gen)+1+n_tap][
#         enxame_fit[linha,len(sep14.gen)+1:len(sep14.gen)+1+n_tap]<0.88] = 0.88

#         enxame_fit[linha,len(sep14.gen)+1:len(sep14.gen)+1+n_tap][
#         enxame_fit[linha,len(sep14.gen)+1:len(sep14.gen)+1+n_tap]>1.12] = 1.12

#         enxame_fit[linha,len(sep14.gen)+1:len(sep14.gen)+1+n_tap] = discreto_tap(enxame_fit[linha,:],n_tap,n_vgen,n_bshunt,sep14)


    if inicial == True:
        
        enxame_fit[0,:]=valor_inicial     
        
    
    w_max=wmax
    
    w_min=0.4
    
    
    j = []
    
    
    tempo = []
        
    perdas = []
    
    pen_v = []
    
    pen_gq = []
    
    pen_tap = []
    
    pen_bsh = []

    
    v_lim_superior = np.repeat(sep.bus['max_vm_pu'][0], n_vgen)
    
    v_lim_inferior = np.repeat(sep.bus['min_vm_pu'][0], n_vgen)
    
    tap_pos, tap_neutral, tap_step_percent,valores_taps = coleta_dados_trafo(sep,relatorio=False)
    
    tap_max = np.repeat(valores_taps[-1], len(tap_pos))
    
    tap_min = np.repeat(valores_taps[0], len(tap_pos))
    
    bsh,b=coleta_dados_bshunt(sep)

    bsh_max=[]
    
    bsh_min=[]
    

    for bs in bsh:
        bsh_max.append([np.max(bs)])
        bsh_min.append([np.min(bs)])


    maximo = np.expand_dims(np.concatenate((v_lim_superior, tap_max, bsh_max), axis = None), 0)
    minimo = np.expand_dims(np.concatenate((v_lim_inferior, tap_min, bsh_min), axis = None), 0)
     
    
    lim_sup = np.tile(maximo, (n_particles,1))
    lim_inf = np.tile(minimo, (n_particles,1))
    
    v_anterior = v_amp*cria_enxame(sep,n_particles)

    for i in range(0,max_iter):
        
        start = time.time()
        
#         mu, sigmma = 0.5, 0.152 # mean and standard deviation

#         r1 = np.random.normal(mu, sigmma, size = (n_particles,enxame_fit.shape[1]))
#         r2 = np.random.normal(mu, sigmma, size = (n_particles,enxame_fit.shape[1]))
        
        r1 = np.random.random_sample(size = (n_particles,1))
        
        r2 = np.random.random_sample(size = (n_particles,1))

#         r1 = np.random.uniform(0,1)
#         r2 = np.random.uniform(0,1)

        enxame_fit_d = np.copy(enxame_fit)
    
        for linha in range(n_particles):
          
            enxame_fit_d[linha][n_vgen:n_vgen+n_tap] = discreto_tap(enxame_fit[linha],n_tap,n_vgen,n_bshunt,sep)
            enxame_fit_d[linha][n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(enxame_fit[linha],n_tap,n_vgen,n_bshunt,sep)
  
       
        enxame_fit[:,-6:] = (fluxo_de_pot(enxame_fit_d,sep))[:,-6:]
        
        enxame_fit[:,-6:] = (fitness(enxame_fit,zeta,psi,sigma,omega))[:,-6:]

        if i==0:
            
            best_particles = enxame_fit.copy()

            global_best = best_particles[np.argsort(best_particles[:, -1])][0,:].copy()
            
            global_matriz = np.tile(global_best, (n_particles,1))
        
           
        for t in range(0,n_particles):
                
            if (enxame_fit[t,-1] < best_particles[t,-1]):
                    
                best_particles[t,:] = enxame_fit[t,:].copy()
                    

        global_best = best_particles[np.argsort(best_particles[:, -1])][0,:].copy()
            
        global_matriz = np.tile(global_best, (n_particles,1))   
            
        enxame_fit_anterior = enxame_fit.copy()
        w_novo = w_max-(w_max-w_min)*(i+1)/max_iter

#         w_novo = w_max*np.exp(-2*((i/max_iter)))

#             print(w_novo)

        fi = c2+c1

    
        constri = 2/np.abs(2-fi - np.sqrt(fi**2-4*fi))
        
        media = np.mean(enxame_fit[:,:-6],axis=0)
    
        fator = np.sum(np.sqrt((np.sum(np.power(enxame_fit[:,:-6]-media,2)))))/n_particles
        
        print('Peso de Inércia', w_novo)
        print('Distância Euclideana', fator)
        print('Fator de Constrição', constri)
          
    
        v_novo = np.multiply(w_novo,v_anterior.copy()) + c1*np.multiply(r1,(best_particles.copy()-enxame_fit.copy())) + c2*np.multiply(r2,(global_matriz.copy()-enxame_fit.copy()))
        
        
 
        
        v_novo[v_novo>0.0075] = 0.0075
        v_novo[v_novo<-0.0075] = -0.0075
        
        enxame_fit_novo = enxame_fit_anterior  + v_novo*constri
        
        v_anterior = v_novo.copy()
        
        
#         for linha in range(n_particles):
          
#             enxame_fit_novo[linha][n_vgen:n_vgen+n_tap] = discreto_tap(enxame_fit_novo[linha],n_tap,n_vgen,n_bshunt,sep)
#             enxame_fit_novo[linha][n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(enxame_fit_novo[linha],n_tap,n_vgen,n_bshunt,sep)
            
            
        enxame_estat = enxame_fit_novo[:,-6:]

        enxame_fit = np.concatenate(( np.clip(enxame_fit_novo[:,0:-6], a_min = lim_inf, a_max = lim_sup, out = enxame_fit_novo[:,0:-6]),enxame_estat),axis=1)   

        end = time.time()

        elapsed = end - start

        j.append(global_best[-1])

        perdas.append(global_best[-6])

        pen_v.append(global_best[-5])

        pen_gq.append(global_best[-4])

        pen_tap.append(global_best[-3])

        pen_bsh.append(global_best[-2])
        
        tempo.append(elapsed)
      

        if relatorio == True:
            
            print(' ')

            print('Melhor Global da Iteração:',i)

            print('Perdas (pu):', global_best[-6])

            print('Penalização de Tensão:', global_best[-5])

            print('Penalização de Geração de Reativo:', global_best[-4])

            print('Penalização do Tap:', global_best[-3])

            print('Penalização do Bshunt:', global_best[-2])

            print('Fitness:', global_best[-1])
            
            print('Tempo: ', elapsed)

            print(' ')

            print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
            
            
    
    if relatorio == True:
        
            plt.figure(figsize=(18,10))
            plt.plot(perdas)
            
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Perdas de Potência Ativa (pu)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(j)
            
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Fitness (J)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_v)
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização de Tensão')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_gq)
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização de Geração de Reativo')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_tap)
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização do TAP')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_bsh)
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização do BShunt')
            plt.xlabel('Número da Iteração')
                       
            
    return j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,global_best, tempo


 ################################################################################################################################################################################################


def otimizacao_gwo_discreto(sep, zeta, psi, sigma, omega, max_iter, n_lobos,valor_inicial,relatorio=True,inicial=True):
        
    alcateia_fit = cria_alcateia(sep,n_lobos)

    if inicial == True:
        
        alcateia_fit[0,:]=valor_inicial
        
    if len(sep.bus) == 14:
        
        n_vgen = 4
        n_tap = 3
        n_bshunt = 1
    
    if len(sep.bus) == 30:
        
        n_vgen = 5
        n_tap = 4
        n_bshunt = 2
        
    
    if len(sep.bus) == 118:
        
        n_vgen = 53
        n_tap = 9
        n_bshunt = 14
        
        
    if len(sep.bus) == 300:
        
        n_vgen = 68
        n_tap = 62
        n_bshunt = 29
    
    
    j = []
    
    perdas = []
    
    tempo = []
    
    pen_v = []
    
    pen_gq = []
    
    pen_tap = []
    
    pen_bsh = []

    
    v_lim_superior = np.repeat(sep.bus['max_vm_pu'][0], len(sep.gen))
    v_lim_inferior = np.repeat(sep.bus['min_vm_pu'][0], len(sep.gen))
    
    tap_pos, tap_neutral, tap_step_percent,valores_taps = coleta_dados_trafo(sep,relatorio=False)
    
    tap_max = np.repeat(valores_taps[-1], len(tap_pos))
    
    tap_min = np.repeat(valores_taps[0], len(tap_pos))
    
    bsh,b=coleta_dados_bshunt(sep)

    bsh_max=[]
    
    bsh_min=[]
    
    alcateias = []
    
    for bs in bsh:
        bsh_max.append([np.max(bs)])
        bsh_min.append([np.min(bs)])


    maximo = np.expand_dims(np.concatenate((v_lim_superior, tap_max, bsh_max), axis = None), 0)
    minimo = np.expand_dims(np.concatenate((v_lim_inferior, tap_min, bsh_min), axis = None), 0)
     
    
    lim_sup = np.tile(maximo, (n_lobos,1))
    lim_inf = np.tile(minimo, (n_lobos,1))
    
    
    for i in range(0,max_iter):

        start = time.time()
       
        alcateia_fit = fluxo_de_pot(alcateia_fit,sep)
        
        alcateia_fit = fitness(alcateia_fit,zeta,psi,sigma,omega)

        alcateia_fit = alcateia_fit[np.argsort(alcateia_fit[:, -1])]
        
        a = (2/10 - (0.1*i*(2/max_iter)))
        
        mu = 0.5
        sigma = 0.15
        
        r1 = np.random.normal(mu, sigma, size = (n_lobos,alcateia_fit[:,0:-6].shape[1]))
        r2 = np.random.normal(mu, sigma, size = (n_lobos,alcateia_fit[:,0:-6].shape[1]))
        
        #r1 = np.random.random_sample(size = (n_lobos,alcateia_fit[:,0:-6].shape[1]))
        
        #r2 = np.random.random_sample(size = (n_lobos,alcateia_fit[:,0:-6].shape[1]))

        A = (2*a*r1) - a
        
        C = 2*r2
        
        if (i == 0):
        
            lobo_alfa = alcateia_fit[0, :].copy()
            lobo_beta = alcateia_fit[1, :].copy()
            lobo_delta = alcateia_fit[2, :].copy()
            
            alfa = np.expand_dims(alcateia_fit[0,0:-6].copy(), 1)
            beta = np.expand_dims(alcateia_fit[1,0:-6].copy(), 1)
            delta = np.expand_dims(alcateia_fit[2,0:-6].copy(), 1)
            
        
        for t in range(3):

            if (alcateia_fit[t, -1] < lobo_alfa[-1]):

                lobo_alfa = alcateia_fit[0,:].copy()
                    
                alcateias.append(alcateia_fit)

                alfa = np.expand_dims(alcateia_fit[1,0:-6].copy(), 1)

            if (alcateia_fit[t,-1] > lobo_alfa[-1] and alcateia_fit[t,-1] < lobo_beta[-1]):

                lobo_beta = alcateia_fit[1,:].copy()

                beta = np.expand_dims(alcateia_fit[1,0:-6].copy(), 1)

            if (alcateia_fit[t,-1] > lobo_alfa[-1] and alcateia_fit[t,-1] > lobo_beta[-1] and alcateia_fit[t,-1] < lobo_delta[-1]):

                lobo_delta = alcateia_fit[2, :].copy()

                delta = np.expand_dims(alcateia_fit[2,0:-6].copy(), 1)         
        

        d_alfa = np.abs(np.multiply(C, alfa.T) - alcateia_fit[:, 0:-6])*0.1

        d_beta = np.abs(np.multiply(C, beta.T) - alcateia_fit[:, 0:-6])*0.1

        d_delta = np.abs(np.multiply(C, delta.T) - alcateia_fit[:, 0:-6])*0.1

        x_alfa = alfa.T - np.multiply(A, d_alfa)

        x_beta = beta.T - np.multiply(A, d_beta)

        x_delta = delta.T - np.multiply(A, d_delta)

        alcateia_fit[:,0:-6] = (x_alfa + x_beta + x_delta)/3

      
        
        for linha in range(n_lobos):
          
            alcateia_fit[linha][n_vgen:n_vgen+n_tap] = discreto_tap(alcateia_fit[linha],n_tap,n_vgen,n_bshunt,sep)
            alcateia_fit[linha][n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(alcateia_fit[linha],n_tap,n_vgen,n_bshunt,sep)
            
        alca_estat = alcateia_fit[:,-6:]

        alcateia_fit = np.concatenate(( np.clip(alcateia_fit[:,0:-6], a_min = lim_inf, a_max = lim_sup, out = alcateia_fit[:,0:-6]),alca_estat),axis=1)
        
        
        end = time.time()

        elapsed = end - start
        
        j.append(lobo_alfa[-1])

        perdas.append(lobo_alfa[-6])

        pen_v.append(lobo_alfa[-5])

        pen_gq.append(lobo_alfa[-4])

        pen_tap.append(lobo_alfa[-3])

        pen_bsh.append(lobo_alfa[-2])       
        
        
        tempo.append(elapsed)
        
        if relatorio == True:
            
            print(' ')

            print('Lobo Alfa da Iteração:',i)

            print('Perdas (pu):',lobo_alfa[-6])

            print('Penalização de Tensão:',lobo_alfa[-5])

            print('Penalização de Geração de Reativo:',lobo_alfa[-4])

            print('Penalização do Tap:',lobo_alfa[-3])

            print('Penalização do Bshunt:',lobo_alfa[-2])

            print('Fitness:',lobo_alfa[-1])
            
            print('Tempo: ',elapsed)

            print(' ')

            print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
            
    if relatorio == True:
        
            plt.figure(figsize=(18,10))
            plt.plot(perdas)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Perdas de Potência Ativa (pu)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(j)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Fitness (J)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_v)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Penalização de Tensão')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_gq)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Penalização de Geração de Reativo')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_tap)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Penalização do TAP')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_bsh)
            plt.grid()
            plt.title('Otimização Por Alcateia de Lobos Cinzentos')
            plt.ylabel('Penalização do BShunt')
            plt.xlabel('Número da Iteração')

    return j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,alcateias,lobo_alfa, lobo_beta, lobo_delta, tempo

    


 ################################################################################################################################################################################################


def ipm_ieee30 (sep30, solver, rtol, otol, max_iter, relatorio = True, remoto = True, arredondado = True, travado = True, minlp = True):

    sep30.res_bus = sep30.res_bus.sort_index()

    Sbase = 100
    
    sep = GEKKO(remote=remoto)

    ########################################################################### Vetor de tensões das barras

    v = np.ones(len(sep30.bus))

    v = sep30.res_bus['vm_pu'].to_numpy()

    ########################################################################### Vetor de ângulos das barras

    theta = np.zeros(len(sep30.bus))

    theta = np.radians(sep30.res_bus['va_degree'].to_numpy())

    ########################################################################### Vetor de potência ativa gerada

    pg = np.zeros(len(sep30.bus))

    i = 0

    sep30.gen = sep30.gen.sort_index()

    sep30.res_gen = sep30.res_gen.sort_index()

    for bus in sep30.gen['bus'].to_numpy():

        pg[bus] = sep30.gen['p_mw'].to_numpy()[i]/Sbase

        i = i+1

    pg[0] = sep30.res_ext_grid['p_mw'].to_numpy()/Sbase

    pg_ls = sep30.ext_grid['max_p_mw'].to_numpy()/Sbase

    pg_li = sep30.ext_grid['min_p_mw'].to_numpy()/Sbase


    ########################################################################### Vetor de potência reativa gerada

    qg = np.zeros(len(sep30.bus))

    i = 0

    sep30.gen = sep30.gen.sort_index()

    for bus in sep30.gen['bus'].to_numpy():

        qg[bus] = sep30.res_gen['q_mvar'].to_numpy()[i]/Sbase

        i = i+1

    qg[0] = sep30.res_ext_grid['q_mvar']/Sbase


    ########################################################################### Vetores de limite de potência reativa

    qg_ls = np.zeros(len(sep30.bus))
    qg_li = np.zeros(len(sep30.bus))

    i=0

    for bus in sep30.gen['bus'].to_numpy():

        qg_ls[bus] = sep30.gen['max_q_mvar'].to_numpy()[i]/Sbase
        qg_li[bus] = sep30.gen['min_q_mvar'].to_numpy()[i]/Sbase


        i=i+1

    qg_ls[0] = sep30.ext_grid['max_q_mvar'].to_numpy()/Sbase

    qg_li[0] = sep30.ext_grid['min_q_mvar'].to_numpy()/Sbase


    ########################################################################### Vetor de potência ativa consumida


    pc = np.zeros(len(sep30.bus))

    i = 0

    sep30.load = sep30.load.sort_index()

    for bus in sep30.load['bus'].to_numpy():

        pc[bus] = sep30.load['p_mw'].to_numpy()[i]/Sbase

        i=i+1


    ########################################################################### Vetor de potência reativa consumida

    qc = np.zeros(len(sep30.bus))

    i = 0

    for bus in sep30.load['bus'].to_numpy():

        qc[bus] = sep30.load['q_mvar'].to_numpy()[i]/Sbase

        i=i+1

    ########################################################################### Vetor de MVAR Shunt

    sh = np.zeros(len(sep30.bus))

    i = 0

    sep30.shunt = sep30.shunt.sort_index()

    for bus in sep30.shunt['bus'].to_numpy():

        sh[bus] = sep30.shunt['q_mvar'].to_numpy()[i]/Sbase

        i=i+1


    ########################################################################### Vetores de condutância e susceptância série

    m_z = np.zeros((5,len(sep30.line)))

    gkm = np.zeros(len(sep30.line))

    bkm = np.zeros(len(sep30.line))

    bo = np.zeros(len(sep30.line))

    bd = np.zeros(len(sep30.line))

    sep30.line = sep30.line.sort_index()

    sep30.bus = sep30.bus.sort_index()

    vbus = sep30.bus.vn_kv.to_numpy(dtype=np.float64)

    zbase = np.power(np.multiply(vbus,1000), 2)/(100*1e6)

    m_z[0,:] = sep30.line.from_bus.to_numpy()

    m_z[1,:] = sep30.line.to_bus.to_numpy()

    bsh = 1e-9*(2*np.pi*60*sep30.line.c_nf_per_km.to_numpy())

    m_z[4,:] = bsh


    for i in range(len(sep30.line.index.ravel())):    

        m_z[2,i] = sep30.line.r_ohm_per_km[i]/zbase[int(m_z[0,i])]

        m_z[3,i] = sep30.line.x_ohm_per_km[i]/zbase[int(m_z[0,i])]

        m_z[4,i] =  m_z[4,i] * zbase[int(m_z[0,i])]


    gkm = np.array(np.divide(m_z[2,:], np.power(m_z[2,:],2)+np.power(m_z[3],2)))

    bo = m_z[0,:]

    bd = m_z[1,:]


    ########################################################################### Vetor de susceptância


    bkm = np.array(np.divide(m_z[3,:], np.power(m_z[2,:],2)+np.power(m_z[3],2)))


    ########################################################################### Vetor de susceptância shunt


    bsh = m_z[4,:]


    ########################################################################### Vetores de condutância e susceptância série

    m_z = np.zeros((5,len(sep30.line)))

    gkm = np.zeros(len(sep30.line))

    bkm = np.zeros(len(sep30.line))

    bo = np.zeros(len(sep30.line))

    bd = np.zeros(len(sep30.line))

    sep30.line = sep30.line.sort_index()

    sep30.bus = sep30.bus.sort_index()

    vbus = sep30.bus.vn_kv.to_numpy(dtype=np.float64)

    zbase = np.power(np.multiply(vbus,1000), 2)/(100*1e6)

    m_z[0,:] = sep30.line.from_bus.to_numpy()

    m_z[1,:] = sep30.line.to_bus.to_numpy()

    bsh = 1e-9*(2*np.pi*60*sep30.line.c_nf_per_km.to_numpy())

    m_z[4,:] = bsh


    for i in range(len(sep30.line.index.ravel())):    

        m_z[2,i] = sep30.line.r_ohm_per_km[i]/zbase[int(m_z[0,i])]

        m_z[3,i] = sep30.line.x_ohm_per_km[i]/zbase[int(m_z[0,i])]

        m_z[4,i] =  m_z[4,i] * zbase[int(m_z[0,i])]


    gkm = np.array(np.divide(m_z[2,:], np.power(m_z[2,:],2)+np.power(m_z[3],2)))

    bo = m_z[0,:]

    bd = m_z[1,:]

    ########################################################################### Vetor de susceptância

    bkm = np.array(np.divide(m_z[3,:], np.power(m_z[2,:],2)+np.power(m_z[3],2)))


    ########################################################################### Vetor de susceptância shunt

    bsh = m_z[4,:]

    ########################################################################### Vetor de tap

    tap_pos = sep30.trafo[~pd.isnull(sep30.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)

    tap_neutral = sep30.trafo[~pd.isnull(sep30.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)

    tap_step_percent = sep30.trafo[~pd.isnull(sep30.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)

    valor_percentual = (tap_pos-tap_neutral)*(tap_step_percent/100) + 1

    valor_percentual = np.resize(valor_percentual,(len(sep30.trafo)))

    to = sep30.trafo['hv_bus'].to_numpy()


    td = sep30.trafo['lv_bus'].to_numpy()

    i = 0

    for i in range(len(valor_percentual)):

        if i < len(tap_pos):

            valor_percentual[i] = valor_percentual[i]

        else:

            valor_percentual[i] = 1


    tap = valor_percentual

    ########################################################################### Vetores de limite de tap

    tap_max = np.ones(len(tap))*1.12


    tap_min = np.ones(len(tap))*0.88

    
    sep = GEKKO(remote=True)

    ################################################### TENSÕES

    vmax = 1.05

    vmin = 0.95

    
    v0 = sep.Var(value=v[0],lb=vmin,ub=vmax)
    
    v1 = sep.Var(value=v[1],lb=vmin,ub=vmax)

    v2 = sep.Var(value=v[2],lb=vmin,ub=vmax)


    v3 = sep.Var(value=v[3],lb=vmin,ub=vmax)


    v4 = sep.Var(value=v[4],lb=vmin,ub=vmax)


    v5 = sep.Var(value=v[5],lb=vmin,ub=vmax)


    v6 = sep.Var(value=v[6],lb=vmin,ub=vmax)


    v7 = sep.Var(value=v[7],lb=vmin,ub=vmax)


    v8 = sep.Var(value=v[8],lb=vmin,ub=vmax)


    v9 = sep.Var(value=v[9],lb=vmin,ub=vmax)


    v10 = sep.Var(value=v[10],lb=vmin,ub=vmax)


    v11 = sep.Var(value=v[11],lb=vmin,ub=vmax)


    v12 = sep.Var(value=v[12],lb=vmin,ub=vmax)


    v13 = sep.Var(value=v[13],lb=vmin,ub=vmax)


    v14 = sep.Var(value=v[14],lb=vmin,ub=vmax)


    v15 = sep.Var(value=v[15],lb=vmin,ub=vmax)


    v16 = sep.Var(value=v[16],lb=vmin,ub=vmax)


    v17 = sep.Var(value=v[17],lb=vmin,ub=vmax)


    v18 = sep.Var(value=v[18],lb=vmin,ub=vmax)


    v19 = sep.Var(value=v[19],lb=vmin,ub=vmax)


    v20 = sep.Var(value=v[20],lb=vmin,ub=vmax)


    v21 = sep.Var(value=v[21],lb=vmin,ub=vmax)


    v22 = sep.Var(value=v[22],lb=vmin,ub=vmax)


    v23 = sep.Var(value=v[23],lb=vmin,ub=vmax)


    v24 = sep.Var(value=v[24],lb=vmin,ub=vmax)


    v25 = sep.Var(value=v[25],lb=vmin,ub=vmax)


    v26 = sep.Var(value=v[26],lb=vmin,ub=vmax)


    v27 = sep.Var(value=v[27],lb=vmin,ub=vmax)


    v28 = sep.Var(value=v[28],lb=vmin,ub=vmax)


    v29 = sep.Var(value=v[29],lb=vmin,ub=vmax)



    ################################################## ÂNGULOS


    theta0 = theta[0]


    theta1 = sep.Var(value=theta[1],lb=-np.pi,ub=np.pi)


    theta2 = sep.Var(value=theta[2],lb=-np.pi,ub=np.pi)


    theta3 = sep.Var(value=theta[3],lb=-np.pi,ub=np.pi)


    theta4 = sep.Var(value=theta[4],lb=-np.pi,ub=np.pi)


    theta5 = sep.Var(value=theta[5],lb=-np.pi,ub=np.pi)


    theta6 = sep.Var(value=theta[6],lb=-np.pi,ub=np.pi)


    theta7 = sep.Var(value=theta[7],lb=-np.pi,ub=np.pi)


    theta8 = sep.Var(value=theta[8],lb=-np.pi,ub=np.pi)


    theta9 = sep.Var(value=theta[9],lb=-np.pi,ub=np.pi)


    theta10 = sep.Var(value=theta[10],lb=-np.pi,ub=np.pi)


    theta11 = sep.Var(value=theta[11],lb=-np.pi,ub=np.pi)


    theta12 = sep.Var(value=theta[12],lb=-np.pi,ub=np.pi)


    theta13 = sep.Var(value=theta[13],lb=-np.pi,ub=np.pi)


    theta14 = sep.Var(value=theta[14],lb=-np.pi,ub=np.pi)


    theta15 = sep.Var(value=theta[15],lb=-np.pi,ub=np.pi)


    theta16 = sep.Var(value=theta[16],lb=-np.pi,ub=np.pi)


    theta17 = sep.Var(value=theta[17],lb=-np.pi,ub=np.pi)


    theta18 = sep.Var(value=theta[18],lb=-np.pi,ub=np.pi)


    theta19 = sep.Var(value=theta[19],lb=-np.pi,ub=np.pi)


    theta20 = sep.Var(value=theta[20],lb=-np.pi,ub=np.pi)


    theta21 = sep.Var(value=theta[21],lb=-np.pi,ub=np.pi)


    theta22 = sep.Var(value=theta[22],lb=-np.pi,ub=np.pi)


    theta23 = sep.Var(value=theta[23],lb=-np.pi,ub=np.pi)


    theta24 = sep.Var(value=theta[24],lb=-np.pi,ub=np.pi)


    theta25 = sep.Var(value=theta[25],lb=-np.pi,ub=np.pi)


    theta26 = sep.Var(value=theta[26],lb=-np.pi,ub=np.pi)


    theta27 = sep.Var(value=theta[27],lb=-np.pi,ub=np.pi)


    theta28 = sep.Var(value=theta[28],lb=-np.pi,ub=np.pi)


    theta29 = sep.Var(value=theta[29],lb=-np.pi,ub=np.pi)


    ################################################## POTÊNCIA ATIVA BARRA DE REFERÊNCIA


    pg0 = sep.Var(value = pg[0], lb = pg_li[0], ub = pg_ls[0])


    ################################################## POTÊNCIA REATIVA BARRA DE REFERÊNCIA


    qg0 = sep.Var(value = qg[0],lb = qg_li[0],ub = qg_ls[0])


    ################################################## POTÊNCIAS REATIVA GERADA DAS DEMAIS BARRAS

    qg1 = sep.Var(value = qg[1],lb = qg_li[1],ub = qg_ls[1])


    qg4 = sep.Var(value = qg[4],lb = qg_li[4],ub = qg_ls[4])


    qg7 = sep.Var(value = qg[7],lb = qg_li[7],ub = qg_ls[7])


    qg10 = sep.Var(value = qg[10],lb = qg_li[10],ub = qg_ls[10])


    qg12 = sep.Var(value = qg[12],lb = qg_li[12],ub = qg_ls[12])


    ################################################## TAPS

    
    if travado == True:

        tap58 = tap[0]#sep.Var(value = tap[0],lb = tap_min[0], ub = tap_max[0])


        tap59 = tap[1]#sep.Var(value = tap[1],lb = tap_min[1], ub = tap_max[1])


        tap311 = tap[2]#sep.Var(value = tap[2],lb = tap_min[2], ub = tap_max[2])


        tap2726 = tap[3]#sep.Var(value = tap[3],lb = tap_min[3], ub = tap_max[3])
        
        
        sh9 = sh[9]#sep.Var(value = sh[9],lb = -0.39, ub = 0)                        

        
        sh23 = sh[23]#sep.Var(value = sh[23],lb = -0.09, ub = 0)                        

    
    else:

        
        if minlp == True:
            
            tap58 = sep.sos1([0.9    , 0.90625, 0.9125 , 0.91875, 0.925  , 0.93125, 0.9375 ,
          0.94375, 0.95   , 0.95625, 0.9625 , 0.96875, 0.975  , 0.98125,
          0.9875 , 0.99375, 1.     , 1.00625, 1.0125 , 1.01875, 1.025  ,
          1.03125, 1.0375 , 1.04375, 1.05   , 1.05625, 1.0625 , 1.06875,
          1.075  , 1.08125, 1.0875 , 1.09375, 1.1 ])

            tap59 = sep.sos1([0.9    , 0.90625, 0.9125 , 0.91875, 0.925  , 0.93125, 0.9375 ,
          0.94375, 0.95   , 0.95625, 0.9625 , 0.96875, 0.975  , 0.98125,
          0.9875 , 0.99375, 1.     , 1.00625, 1.0125 , 1.01875, 1.025  ,
          1.03125, 1.0375 , 1.04375, 1.05   , 1.05625, 1.0625 , 1.06875,
          1.075  , 1.08125, 1.0875 , 1.09375, 1.1 ])

            tap311 = sep.sos1([0.9    , 0.90625, 0.9125 , 0.91875, 0.925  , 0.93125, 0.9375 ,
          0.94375, 0.95   , 0.95625, 0.9625 , 0.96875, 0.975  , 0.98125,
          0.9875 , 0.99375, 1.     , 1.00625, 1.0125 , 1.01875, 1.025  ,
          1.03125, 1.0375 , 1.04375, 1.05   , 1.05625, 1.0625 , 1.06875,
          1.075  , 1.08125, 1.0875 , 1.09375, 1.1 ])

            tap2726 = sep.sos1([0.9    , 0.90625, 0.9125 , 0.91875, 0.925  , 0.93125, 0.9375 ,
          0.94375, 0.95   , 0.95625, 0.9625 , 0.96875, 0.975  , 0.98125,
          0.9875 , 0.99375, 1.     , 1.00625, 1.0125 , 1.01875, 1.025  ,
          1.03125, 1.0375 , 1.04375, 1.05   , 1.05625, 1.0625 , 1.06875,
          1.075  , 1.08125, 1.0875 , 1.09375, 1.1 ])
           

                      
            tap2726.value = tap[3]

            tap58.value = tap[0]

            tap59.value = tap[1]

            tap311.value = tap[2]
            
            sh23 = sep.sos1([0, -0.04, -0.05, -0.09])

            sh9 = sep.sos1([0, -0.2, -0.05, -0.15, -0.19, -0.24, -0.34, -0.39])

            sh9.value = sh[9]

            sh23.value = sh[23]

        else:
            
            tap58 = sep.Var(value = tap[0],lb = tap_min[0], ub = tap_max[0])

            tap59 = sep.Var(value = tap[1],lb = tap_min[1], ub = tap_max[1])

            tap311 = sep.Var(value = tap[2],lb = tap_min[2], ub = tap_max[2])

            tap2726 = sep.Var(value = tap[3],lb = tap_min[3], ub = tap_max[3])

            sh9 = sep.Var(value = sh[9],lb = -0.4, ub = 0)                        

            sh23 = sep.Var(value = sh[23],lb = -0.09, ub = 0)    



        
    tap108 = 1

    tap98 = 1

    tap1112 = 1


    ################################################## INJEÇÃO MVAR Capacitivo



    ################################################## Bshunt Trafo


    sep30.trafo = sep30.trafo.sort_index()

    barras = sep30.trafo['hv_bus'].to_numpy()

    xkm = (sep30.trafo['vk_percent'].to_numpy()/100)*(1000/sep30.trafo['sn_mva'].to_numpy())

    bkmt = 10/xkm

    # EQUAÇÕES DO FLUXO DE CARGA (POTÊNCIA ATIVA E REATIVA)

    ############################ BARRA 0 - ATIVA

    sep.Equation(

        pg0 - pc[0] - 


        # LINHA 0 - BARRA 0 COM 1

        (gkm[0]*(v0**2) - v0*v1*(gkm[0]*sep.cos(theta0-theta1) - bkm[0]*sep.sin(theta0-theta1)) +  

        # LINHA 1 - BARRA 0 COM 2

         gkm[1]*(v0**2) - v0*v2*(gkm[1]*sep.cos(theta0-theta2)-bkm[1]*sep.sin(theta0-theta2))) == 0


            )


    ############################ BARRA 0 - REATIVA


    sep.Equation(


        qg0 - qc[0] - 

        # LINHA 0 - BARRA 0 COM 1

        ((-(v0**2)*(-bkm[0]+bsh[0]/2)) + v0*v1*(-bkm[0]*sep.cos(theta0-theta1) - gkm[0]*sep.sin(theta0-theta1)) +

        # LINHA 1 - BARRA 0 COM 4


        (-(v0**2)*(-bkm[1]+bsh[1]/2)) + v0*v2*(-bkm[1]*sep.cos(theta0-theta2) - gkm[1]*sep.sin(theta0-theta2))) == 0


            )


    ############################ BARRA 1 - ATIVA

    sep.Equation(

        pg[1] - pc[1] - 


        # LINHA 0 - BARRA 0 COM 1

        (gkm[0]*(v1**2) - v0*v1*(gkm[0]*sep.cos(theta1-theta0) - bkm[0]*sep.sin(theta1-theta0)) +  

        # LINHA 2 - BARRA 1 COM 3

         gkm[2]*(v1**2) - v1*v3*(gkm[2]*sep.cos(theta1-theta3)-bkm[2]*sep.sin(theta1-theta3)) +

        # LINHA 4 - BARRA 1 COM 4

         gkm[4]*(v1**2) - v1*v4*(gkm[4]*sep.cos(theta1-theta4)-bkm[4]*sep.sin(theta1-theta4)) +

        # LINHA 5 - BARRA 1 COM 5

         gkm[5]*(v1**2) - v1*v5*(gkm[5]*sep.cos(theta1-theta5)-bkm[5]*sep.sin(theta1-theta5))) == 0


            )


    ############################ BARRA 1 - REATIVA


    sep.Equation(


        qg1 - qc[1] - 

         # LINHA 0 - BARRA 0 COM 1

        ((-(v1**2)*(-bkm[0]+bsh[0]/2)) + v0*v1*(-bkm[0]*sep.cos(theta1-theta0) - gkm[0]*sep.sin(theta1-theta0)) +

        # LINHA 2 - BARRA 1 COM 3

        (-(v1**2)*(-bkm[2]+bsh[2]/2)) + v3*v1*(-bkm[2]*sep.cos(theta1-theta3) - gkm[2]*sep.sin(theta1-theta3)) +

        # LINHA 4 - BARRA 1 COM 4


        (-(v1**2)*(-bkm[4]+bsh[4]/2)) + v4*v1*(-bkm[4]*sep.cos(theta1-theta4) - gkm[4]*sep.sin(theta1-theta4)) +

        # LINHA 5 - BARRA 1 COM 5

        (-(v1**2)*(-bkm[5]+bsh[5]/2)) + v1*v5*(-bkm[5]*sep.cos(theta1-theta5) - gkm[5]*sep.sin(theta1-theta5))) == 0

            )

    ############################ BARRA 2 - ATIVA

    sep.Equation(

        pg[2] - pc[2] - 


        # LINHA 2 - BARRA 1 COM 3

        (gkm[1]*(v2**2) - v0*v2*(gkm[1]*sep.cos(theta2-theta0) - bkm[1]*sep.sin(theta2-theta0)) +  

        # LINHA 3 - BARRA 2 COM 3

         gkm[3]*(v2**2) - v2*v3*(gkm[3]*sep.cos(theta2-theta3)-bkm[3]*sep.sin(theta2-theta3))) == 0


            )


    ############################ BARRA 2 - REATIVA


    sep.Equation(


        qg[2] - qc[2] - 

         # LINHA 1 - BARRA 0 COM 2

        ((-(v2**2)*(-bkm[1]+bsh[1]/2)) + v0*v2*(-bkm[1]*sep.cos(theta2-theta0) - gkm[1]*sep.sin(theta2-theta0)) +

        # LINHA 3 - BARRA 2 COM 3

        (-(v2**2)*(-bkm[3]+bsh[3]/2)) + v3*v2*(-bkm[3]*sep.cos(theta2-theta3) - gkm[3]*sep.sin(theta2-theta3))) == 0

            )


    ############################ BARRA 3 - ATIVA

    sep.Equation(

        pg[3] - pc[3] - 


        # LINHA 2 - BARRA 1 COM 3

        (gkm[2]*(v3**2) - v1*v3*(gkm[2]*sep.cos(theta3-theta1) - bkm[2]*sep.sin(theta3-theta1)) +  

        # LINHA 3 - BARRA 2 COM 3

         gkm[3]*(v3**2) - v2*v3*(gkm[3]*sep.cos(theta3-theta2)-bkm[3]*sep.sin(theta3-theta2)) + 

        # LINHA 6 - BARRA 3 COM 5

         gkm[6]*(v3**2) - v5*v3*(gkm[6]*sep.cos(theta3-theta5)-bkm[6]*sep.sin(theta3-theta5)) + 

        # TRAFO 4 - BARRA 3 COM 11

        -(v3*v11/tap311)*-bkmt[4]*sep.sin(theta3-theta11)) == 0


            )


    ############################ BARRA 3 - REATIVA


    sep.Equation(


        qg[3] - qc[3] - 

        # LINHA 2 - BARRA 1 COM 3

        ((-(v3**2)*(-bkm[2]+bsh[2]/2)) + v1*v3*(-bkm[2]*sep.cos(theta3-theta1) - gkm[2]*sep.sin(theta3-theta1)) +

        # LINHA 3 - BARRA 2 COM 3

        (-(v3**2)*(-bkm[3]+bsh[3]/2)) + v2*v3*(-bkm[3]*sep.cos(theta3-theta2) - gkm[3]*sep.sin(theta3-theta2)) +

        # LINHA 6 - BARRA 3 COM 5

        (-(v3**2)*(-bkm[6]+bsh[6]/2)) + v3*v5*(-bkm[6]*sep.cos(theta3-theta5) - gkm[6]*sep.sin(theta3-theta5)) +

        # TRAFO 4 - BARRA 3 COM 11

        -(-bkmt[4]/tap311**2)*v3**2 + (v3*v11/tap311)*-bkmt[4]*sep.cos(theta3-theta11)) == 0


                )

    ############################ BARRA 4 - ATIVA

    sep.Equation(

        pg[4] - pc[4] - 


        # LINHA 4 - BARRA 1 COM 4

        (gkm[4]*(v4**2) - v1*v4*(gkm[4]*sep.cos(theta4-theta1) - bkm[4]*sep.sin(theta4-theta1)) +  

        # LINHA 7 - BARRA 4 COM 6

         gkm[7]*(v4**2) - v4*v6*(gkm[7]*sep.cos(theta4-theta6)-bkm[7]*sep.sin(theta4-theta6))) == 0


            )


    ############################ BARRA 4 - REATIVA


    sep.Equation(


        qg4 - qc[4] - 

        # LINHA 4 - BARRA 1 COM 4

        ((-(v4**2)*(-bkm[4]+bsh[4]/2)) + v1*v4*(-bkm[4]*sep.cos(theta4-theta1) - gkm[4]*sep.sin(theta4-theta1)) +

        # LINHA 7 - BARRA 4 COM 6

        (-(v4**2)*(-bkm[7]+bsh[7]/2)) + v4*v6*(-bkm[7]*sep.cos(theta4-theta6) - gkm[7]*sep.sin(theta4-theta6))) == 0


                )


    ############################ BARRA 5 - ATIVA

    sep.Equation(

        pg[5] - pc[5] - 


        # LINHA 5 - BARRA 1 COM 5

        (gkm[5]*(v5**2) - v1*v5*(gkm[5]*sep.cos(theta5-theta1) - bkm[5]*sep.sin(theta5-theta1)) +  

        # LINHA 6 - BARRA 3 COM 5

        gkm[6]*(v5**2) - v3*v5*(gkm[6]*sep.cos(theta5-theta3) - bkm[6]*sep.sin(theta5-theta3)) +  

        # LINHA 8 - BARRA 5 COM 6

         gkm[8]*(v5**2) - v6*v5*(gkm[8]*sep.cos(theta5-theta6) - bkm[8]*sep.sin(theta5-theta6)) +  

        # LINHA 9 - BARRA 5 COM 7

         gkm[9]*(v5**2) - v7*v5*(gkm[9]*sep.cos(theta5-theta7) - bkm[9]*sep.sin(theta5-theta7)) +  

        # LINHA 33 - BARRA 5 COM 27

         gkm[33]*(v5**2) - v27*v5*(gkm[33]*sep.cos(theta5-theta27) - bkm[33]*sep.sin(theta5-theta27)) +  


        # TRAFO 0 - BARRA 5 COM 8

         -(v5*v8/tap58)*-bkmt[0]*sep.sin(theta5-theta8) +


        # TRAFO 1 - BARRA 5 COM 9

       -(v5*v9/tap59)*-bkmt[1]*sep.sin(theta5-theta9))== 0


            )


    ############################ BARRA 5 - REATIVA


    sep.Equation(


        qg[5] - qc[5] - 

        # LINHA 5 - BARRA 1 COM 5

        ((-(v5**2)*(-bkm[5]+bsh[5]/2)) + v1*v5*(-bkm[5]*sep.cos(theta5-theta1) - gkm[5]*sep.sin(theta5-theta1)) +

        # LINHA 6 - BARRA 3 COM 5

         (-(v5**2)*(-bkm[6]+bsh[6]/2)) + v3*v5*(-bkm[6]*sep.cos(theta5-theta3) - gkm[6]*sep.sin(theta5-theta3)) +

        # LINHA 8 - BARRA 5 COM 6

         (-(v5**2)*(-bkm[8]+bsh[8]/2)) + v5*v6*(-bkm[8]*sep.cos(theta5-theta6) - gkm[8]*sep.sin(theta5-theta6)) +

        # LINHA 9 - BARRA 5 COM 7

         (-(v5**2)*(-bkm[9]+bsh[9]/2)) + v5*v7*(-bkm[9]*sep.cos(theta5-theta7) - gkm[9]*sep.sin(theta5-theta7)) +

        # LINHA 33 - BARRA 5 COM 27

         (-(v5**2)*(-bkm[33]+bsh[33]/2)) + v5*v27*(-bkm[33]*sep.cos(theta5-theta27) - gkm[33]*sep.sin(theta5-theta27)) +

        # TRAFO 0 - BARRA 5 COM 8

         -(-bkmt[0]/tap58**2)*v5**2 + (v5*v8/tap58)*-bkmt[0]*sep.cos(theta5-theta8) +

        # TRAFO 1 - BARRA 5 COM 9

        -(-bkmt[1]/tap59**2)*v5**2 + (v5*v9/tap59)*-bkmt[1]*sep.cos(theta5-theta9)) == 0


                )


    ############################ BARRA 6 - ATIVA

    sep.Equation(

        pg[6] - pc[6] - 


        # LINHA 7 - BARRA 4 COM 6

        (gkm[7]*(v6**2) - v6*v4*(gkm[7]*sep.cos(theta6-theta4) - bkm[7]*sep.sin(theta6-theta4)) +  

        # LINHA 8 - BARRA 5 COM 6

         gkm[8]*(v6**2) - v5*v6*(gkm[8]*sep.cos(theta6-theta5)-bkm[8]*sep.sin(theta6-theta5))) == 0


            )


    ############################ BARRA 6 - REATIVA


    sep.Equation(


        qg[6] - qc[6] - 

        # LINHA 7 - BARRA 4 COM 6

        ((-(v6**2)*(-bkm[7]+bsh[7]/2)) + v6*v4*(-bkm[7]*sep.cos(theta6-theta4) - gkm[7]*sep.sin(theta6-theta4)) +

        # LINHA 8 - BARRA 5 COM 6

        (-(v6**2)*(-bkm[8]+bsh[8]/2)) + v5*v6*(-bkm[8]*sep.cos(theta6-theta5) - gkm[8]*sep.sin(theta6-theta5))) == 0


                )

    ############################ BARRA 7 - ATIVA

    sep.Equation(

        pg[7] - pc[7] - 


        # LINHA 9 - BARRA 5 COM 7

        (gkm[9]*(v7**2) - v5*v7*(gkm[9]*sep.cos(theta7-theta5) - bkm[9]*sep.sin(theta7-theta5)) +  

        # LINHA 32 - BARRA 7 COM 27

         gkm[32]*(v7**2) - v7*v27*(gkm[32]*sep.cos(theta7-theta27)-bkm[32]*sep.sin(theta7-theta27))) == 0


            )


    ############################ BARRA 7 - REATIVA


    sep.Equation(


        qg7 - qc[7] - 

        # LINHA 9 - BARRA 5 COM 7

        ((-(v7**2)*(-bkm[9]+bsh[9]/2)) + v5*v7*(-bkm[9]*sep.cos(theta7-theta5) - gkm[9]*sep.sin(theta7-theta5)) +

        # LINHA 32 - BARRA 7 COM 27

        (-(v7**2)*(-bkm[32]+bsh[32]/2)) + v27*v7*(-bkm[32]*sep.cos(theta7-theta27) - gkm[32]*sep.sin(theta7-theta27))) == 0


                )

    ############################ BARRA 8 - ATIVA

    sep.Equation(

        pg[8] - pc[8] - 


        # TRAFO 0 - BARRA 5 COM 8

        (-(v5*v8/tap58)*-bkmt[0]*sep.sin(theta8-theta5) +

        # TRAFO 2 - BARRA 10 COM 8

         -(v10*v8/tap108)*-bkmt[2]*sep.sin(theta8-theta10) +

        # TRAFO 3 - BARRA 9 COM 8

         -(v9*v8/tap98)*-bkmt[3]*sep.sin(theta8-theta9)) == 0

            )


    ############################ BARRA 8 - REATIVA


    sep.Equation(


        qg[8] - qc[8] - 

        # TRAFO 0 - BARRA 5 COM 8

       (-(-bkmt[0])*v8**2 + (v5*v8/tap58)*-bkmt[0]*sep.cos(theta8-theta5) +

        # TRAFO 2 - BARRA 10 COM 8

        -(-bkmt[2])*v8**2 + (v10*v8/tap108)*-bkmt[2]*sep.cos(theta8-theta10) +

        # TRAFO 3 - BARRA 9 COM 8

        -(-bkmt[3])*v8**2 + (v9*v8/tap98)*-bkmt[3]*sep.cos(theta8-theta9)) == 0


                )


    ############################ BARRA 9 - ATIVA

    sep.Equation(

        pg[9] - pc[9] - 

         # TRAFO 1 - BARRA 5 COM 9

       (-(v9*v5/tap59)*-bkmt[1]*sep.sin(theta9-theta5) +

        # TRAFO 3 - BARRA 9 COM 8

        -(v8*v9/tap98)*-bkmt[3]*sep.sin(theta9-theta8) +

        # LINHA 18 - BARRA 9 COM 19

         gkm[18]*(v9**2) - v9*v19*(gkm[18]*sep.cos(theta9-theta19)-bkm[18]*sep.sin(theta9-theta19)) +

        # LINHA 19 - BARRA 9 COM 16

         gkm[19]*(v9**2) - v9*v16*(gkm[19]*sep.cos(theta9-theta16)-bkm[19]*sep.sin(theta9-theta16)) +

        # LINHA 20 - BARRA 9 COM 20

         gkm[20]*(v9**2) - v9*v20*(gkm[20]*sep.cos(theta9-theta20)-bkm[20]*sep.sin(theta9-theta20)) +

        # LINHA 21 - BARRA 9 COM 21

         gkm[21]*(v9**2) - v9*v21*(gkm[21]*sep.cos(theta9-theta21)-bkm[21]*sep.sin(theta9-theta21))) == 0


            )


    ############################ BARRA 9 - REATIVA


    sep.Equation(


        qg[9] - qc[9] - sh9*v9*v9 -

        # TRAFO 1 - BARRA 5 COM 9

       (-(-bkmt[1])*v9**2 + (v5*v9/tap59)*-bkmt[1]*sep.cos(theta9-theta5) +

        # TRAFO 3 - BARRA 9 COM 8

        -(-bkmt[3]/tap98**2)*v9**2 + (v8*v9/tap98)*-bkmt[3]*sep.cos(theta9-theta8) +

        # LINHA 18 - BARRA 9 COM 19

         (-(v9**2)*(-bkm[18]+bsh[18]/2)) + v9*v19*(-bkm[18]*sep.cos(theta9-theta19) - gkm[18]*sep.sin(theta9-theta19)) +

        # LINHA 19 - BARRA 9 COM 16

         (-(v9**2)*(-bkm[19]+bsh[19]/2)) + v9*v16*(-bkm[19]*sep.cos(theta9-theta16) - gkm[19]*sep.sin(theta9-theta16)) +

        # LINHA 20 - BARRA 9 COM 20

         (-(v9**2)*(-bkm[20]+bsh[20]/2)) + v9*v20*(-bkm[20]*sep.cos(theta9-theta20) - gkm[20]*sep.sin(theta9-theta20)) +

        # LINHA 21 - BARRA 9 COM 21

         (-(v9**2)*(-bkm[21]+bsh[21]/2)) + v9*v21*(-bkm[21]*sep.cos(theta9-theta21) - gkm[21]*sep.sin(theta9-theta21))) == 0

                )

    ############################ BARRA 10 - ATIVA

    sep.Equation(

        pg[10] - pc[10] - 

        # TRAFO 2 - BARRA 10 COM 8

        (-(v10*v8/tap108)*-bkmt[2]*sep.sin(theta10-theta8)) == 0


            )


    ############################ BARRA 10 - REATIVA


    sep.Equation(


        qg10 - qc[10] - (

        # TRAFO 2 - BARRA 10 COM 8

        -(-bkmt[2]/tap108**2)*v10**2 + (v8*v10/tap108)*-bkmt[2]*sep.cos(theta10-theta8)) == 0

                )

    ############################ BARRA 11 - ATIVA

    sep.Equation(

        pg[11] - pc[11] - 

        # TRAFO 5 - BARRA 11 COM 12

        (-(v11*v12/tap1112)*-bkmt[5]*sep.sin(theta11-theta12) +

         # TRAFO 4 - BARRA 3 COM 11

         -(v3*v11/tap311)*-bkmt[4]*sep.sin(theta11-theta3) +

         # LINHA 10 - BARRA 11 COM 13

         gkm[10]*(v11**2) - v11*v13*(gkm[10]*sep.cos(theta11-theta13)-bkm[10]*sep.sin(theta11-theta13)) +

         # LINHA 11 - BARRA 11 COM 14

         gkm[11]*(v11**2) - v11*v14*(gkm[11]*sep.cos(theta11-theta14)-bkm[11]*sep.sin(theta11-theta14)) +

         # LINHA 12 - BARRA 11 COM 15

         gkm[12]*(v11**2) - v11*v15*(gkm[12]*sep.cos(theta11-theta15)-bkm[12]*sep.sin(theta11-theta15))) == 0


            )


    ############################ BARRA 11 - REATIVA


    sep.Equation(


        qg[11] - qc[11] - (

        # TRAFO 5 - BARRA 11 COM 12

        -(-bkmt[5]/tap1112**2)*v11**2 + (v11*v12/tap1112)*-bkmt[5]*sep.cos(theta11-theta12) +

        # TRAFO 4 - BARRA 13 COM 11

        -(-bkmt[4])*v11**2 + (v11*v3/tap311)*-bkmt[4]*sep.cos(theta11-theta3) +

        # LINHA 10 - BARRA 11 COM 13

         (-(v11**2)*(-bkm[10]+bsh[10]/2)) + v11*v13*(-bkm[10]*sep.cos(theta11-theta13) - gkm[10]*sep.sin(theta11-theta13)) +

        # LINHA 11 - BARRA 11 COM 14


         (-(v11**2)*(-bkm[11]+bsh[11]/2)) + v11*v14*(-bkm[11]*sep.cos(theta11-theta14) - gkm[11]*sep.sin(theta11-theta14)) +

         # LINHA 12 - BARRA 11 COM 15


         (-(v11**2)*(-bkm[12]+bsh[12]/2)) + v11*v15*(-bkm[12]*sep.cos(theta11-theta15) - gkm[12]*sep.sin(theta11-theta15))) == 0

                )

    ############################ BARRA 12 - ATIVA

    sep.Equation(

        pg[12] - pc[12] - 

        # TRAFO 5 - BARRA 11 COM 12

        ( -(v11*v12/tap1112)*-bkmt[5]*sep.sin(theta11-theta12)) == 0    

            )


    ############################ BARRA 12 - REATIVA


    sep.Equation(


        qg12 - qc[12] - (

        # TRAFO 5 - BARRA 11 COM 12


        -(-bkmt[5])*v12**2 + (v12*v11/tap1112)*-bkmt[5]*sep.cos(theta12-theta11)) ==0


                )


    ############################ BARRA 13 - ATIVA

    sep.Equation(

        pg[13] - pc[13] - (

        #LINHA 10 - BARRA 11 COM 13

        gkm[10]*(v13**2) - v13*v11*(gkm[10]*sep.cos(theta13-theta11)-bkm[10]*sep.sin(theta13-theta11)) +

        #LINHA 13 - BARRA 13 COM 14

        gkm[13]*(v13**2) - v13*v14*(gkm[13]*sep.cos(theta13-theta14)-bkm[13]*sep.sin(theta13-theta14))) == 0


            )


    ############################ BARRA 13 - REATIVA


    sep.Equation(


        qg[13] - qc[13] - (


       #LINHA 10 - BARRA 11 COM 13


    (-(v13**2)*(-bkm[10]+bsh[10]/2)) + v11*v13*(-bkm[10]*sep.cos(theta13-theta11) - gkm[10]*sep.sin(theta13-theta11)) +


        #LINHA 13 - BARRA 13 COM 14

    (-(v13**2)*(-bkm[13]+bsh[13]/2)) + v14*v13*(-bkm[13]*sep.cos(theta13-theta14) - gkm[13]*sep.sin(theta13-theta14))) == 0


    )


    ############################ BARRA 14 - ATIVA

    sep.Equation(

        pg[14] - pc[14] - (

        #LINHA 11 - BARRA 11 COM 14

        gkm[11]*(v14**2) - v14*v11*(gkm[11]*sep.cos(theta14-theta11)-bkm[11]*sep.sin(theta14-theta11)) +

        #LINHA 15 - BARRA 14 COM 17

        gkm[15]*(v14**2) - v14*v17*(gkm[15]*sep.cos(theta14-theta17)-bkm[15]*sep.sin(theta14-theta17)) +

        #LINHA 23 - BARRA 14 COM 22

        gkm[23]*(v14**2) - v14*v22*(gkm[23]*sep.cos(theta14-theta22)-bkm[23]*sep.sin(theta14-theta22)) +

        #LINHA 13 - BARRA 13 COM 14

        gkm[13]*(v14**2) - v13*v14*(gkm[13]*sep.cos(theta14-theta13)-bkm[13]*sep.sin(theta14-theta13))) == 0


            )


    ############################ BARRA 14 - REATIVA


    sep.Equation(


        qg[14] - qc[14] - (


       #LINHA 11 - BARRA 11 COM 14


    (-(v14**2)*(-bkm[11]+bsh[11]/2)) + v11*v14*(-bkm[11]*sep.cos(theta14-theta11) - gkm[11]*sep.sin(theta14-theta11)) +


       #LINHA 15 - BARRA 14 COM 17


    (-(v14**2)*(-bkm[15]+bsh[15]/2)) + v14*v17*(-bkm[15]*sep.cos(theta14-theta17) - gkm[15]*sep.sin(theta14-theta17)) +

       #LINHA 23 - BARRA 14 COM 22


    (-(v14**2)*(-bkm[23]+bsh[23]/2)) + v14*v22*(-bkm[23]*sep.cos(theta14-theta22) - gkm[23]*sep.sin(theta14-theta22)) +


        #LINHA 13 - BARRA 13 COM 14

    (-(v14**2)*(-bkm[13]+bsh[13]/2)) + v14*v13*(-bkm[13]*sep.cos(theta14-theta13) - gkm[13]*sep.sin(theta14-theta13))) == 0


    )


    ############################ BARRA 15 - ATIVA

    sep.Equation(

        pg[15] - pc[15] - (

        #LINHA 12 - BARRA 11 COM 15

        gkm[12]*(v15**2) - v15*v11*(gkm[12]*sep.cos(theta15-theta11)-bkm[12]*sep.sin(theta15-theta11)) +

        #LINHA 14 - BARRA 15 COM 16

        gkm[14]*(v15**2) - v16*v15*(gkm[14]*sep.cos(theta15-theta16)-bkm[14]*sep.sin(theta15-theta16))) == 0


            )


    ############################ BARRA 15 - REATIVA


    sep.Equation(


        qg[15] - qc[15] - (


       #LINHA 12 - BARRA 11 COM 15


    (-(v15**2)*(-bkm[12]+bsh[12]/2)) + v15*v11*(-bkm[12]*sep.cos(theta15-theta11) - gkm[12]*sep.sin(theta15-theta11)) +


        #LINHA 14 - BARRA 15 COM 16

    (-(v15**2)*(-bkm[14]+bsh[14]/2)) + v15*v16*(-bkm[14]*sep.cos(theta15-theta16) - gkm[14]*sep.sin(theta15-theta16))) == 0


    )

    ############################ BARRA 16 - ATIVA

    sep.Equation(

        pg[16] - pc[16] - (

        #LINHA 19 - BARRA 9 COM 16

        gkm[19]*(v16**2) - v9*v16*(gkm[19]*sep.cos(theta16-theta9)-bkm[19]*sep.sin(theta16-theta9)) +

        #LINHA 14 - BARRA 15 COM 16

        gkm[14]*(v16**2) - v16*v15*(gkm[14]*sep.cos(theta16-theta15)-bkm[14]*sep.sin(theta16-theta15))) == 0


            )


    ############################ BARRA 16 - REATIVA


    sep.Equation(


        qg[16] - qc[16] - (


       #LINHA 19 - BARRA 9 COM 16


    (-(v16**2)*(-bkm[19]+bsh[19]/2)) + v16*v9*(-bkm[19]*sep.cos(theta16-theta9) - gkm[19]*sep.sin(theta16-theta9)) +


        #LINHA 14 - BARRA 15 COM 16

    (-(v16**2)*(-bkm[14]+bsh[14]/2)) + v15*v16*(-bkm[14]*sep.cos(theta16-theta15) - gkm[14]*sep.sin(theta16-theta15))) == 0


    )

    ############################ BARRA 17 - ATIVA

    sep.Equation(

        pg[17] - pc[17] - (

        #LINHA 15 - BARRA 14 COM 17

        gkm[15]*(v17**2) - v14*v17*(gkm[15]*sep.cos(theta17-theta14)-bkm[15]*sep.sin(theta17-theta14)) +

        #LINHA 16 - BARRA 17 COM 18

        gkm[16]*(v17**2) - v18*v17*(gkm[16]*sep.cos(theta17-theta18)-bkm[16]*sep.sin(theta17-theta18))) == 0


            )


    ############################ BARRA 17 - REATIVA


    sep.Equation(


        qg[17] - qc[17] - (


       #LINHA 15 - BARRA 14 COM 17


    (-(v17**2)*(-bkm[15]+bsh[15]/2)) + v17*v14*(-bkm[15]*sep.cos(theta17-theta14) - gkm[15]*sep.sin(theta17-theta14)) +


        #LINHA 16 - BARRA 17 COM 18

    (-(v17**2)*(-bkm[16]+bsh[16]/2)) + v17*v18*(-bkm[16]*sep.cos(theta17-theta18) - gkm[16]*sep.sin(theta17-theta18))) == 0


    )

    ############################ BARRA 18 - ATIVA

    sep.Equation(

        pg[18] - pc[18] - (

        #LINHA 17 - BARRA 18 COM 19

        gkm[17]*(v18**2) - v18*v19*(gkm[17]*sep.cos(theta18-theta19)-bkm[17]*sep.sin(theta18-theta19)) +

        #LINHA 16 - BARRA 17 COM 18

        gkm[16]*(v18**2) - v18*v17*(gkm[16]*sep.cos(theta18-theta17)-bkm[16]*sep.sin(theta18-theta17))) == 0


            )


    ############################ BARRA 18 - REATIVA


    sep.Equation(


        qg[18] - qc[18] - (


       #LINHA 17 - BARRA 18 COM 19


    (-(v18**2)*(-bkm[17]+bsh[17]/2)) + v18*v19*(-bkm[17]*sep.cos(theta18-theta19) - gkm[17]*sep.sin(theta18-theta19)) +


        #LINHA 16 - BARRA 17 COM 18

    (-(v18**2)*(-bkm[16]+bsh[16]/2)) + v17*v18*(-bkm[16]*sep.cos(theta18-theta17) - gkm[16]*sep.sin(theta18-theta17))) == 0


    )


    ############################ BARRA 19 - ATIVA

    sep.Equation(

        pg[19] - pc[19] - (

        #LINHA 17 - BARRA 18 COM 19

        gkm[17]*(v19**2) - v18*v19*(gkm[17]*sep.cos(theta19-theta18)-bkm[17]*sep.sin(theta19-theta18)) +

        #LINHA 18 - BARRA 9 COM 19

        gkm[18]*(v19**2) - v9*v19*(gkm[18]*sep.cos(theta19-theta9)-bkm[18]*sep.sin(theta19-theta9))) == 0


            )


    ############################ BARRA 19 - REATIVA


    sep.Equation(


        qg[19] - qc[19] - (


       #LINHA 17 - BARRA 18 COM 19


    (-(v19**2)*(-bkm[17]+bsh[17]/2)) + v18*v19*(-bkm[17]*sep.cos(theta19-theta18) - gkm[17]*sep.sin(theta19-theta18)) +


        #LINHA 18 - BARRA 9 COM 19

    (-(v19**2)*(-bkm[18]+bsh[18]/2)) + v9*v19*(-bkm[18]*sep.cos(theta19-theta9) - gkm[18]*sep.sin(theta19-theta9))) == 0


    )


    ############################ BARRA 20 - ATIVA

    sep.Equation(

        pg[20] - pc[20] - (

        #LINHA 20 - BARRA 9 COM 20

        gkm[20]*(v20**2) - v20*v9*(gkm[20]*sep.cos(theta20-theta9)-bkm[20]*sep.sin(theta20-theta9)) +

        #LINHA 22 - BARRA 21 COM 20

        gkm[22]*(v20**2) - v21*v20*(gkm[22]*sep.cos(theta20-theta21)-bkm[22]*sep.sin(theta20-theta21))) == 0


            )


    ############################ BARRA 20 - REATIVA


    sep.Equation(


        qg[20] - qc[20] - (


       #LINHA 20 - BARRA 9 COM 20


    (-(v20**2)*(-bkm[20]+bsh[20]/2)) + v9*v20*(-bkm[20]*sep.cos(theta20-theta9) - gkm[20]*sep.sin(theta20-theta9)) +


        #LINHA 22 - BARRA 20 COM 21

    (-(v20**2)*(-bkm[22]+bsh[22]/2)) + v20*v21*(-bkm[22]*sep.cos(theta20-theta21) - gkm[22]*sep.sin(theta20-theta21))) == 0


    )


    ############################ BARRA 21 - ATIVA

    sep.Equation(

        pg[21] - pc[21] - (

        #LINHA 21 - BARRA 9 COM 21

        gkm[21]*(v21**2) - v21*v9*(gkm[21]*sep.cos(theta21-theta9)-bkm[21]*sep.sin(theta21-theta9)) +


        #LINHA 24 - BARRA 21 COM 23

        gkm[24]*(v21**2) - v21*v23*(gkm[24]*sep.cos(theta21-theta23)-bkm[24]*sep.sin(theta21-theta23)) +


        #LINHA 22 - BARRA 21 COM 20

        gkm[22]*(v21**2) - v21*v20*(gkm[22]*sep.cos(theta21-theta20)-bkm[22]*sep.sin(theta21-theta20))) == 0


            )

    ############################ BARRA 21 - REATIVA


    sep.Equation(


        qg[21] - qc[21] - (



        #LINHA 21 - BARRA 9 COM 21


    (-(v21**2)*(-bkm[21]+bsh[21]/2)) + v9*v21*(-bkm[21]*sep.cos(theta21-theta9) - gkm[21]*sep.sin(theta21-theta9)) +

        #LINHA 24 - BARRA 21 COM 23


    (-(v21**2)*(-bkm[24]+bsh[24]/2)) + v23*v21*(-bkm[24]*sep.cos(theta21-theta23) - gkm[24]*sep.sin(theta21-theta23)) +


        #LINHA 22 - BARRA 21 COM 20

    (-(v21**2)*(-bkm[22]+bsh[22]/2)) + v20*v21*(-bkm[22]*sep.cos(theta21-theta20) - gkm[22]*sep.sin(theta21-theta20))) == 0


    )
         
        ############################ BARRA 22 - ATIVA

    sep.Equation(

        pg[22] - pc[22] - (

        #LINHA 23 - BARRA 14 COM 22

        gkm[23]*(v22**2) - v22*v14*(gkm[23]*sep.cos(theta22-theta14)-bkm[23]*sep.sin(theta22-theta14)) +


        #LINHA 25 - BARRA 22 COM 23

        gkm[25]*(v22**2) - v22*v23*(gkm[25]*sep.cos(theta22-theta23)-bkm[25]*sep.sin(theta22-theta23))) == 0


            )

    ############################ BARRA 21 - REATIVA


    sep.Equation(


        qg[22] - qc[22] - (



        #LINHA 23 - BARRA 14 COM 22


    (-(v22**2)*(-bkm[23]+bsh[23]/2)) + v14*v22*(-bkm[23]*sep.cos(theta22-theta14) - gkm[23]*sep.sin(theta22-theta14)) +


        #LINHA 25 - BARRA 22 COM 23

    (-(v22**2)*(-bkm[25]+bsh[25]/2)) + v22*v23*(-bkm[25]*sep.cos(theta22-theta23) - gkm[25]*sep.sin(theta22-theta23))) == 0


    )

    ############################ BARRA 23 - ATIVA

    sep.Equation(


        pg[23] - pc[23] - (    

        #LINHA 24 - BARRA 21 COM 23

        gkm[24]*(v23**2) - v21*v23*(gkm[24]*sep.cos(theta23-theta21)-bkm[24]*sep.sin(theta23-theta21)) +


        #LINHA 26 - BARRA 23 COM 24

        gkm[26]*(v23**2) - v23*v24*(gkm[26]*sep.cos(theta23-theta24)-bkm[26]*sep.sin(theta23-theta24)) +

        #LINHA 25 - BARRA 22 COM 23

        gkm[25]*(v23**2) - v22*v23*(gkm[25]*sep.cos(theta23-theta22)-bkm[25]*sep.sin(theta23-theta22))) == 0


            )

    ############################ BARRA 23 - REATIVA


    sep.Equation(


        qg[23] - qc[23] - sh23*v23*v23 - (



        #LINHA 24 - BARRA 21 COM 23


    (-(v23**2)*(-bkm[24]+bsh[24]/2)) + v21*v23*(-bkm[24]*sep.cos(theta23-theta21) - gkm[24]*sep.sin(theta23-theta21)) +


        #LINHA 26 - BARRA 23 COM 24


    (-(v23**2)*(-bkm[26]+bsh[26]/2)) + v24*v23*(-bkm[26]*sep.cos(theta23-theta24) - gkm[26]*sep.sin(theta23-theta24)) +


        #LINHA 25 - BARRA 22 COM 23

    (-(v23**2)*(-bkm[25]+bsh[25]/2)) + v22*v23*(-bkm[25]*sep.cos(theta23-theta22) - gkm[25]*sep.sin(theta23-theta22))) == 0


    )

    ############################ BARRA 24 - ATIVA

    sep.Equation(


        pg[24] - pc[24] - (    

        #LINHA 28 - BARRA 24 COM 26

        gkm[28]*(v24**2) - v24*v26*(gkm[28]*sep.cos(theta24-theta26)-bkm[28]*sep.sin(theta24-theta26)) +


        #LINHA 26 - BARRA 23 COM 24

        gkm[26]*(v24**2) - v23*v24*(gkm[26]*sep.cos(theta24-theta23)-bkm[26]*sep.sin(theta24-theta23)) +

        #LINHA 27 - BARRA 24 COM 25

        gkm[27]*(v24**2) - v24*v25*(gkm[27]*sep.cos(theta24-theta25)-bkm[27]*sep.sin(theta24-theta25))) == 0


            )

    ############################ BARRA 24 - REATIVA


    sep.Equation(


        qg[24] - qc[24] - (


        #LINHA 28 - BARRA 24 COM 26


    (-(v24**2)*(-bkm[28]+bsh[28]/2)) + v26*v24*(-bkm[28]*sep.cos(theta24-theta26) - gkm[28]*sep.sin(theta24-theta26)) +


        #LINHA 26 - BARRA 23 COM 24


    (-(v24**2)*(-bkm[26]+bsh[26]/2)) + v24*v23*(-bkm[26]*sep.cos(theta24-theta23) - gkm[26]*sep.sin(theta24-theta23)) +


        #LINHA 27 - BARRA 24 COM 25

    (-(v24**2)*(-bkm[27]+bsh[27]/2)) + v24*v25*(-bkm[27]*sep.cos(theta24-theta25) - gkm[27]*sep.sin(theta24-theta25))) == 0


    )


    ############################ BARRA 25 - ATIVA

    sep.Equation(


        pg[25] - pc[25] - (    

        #LINHA 27 - BARRA 24 COM 25

        gkm[27]*(v25**2) - v24*v25*(gkm[27]*sep.cos(theta25-theta24)-bkm[27]*sep.sin(theta25-theta24))) == 0


            )

    ############################ BARRA 25 - REATIVA


    sep.Equation(


        qg[25] - qc[25] - (

        #LINHA 27 - BARRA 24 COM 25

    (-(v25**2)*(-bkm[27]+bsh[27]/2)) + v24*v25*(-bkm[27]*sep.cos(theta25-theta24) - gkm[27]*sep.sin(theta25-theta24))) == 0


    )

    ############################ BARRA 26 - ATIVA

    sep.Equation(


        pg[26] - pc[26] - (    

        #LINHA 28 - BARRA 24 COM 26

        gkm[28]*(v26**2) - v24*v26*(gkm[28]*sep.cos(theta26-theta24)-bkm[28]*sep.sin(theta26-theta24))+

        #LINHA 29 - BARRA 26 COM 28

        gkm[29]*(v26**2) - v26*v28*(gkm[29]*sep.cos(theta26-theta28)-bkm[29]*sep.sin(theta26-theta28))+

        #LINHA 30 - BARRA 26 COM 29

        gkm[30]*(v26**2) - v26*v29*(gkm[30]*sep.cos(theta26-theta29)-bkm[30]*sep.sin(theta26-theta29))+

        #TRAFO 6 - BARRA 27 COM 26

        -(v26*v27/tap2726)*-bkmt[6]*sep.sin(theta26-theta27)) == 0

            )

    ############################ BARRA 26 - REATIVA


    sep.Equation(


        qg[26] - qc[26] - (

        #LINHA 28 - BARRA 24 COM 26

        (-(v26**2)*(-bkm[28]+bsh[28]/2)) + v26*v24*(-bkm[28]*sep.cos(theta26-theta24) - gkm[28]*sep.sin(theta26-theta24)) +

        #LINHA 29 - BARRA 26 COM 28

        (-(v26**2)*(-bkm[29]+bsh[29]/2)) + v28*v26*(-bkm[29]*sep.cos(theta26-theta28) - gkm[29]*sep.sin(theta26-theta28)) +

        #LINHA 30 - BARRA 26 COM 29

        (-(v26**2)*(-bkm[30]+bsh[30]/2)) + v26*v29*(-bkm[30]*sep.cos(theta26-theta29) - gkm[30]*sep.sin(theta26-theta29)) +

        #TRAFO 6 - BARRA 27 COM 26
       -(-bkmt[6])*v26**2 + (v27*v26/tap2726)*-bkmt[6]*sep.cos(theta26-theta27)) == 0
    )

    ############################ BARRA 27 - ATIVA

    sep.Equation(


        pg[27] - pc[27] - (    

        #LINHA 32 - BARRA 7 COM 27

        gkm[32]*(v27**2) - v7*v27*(gkm[32]*sep.cos(theta27-theta7)-bkm[32]*sep.sin(theta27-theta7))+

        #LINHA 33 - BARRA 5 COM 27

        gkm[33]*(v27**2) - v27*v5*(gkm[33]*sep.cos(theta27-theta5)-bkm[33]*sep.sin(theta27-theta5))+

        #TRAFO 6 - BARRA 27 COM 26

        -(v27*v26/tap2726)*-bkmt[6]*sep.sin(theta27-theta26)) == 0

            )

    ############################ BARRA 27 - REATIVA


    sep.Equation(


        qg[27] - qc[27] - (

        #LINHA 32 - BARRA 7 COM 27

        (-(v27**2)*(-bkm[32]+bsh[32]/2)) + v27*v7*(-bkm[32]*sep.cos(theta27-theta7) - gkm[32]*sep.sin(theta27-theta7)) +

        #LINHA 33 - BARRA 5 COM 27

        (-(v27**2)*(-bkm[33]+bsh[33]/2)) + v27*v5*(-bkm[33]*sep.cos(theta27-theta5) - gkm[33]*sep.sin(theta27-theta5)) +


        #TRAFO 6 - BARRA 27 COM 26
        -(-bkmt[6]/tap2726**2)*v27**2 + (v27*v26/tap2726)*-bkmt[6]*sep.cos(theta27-theta26)) == 0
    )


    ############################ BARRA 28 - ATIVA

    sep.Equation(

        pg[28] - pc[28] - (

        #LINHA 29 - BARRA 26 COM 28

        gkm[29]*(v28**2) - v28*v26*(gkm[29]*sep.cos(theta28-theta26)-bkm[29]*sep.sin(theta28-theta26)) +


        #LINHA 31 - BARRA 28 COM 29

        gkm[31]*(v28**2) - v28*v29*(gkm[31]*sep.cos(theta28-theta29)-bkm[31]*sep.sin(theta28-theta29))) == 0


            )

    ############################ BARRA 28 - REATIVA


    sep.Equation(


        qg[28] - qc[28] - (


        #LINHA 29 - BARRA 26 COM 28


    (-(v28**2)*(-bkm[29]+bsh[29]/2)) + v26*v28*(-bkm[29]*sep.cos(theta28-theta26) - gkm[29]*sep.sin(theta28-theta26)) +


        #LINHA 31 - BARRA 28 COM 29

    (-(v28**2)*(-bkm[31]+bsh[31]/2)) + v28*v29*(-bkm[31]*sep.cos(theta28-theta29) - gkm[31]*sep.sin(theta28-theta29))) == 0


    )

    ############################ BARRA 29 - ATIVA

    sep.Equation(

        pg[29] - pc[29] - (

        #LINHA 30 - BARRA 26 COM 29

        gkm[30]*(v29**2) - v29*v26*(gkm[30]*sep.cos(theta29-theta26)-bkm[30]*sep.sin(theta29-theta26)) +


        #LINHA 31 - BARRA 28 COM 29

        gkm[31]*(v29**2) - v28*v29*(gkm[31]*sep.cos(theta29-theta28)-bkm[31]*sep.sin(theta29-theta28))) == 0


            )

    ############################ BARRA 29 - REATIVA


    sep.Equation(


        qg[29] - qc[29] - (


        #LINHA 30 - BARRA 26 COM 29


    (-(v29**2)*(-bkm[30]+bsh[30]/2)) + v26*v29*(-bkm[30]*sep.cos(theta29-theta26) - gkm[30]*sep.sin(theta29-theta26)) +


        #LINHA 31 - BARRA 28 COM 29

    (-(v29**2)*(-bkm[31]+bsh[31]/2)) + v28*v29*(-bkm[31]*sep.cos(theta29-theta28) - gkm[31]*sep.sin(theta29-theta28))) == 0


    )
    
    ################################################## FUNÇÃO OBJETIVO (PERDAS DE POTÊNCIA ATIVA)       



    sep.Obj( 


        # 0 -> LINHA BARRA 0 - BARRA 1

        gkm[0]*((v0**2) + (v1**2) - 2*v0*v1*sep.cos(0-theta1)) +

        # 1 -> LINHA BARRA 0 - BARRA 2


        gkm[1]*((v0**2) + (v2**2) - 2*v0*v2*sep.cos(0-theta2)) +

        # 2 -> LINHA BARRA 1 - BARRA 3


        gkm[2]*((v1**2) + (v3**2) - 2*v1*v3*sep.cos(theta1-theta3)) +

        # 3 -> LINHA BARRA 2 - BARRA 3

        gkm[3]*((v2**2) + (v3**2) - 2*v2*v3*sep.cos(theta2-theta3)) +

        # 4 -> LINHA BARRA 1 - BARRA 4

        gkm[4]*((v1**2) + (v4**2) - 2*v1*v4*sep.cos(theta1-theta4)) +


        # 5 -> LINHA BARRA 1 - BARRA 5

        gkm[5]*((v1**2) + (v5**2) - 2*v1*v5*sep.cos(theta1-theta5)) +

        # 6 -> LINHA BARRA 3 - BARRA 5

        gkm[6]*((v5**2) + (v3**2) - 2*v5*v3*sep.cos(theta3-theta5)) +

        # 7 -> LINHA BARRA 4 - BARRA 6

        gkm[7]*((v4**2) + (v6**2) - 2*v4*v6*sep.cos(theta4-theta6)) +

        # 8 -> LINHA BARRA 5 - BARRA 6

        gkm[8]*((v5**2) + (v6**2) - 2*v5*v6*sep.cos(theta5-theta6)) +

        # 9 -> LINHA BARRA 5 - BARRA 7

        gkm[9]*((v5**2) + (v7**2) - 2*v5*v7*sep.cos(theta5-theta7)) +

        # 10 -> LINHA BARRA 11 - BARRA 13

        gkm[10]*((v11**2) + (v13**2) - 2*v11*v13*sep.cos(theta11-theta13)) +

        # 11 -> LINHA BARRA 11 - BARRA 14

        gkm[11]*((v11**2) + (v14**2) - 2*v11*v14*sep.cos(theta11-theta14)) +

        # 12 -> LINHA BARRA 11 - BARRA 15

        gkm[12]*((v11**2) + (v15**2) - 2*v11*v15*sep.cos(theta11-theta15)) +

        # 13 -> LINHA BARRA 13 - BARRA 14

        gkm[13]*((v13**2) + (v14**2) - 2*v13*v14*sep.cos(theta13-theta14)) +

        # 14 -> LINHA BARRA 15 - BARRA 16

        gkm[14]*((v15**2) + (v16**2) - 2*v15*v16*sep.cos(theta15-theta16)) +

        # 15 -> LINHA BARRA 14 - BARRA 17

        gkm[15]*((v14**2) + (v17**2) - 2*v14*v17*sep.cos(theta14-theta17)) +

        # 16 -> LINHA BARRA 17 - BARRA 18

        gkm[16]*((v17**2) + (v18**2) - 2*v17*v18*sep.cos(theta17-theta18)) +

        # 17 -> LINHA BARRA 18 - BARRA 19

        gkm[17]*((v18**2) + (v19**2) - 2*v18*v19*sep.cos(theta18-theta19)) +

        # 18 -> LINHA BARRA 9 - BARRA 19

        gkm[18]*((v9**2) + (v19**2) - 2*v9*v19*sep.cos(theta9-theta19)) +

        # 19 -> LINHA BARRA 9 - BARRA 16

        gkm[19]*((v9**2) + (v16**2) - 2*v9*v16*sep.cos(theta9-theta16)) +

        # 20 -> LINHA BARRA 9 - BARRA 20

        gkm[20]*((v9**2) + (v20**2) - 2*v9*v20*sep.cos(theta9-theta20)) +

        # 21 -> LINHA BARRA 9 - BARRA 21

        gkm[21]*((v9**2) + (v21**2) - 2*v9*v21*sep.cos(theta9-theta21)) +

        # 22 -> LINHA BARRA 20 - BARRA 21

        gkm[22]*((v20**2) + (v21**2) - 2*v20*v21*sep.cos(theta20-theta21)) +

        # 23 -> LINHA BARRA 14 - BARRA 22

        gkm[23]*((v14**2) + (v22**2) - 2*v14*v22*sep.cos(theta14-theta22)) +

        # 24 -> LINHA BARRA 21 - BARRA 23

        gkm[24]*((v21**2) + (v23**2) - 2*v21*v23*sep.cos(theta21-theta23)) +

        # 25 -> LINHA BARRA 22 - BARRA 23

        gkm[25]*((v22**2) + (v23**2) - 2*v22*v23*sep.cos(theta22-theta23)) +

        # 26 -> LINHA BARRA 23 - BARRA 24

        gkm[26]*((v23**2) + (v24**2) - 2*v23*v24*sep.cos(theta23-theta24)) +

        # 27 -> LINHA BARRA 24 - BARRA 25

        gkm[27]*((v24**2) + (v25**2) - 2*v24*v25*sep.cos(theta24-theta25)) +

        # 28 -> LINHA BARRA 24 - BARRA 26

        gkm[28]*((v24**2) + (v26**2) - 2*v24*v26*sep.cos(theta24-theta26)) +

        # 29 -> LINHA BARRA 26 - BARRA 28

        gkm[29]*((v26**2) + (v28**2) - 2*v26*v28*sep.cos(theta26-theta28)) +

        # 30 -> LINHA BARRA 26 - BARRA 29

        gkm[30]*((v26**2) + (v29**2) - 2*v26*v29*sep.cos(theta26-theta29)) +

        # 31 -> LINHA BARRA 28 - BARRA 29

        gkm[31]*((v28**2) + (v29**2) - 2*v28*v29*sep.cos(theta28-theta29)) +

        # 32 -> LINHA BARRA 7 - BARRA 27

        gkm[32]*((v7**2) + (v27**2) - 2*v7*v27*sep.cos(theta7-theta27)) +

        # 33 -> LINHA BARRA 5 - BARRA 27

        gkm[33]*((v5**2) + (v27**2) - 2*v5*v27*sep.cos(theta5-theta27))

        )
    sep.options.WEB = 0
    sep.options.SOLVER = solver
    
    sep.options.MAX_ITER = max_iter
    
    sep.options.RTOL = rtol
    
    sep.options.RTOL = otol
      
    sep.solve(disp=relatorio, server='https://byu.apmonitor.com')
    
    if travado == False:
    
        solution = np.abs(np.array([v1.value[0], v4.value[0], v7.value[0], v10.value[0], v12.value[0], tap58.value[0], tap59.value[0], tap311.value[0], tap2726.value[0], sh9.value[0],sh23.value[0]]))

        if arredondado == True:
            
            solution[5:5+4] = discreto_tap(solution,4,5,2,sep30)
            solution[4+5:5+4+2]=discreto_bshunt(solution,4,5,2,sep30)
            
            
    else:
        
        solution = np.abs(np.array([v1.value[0], v4.value[0], v7.value[0], v10.value[0], v12.value[0], tap58, tap59, tap311, tap2726, sh9,sh23]))

    return np.abs(solution), sep.options.SOLVETIME, sep.options.OBJFCNVAL,v0.value[0]
      
   


 ################################################################################################################################################################################################


def ipm_ieee14 (sep14, solver, rtol, otol, max_iter, relatorio = True, remoto = True, arredondado = True, travado = True, minlp = True):
    
    sep = GEKKO(remote=remoto)

    sep14.res_bus = sep14.res_bus.sort_index()

    Sbase = 100

    ########################################################################### Vetor de tensões das barras

    v = np.ones(len(sep14.bus))

    #v = sep14.res_bus['vm_pu'].to_numpy()

    ########################################################################### Vetor de ângulos das barras

    theta = np.zeros(len(sep14.bus))

    #theta = np.radians(sep14.res_bus['va_degree'].to_numpy())

    ########################################################################### Vetor de potência ativa gerada

    pg = np.zeros(len(sep14.bus))

    i = 0

    sep14.gen = sep14.gen.sort_index()

    sep14.res_gen = sep14.res_gen.sort_index()

    for bus in sep14.gen['bus'].to_numpy():

        pg[bus] = sep14.gen['p_mw'].to_numpy()[i]/Sbase

        i = i+1

    pg[0] = sep14.res_ext_grid['p_mw'].to_numpy()/100

    pg_ls = sep14.ext_grid['max_p_mw'].to_numpy()/100

    pg_li = sep14.ext_grid['min_p_mw'].to_numpy()/100


    ########################################################################### Vetor de potência reativa gerada

    qg = np.zeros(len(sep14.bus))

    i = 0

    sep14.gen = sep14.gen.sort_index()

    for bus in sep14.gen['bus'].to_numpy():

        qg[bus] = sep14.res_gen['q_mvar'].to_numpy()[i]/Sbase

        i = i+1

    qg[0] = sep14.res_ext_grid['q_mvar'][0]/Sbase


    ########################################################################### Vetores de limite de potência reativa

    qg_ls = np.zeros(len(sep14.bus))
    qg_li = np.zeros(len(sep14.bus))

    i=0

    for bus in sep14.gen['bus'].to_numpy():

        qg_ls[bus] = sep14.gen['max_q_mvar'].to_numpy()[i]/Sbase
        qg_li[bus] = sep14.gen['min_q_mvar'].to_numpy()[i]/Sbase


        i=i+1

    qg_ls[0] = sep14.ext_grid['max_q_mvar'].to_numpy()/Sbase

    qg_li[0] = sep14.ext_grid['min_q_mvar'].to_numpy()/Sbase


    ########################################################################### Vetor de potência ativa consumida


    pc = np.zeros(len(sep14.bus))

    i = 0

    sep14.load = sep14.load.sort_index()

    for bus in sep14.load['bus'].to_numpy():

        pc[bus] = sep14.load['p_mw'].to_numpy()[i]/Sbase

        i=i+1


    ########################################################################### Vetor de potência reativa consumida

    qc = np.zeros(len(sep14.bus))

    i = 0

    for bus in sep14.load['bus'].to_numpy():

        qc[bus] = sep14.load['q_mvar'].to_numpy()[i]/Sbase

        i=i+1

    ########################################################################### Vetor de MVAR Shunt

    sh = np.zeros(len(sep14.bus))

    i = 0

    sep14.shunt = sep14.shunt.sort_index()

    for bus in sep14.shunt['bus'].to_numpy():

        sh[bus] = sep14.shunt['q_mvar'].to_numpy()[i]/Sbase

        i=i+1


    ########################################################################### Vetores de condutância e susceptância série

    m_z = np.zeros((5,len(sep14.line)))

    gkm = np.zeros(len(sep14.line))

    bkm = np.zeros(len(sep14.line))

    bo = np.zeros(len(sep14.line))

    bd = np.zeros(len(sep14.line))

    sep14.line = sep14.line.sort_index()

    sep14.bus = sep14.bus.sort_index()

    vbus = sep14.bus.vn_kv.to_numpy(dtype=np.float64)

    zbase = np.power(np.multiply(vbus,1000), 2)/(100*1e6)

    m_z[0,:] = sep14.line.from_bus.to_numpy()

    m_z[1,:] = sep14.line.to_bus.to_numpy()

    bsh = 1e-9*(2*np.pi*60*sep14.line.c_nf_per_km.to_numpy())

    m_z[4,:] = bsh


    for i in range(len(sep14.line.index.ravel())):    

        m_z[2,i] = sep14.line.r_ohm_per_km[i]/zbase[int(m_z[0,i])]

        m_z[3,i] = sep14.line.x_ohm_per_km[i]/zbase[int(m_z[0,i])]

        m_z[4,i] =  m_z[4,i] * zbase[int(m_z[0,i])]


    gkm = np.array(np.divide(m_z[2,:], np.power(m_z[2,:],2)+np.power(m_z[3],2)))

    bo = m_z[0,:]

    bd = m_z[1,:]

    ########################################################################### Vetor de susceptância

    bkm = np.array(np.divide(m_z[3,:], np.power(m_z[2,:],2)+np.power(m_z[3],2)))


    ########################################################################### Vetor de susceptância shunt

    bsh = m_z[4,:]

    ########################################################################### Vetores de limite de shunt




    ########################################################################### Vetor de tap

    tap_pos = sep14.trafo[~pd.isnull(sep14.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)

    tap_neutral = sep14.trafo[~pd.isnull(sep14.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)

    tap_step_percent = sep14.trafo[~pd.isnull(sep14.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)

    valor_percentual = (tap_pos-tap_neutral)*(tap_step_percent/100) + 1

    valor_percentual = np.resize(valor_percentual,(len(sep14.trafo)))

    to = sep14.trafo['hv_bus'].to_numpy()


    td = sep14.trafo['lv_bus'].to_numpy()

    i = 0

    for i in range(len(valor_percentual)):

        if i < len(tap_pos):

            valor_percentual[i] = valor_percentual[i]

        else:

            valor_percentual[i] = 1


    tap = valor_percentual

    ########################################################################### Vetores de limite de tap

    tap_max = np.ones(len(tap))*1.12


    tap_min = np.ones(len(tap))*0.88
    sep = GEKKO(remote=True)

    ################################################### TENSÕES

    vmax = 1.05
    vmin = 0.95

    v00 = v[0]

    v0 = sep.Var(value=v00,lb=vmin,ub=vmax)
    
    v1 = sep.Var(value=v[1],lb=vmin,ub=vmax)


    v2 = sep.Var(value=v[2],lb=vmin,ub=vmax)


    v3 = sep.Var(value=v[3],lb=vmin,ub=vmax)


    v4 = sep.Var(value=v[4],lb=vmin,ub=vmax)


    v5 = sep.Var(value=v[5],lb=vmin,ub=vmax)


    v6 = sep.Var(value=v[6],lb=vmin,ub=vmax)


    v7 = sep.Var(value=v[7],lb=vmin,ub=vmax)


    v8 = sep.Var(value=v[8],lb=vmin,ub=vmax)


    v9 = sep.Var(value=v[9],lb=vmin,ub=vmax)


    v10 = sep.Var(value=v[10],lb=vmin,ub=vmax)


    v11 = sep.Var(value=v[11],lb=vmin,ub=vmax)


    v12 = sep.Var(value=v[12],lb=vmin,ub=vmax)


    v13 = sep.Var(value=v[13],lb=vmin,ub=vmax)



    ################################################## ÂNGULOS


    theta0 = 0
        
    theta1 = sep.Var(value=theta[1],lb=-np.pi,ub=np.pi)

    theta2 = sep.Var(value=theta[2],lb=-np.pi,ub=np.pi)

    theta3 = sep.Var(value=theta[3],lb=-np.pi,ub=np.pi)

    theta4 = sep.Var(value=theta[4],lb=-np.pi,ub=np.pi)

    theta5 = sep.Var(value=theta[5],lb=-np.pi,ub=np.pi)

    theta6 = sep.Var(value=theta[6],lb=-np.pi,ub=np.pi)

    theta7 = sep.Var(value=theta[7],lb=-np.pi,ub=np.pi)

    theta8 = sep.Var(value=theta[8],lb=-np.pi,ub=np.pi)

    theta9 = sep.Var(value=theta[9],lb=-np.pi,ub=np.pi)

    theta10 = sep.Var(value=theta[10],lb=-np.pi,ub=np.pi)

    theta11 = sep.Var(value=theta[11],lb=-np.pi,ub=np.pi)

    theta12 = sep.Var(value=theta[12],lb=-np.pi,ub=np.pi)

    theta13 = sep.Var(value=theta[13],lb=-np.pi,ub=np.pi)


    ################################################## POTÊNCIA ATIVA BARRA DE REFERÊNCIA


    pg0 = sep.Var(value = pg[0], lb = pg_li[0], ub = pg_ls[0])


    ################################################## POTÊNCIA REATIVA BARRA DE REFERÊNCIA


    qg0 = sep.Var(value = qg[0],lb = qg_li[0],ub = qg_ls[0])


    ################################################## POTÊNCIAS REATIVA GERADA DAS DEMAIS BARRAS

    qg1 = sep.Var(value = qg[1],lb = qg_li[1],ub = qg_ls[1])


    qg2 = sep.Var(value = qg[2],lb = qg_li[2],ub = qg_ls[2])


    qg5 = sep.Var(value = qg[5],lb = qg_li[5],ub = qg_ls[5])


    qg7 = sep.Var(value = qg[7],lb = qg_li[7],ub = qg_ls[7])


    ################################################## POTÊNCIAS REATIVA GERADA DAS DEMAIS BARRAS
    
    if travado == True:
    
        tap36 = tap[0] #sep.Var(value = tap[0],lb = tap_min[0], ub = tap_max[0])


        tap38 = tap[1] #sep.Var(value = tap[1],lb = tap_min[1], ub = tap_max[1])


        tap45 = tap[2] #sep.Var(value = tap[2],lb = tap_min[2], ub = tap_max[2])

    
        sh8 = sh[8] #sep.Var(value = sh[8],lb = -0.39, ub = 0)  
        

    else:

        
        if minlp == True:
         

            tap36 = sep.sos1([0.9    , 0.90625, 0.9125 , 0.91875, 0.925  , 0.93125, 0.9375 ,
               0.94375, 0.95   , 0.95625, 0.9625 , 0.96875, 0.975  , 0.98125,
               0.9875 , 0.99375, 1.     , 1.00625, 1.0125 , 1.01875, 1.025  ,
               1.03125, 1.0375 , 1.04375, 1.05   , 1.05625, 1.0625 , 1.06875,
               1.075  , 1.08125, 1.0875 , 1.09375, 1.1 ])


            tap38 = sep.sos1([0.9    , 0.90625, 0.9125 , 0.91875, 0.925  , 0.93125, 0.9375 ,
               0.94375, 0.95   , 0.95625, 0.9625 , 0.96875, 0.975  , 0.98125,
               0.9875 , 0.99375, 1.     , 1.00625, 1.0125 , 1.01875, 1.025  ,
               1.03125, 1.0375 , 1.04375, 1.05   , 1.05625, 1.0625 , 1.06875,
               1.075  , 1.08125, 1.0875 , 1.09375, 1.1 ])


            tap45 = sep.sos1([0.9    , 0.90625, 0.9125 , 0.91875, 0.925  , 0.93125, 0.9375 ,
               0.94375, 0.95   , 0.95625, 0.9625 , 0.96875, 0.975  , 0.98125,
               0.9875 , 0.99375, 1.     , 1.00625, 1.0125 , 1.01875, 1.025  ,
               1.03125, 1.0375 , 1.04375, 1.05   , 1.05625, 1.0625 , 1.06875,
               1.075  , 1.08125, 1.0875 , 1.09375, 1.1 ])

            

            tap67 = 1

            tap68 = 1


            sh8 = sep.sos1([0, -0.2, -0.05, -0.15, -0.19, -0.24, -0.34, -0.39])                     

            sh8.value = sh[8]
            
            tap36.value = tap[0]
            
            tap38.value = tap[1]
            
            tap45.value = tap[2]
            
       
        else:
     
        
            tap36 = sep.Var(value = tap[0],lb = tap_min[0], ub = tap_max[0])

            tap38 = sep.Var(value = tap[1],lb = tap_min[1], ub = tap_max[1])

            tap45 = sep.Var(value = tap[2],lb = tap_min[2], ub = tap_max[2])

            sh8 = sep.Var(value = sh[8],lb = -0.39, ub = 0)                        

        tap67 = 1

        tap68 = 1
        
    ################################################## Bshunt Trafo


    sep14.trafo = sep14.trafo.sort_index()

    barras = sep14.trafo['hv_bus'].to_numpy()

    xkm = (sep14.trafo['vk_percent'].to_numpy()/100)*(1000/sep14.trafo['sn_mva'].to_numpy())

    bkmt = 10/xkm
    


    ################################################## EQUAÇÕES DO FLUXO DE CARGA (POTÊNCIA ATIVA E REATIVA)

    ############################ BARRA 0 - ATIVA

    sep.Equation(

        pg0 - pc[0] - 


        # LINHA 0 - BARRA 0 COM 1

        (gkm[0]*(v0**2) - v0*v1*(gkm[0]*sep.cos(theta0-theta1) - bkm[0]*sep.sin(theta0-theta1)) +  

        # LINHA 1 - BARRA 0 COM 4

         gkm[1]*(v0**2) - v0*v4*(gkm[1]*sep.cos(theta0-theta4)-bkm[1]*sep.sin(theta0-theta4))) ==0


            )


    ############################ BARRA 0 - REATIVA


    sep.Equation(


        qg0 - qc[0] - 

        # LINHA 0 - BARRA 0 COM 1

        ((-(v0**2)*(-bkm[0]+bsh[0]/2)) + v0*v1*(-bkm[0]*sep.cos(theta0-theta1) - gkm[0]*sep.sin(theta0-theta1)) +

        # LINHA 1 - BARRA 0 COM 4


        (-(v0**2)*(-bkm[1]+bsh[1]/2)) + v0*v4*(-bkm[1]*sep.cos(theta0-theta4) - gkm[1]*sep.sin(theta0-theta4))) == 0


            )


    ############################ BARRA 1 - ATIVA

    sep.Equation(

        pg[1] - pc[1] - 

        # LINHA 0 - BARRA 1 COM 0

        (gkm[0]*(v1**2) - v0*v1*(gkm[0]*sep.cos(theta1-theta0) - bkm[0]*sep.sin(theta1-theta0)) +  

        # LINHA 2 - BARRA 1 COM 2

         gkm[2]*(v1**2) - v1*v2*(gkm[2]*sep.cos(theta1-theta2)-bkm[2]*sep.sin(theta1-theta2)) +

         # LINHA 3 - BARRA 1 COM 3

         gkm[3]*(v1**2) - v1*v3*(gkm[3]*sep.cos(theta1-theta3)-bkm[3]*sep.sin(theta1-theta3)) +

         # LINHA 4 - BARRA 1 COM 4

         gkm[4]*(v1**2) - v1*v4*(gkm[4]*sep.cos(theta1-theta4)-bkm[4]*sep.sin(theta1-theta4))) == 0


            )


    ############################ BARRA 1 - REATIVA


    sep.Equation(


        qg1 - qc[1] - 

        # LINHA 0 - BARRA 1 COM 0

        ((-(v1**2)*(-bkm[0]+bsh[0]/2)) + v1*v0*(-bkm[0]*sep.cos(theta1-theta0) - gkm[0]*sep.sin(theta1-theta0)) +

        # LINHA 2 - BARRA 1 COM 2


        (-(v1**2)*(-bkm[2]+bsh[2]/2)) + v1*v2*(-bkm[2]*sep.cos(theta1-theta2) - gkm[2]*sep.sin(theta1-theta2)) +

        # LINHA 3 - BARRA 1 COM 3

        (-(v1**2)*(-bkm[3]+bsh[3]/2)) + v1*v3*(-bkm[3]*sep.cos(theta1-theta3) - gkm[3]*sep.sin(theta1-theta3)) +

        # LINHA 4 - BARRA 1 COM 4

        (-(v1**2)*(-bkm[4]+bsh[4]/2)) + v1*v4*(-bkm[4]*sep.cos(theta1-theta4) - gkm[4]*sep.sin(theta1-theta4)))  == 0

        )


    ############################ BARRA 2 - ATIVA

    sep.Equation(

        pg[2] - pc[2] - 

        # LINHA 2 - BARRA 2 COM 1

        (gkm[2]*(v2**2) - v2*v1*(gkm[2]*sep.cos(theta2-theta1) - bkm[2]*sep.sin(theta2-theta1)) +  

        # LINHA 5 - BARRA 2 COM 3

         gkm[5]*(v2**2) - v2*v3*(gkm[5]*sep.cos(theta2-theta3)-bkm[5]*sep.sin(theta2-theta3))) == 0


            )


    ############################ BARRA 2 - REATIVA


    sep.Equation(


        qg2 - qc[2] - 

        # LINHA 2 - BARRA 2 COM 1

        ((-(v2**2)*(-bkm[2]+bsh[2]/2)) + v2*v1*(-bkm[2]*sep.cos(theta2-theta1) - gkm[2]*sep.sin(theta2-theta1)) +


        # LINHA 5 - BARRA 2 COM 3


        (-(v2**2)*(-bkm[5]+bsh[5]/2)) + v3*v2*(-bkm[5]*sep.cos(theta2-theta3) - gkm[5]*sep.sin(theta2-theta3))) == 0

        )


    ############################ BARRA 3 - ATIVA

    sep.Equation(

        pg[3] - pc[3] - 

        # LINHA 3 - BARRA 3 COM 1

        (gkm[3]*(v3**2) - v3*v1*(gkm[3]*sep.cos(theta3-theta1) - bkm[3]*sep.sin(theta3-theta1)) +  


        # LINHA 5 - BARRA 3 COM 2

        gkm[5]*(v3**2) - v3*v2*(gkm[5]*sep.cos(theta3-theta2) - bkm[5]*sep.sin(theta3-theta2)) +  


        # LINHA 6 - BARRA 3 COM 4

         gkm[6]*(v3**2) - v4*v3*(gkm[6]*sep.cos(theta3-theta4)-bkm[6]*sep.sin(theta3-theta4))+

        # TRAFO 0 - BARRA 3 COM 6

        -(v3*v6/tap36)*-bkmt[0]*sep.sin(theta3-theta6) + 

        # TRAFO 1 - BARRA 3 COM 8

         -(v3*v8/tap38)*-bkmt[1]*sep.sin(theta3-theta8)) == 0

            )



    ############################ BARRA 3 - REATIVA


    sep.Equation(


        qg[3] - qc[3] - 


        # LINHA 3 - BARRA 3 COM 1

        ((-(v3**2)*(-bkm[3]+bsh[3]/2)) + v3*v1*(-bkm[3]*sep.cos(theta3-theta1) - gkm[3]*sep.sin(theta3-theta1)) +


        # LINHA 5 - BARRA 3 COM 2


        (-(v3**2)*(-bkm[5]+bsh[5]/2)) + v3*v2*(-bkm[5]*sep.cos(theta3-theta2) - gkm[5]*sep.sin(theta3-theta2)) +


         # LINHA 6 - BARRA 3 COM 4

        (-(v3**2)*(-bkm[6]+bsh[6]/2)) + v3*v4*(-bkm[6]*sep.cos(theta3-theta4) - gkm[6]*sep.sin(theta3-theta4))+


         # TRAFO 0 - BARRA 3 COM 6

        -(-bkmt[0]/tap36**2)*v3**2 + (v3*v6/tap36)*-bkmt[0]*sep.cos(theta3-theta6)+


         # TRAFO 1 - BARRA 3 COM 8

        -(-bkmt[1]/tap38**2)*v3**2 + (v3*v8/tap38)*-bkmt[1]*sep.cos(theta3-theta8)) == 0

        )


    ############################ BARRA 4 - ATIVA

    sep.Equation(

        pg[4] - pc[4] - 

        # LINHA 1 - BARRA 4 COM 0

        (gkm[1]*(v4**2) - v4*v0*(gkm[1]*sep.cos(theta4-theta0) - bkm[1]*sep.sin(theta4-theta0)) +  


        # LINHA 4 - BARRA 4 COM 1

        gkm[4]*(v4**2) - v4*v1*(gkm[4]*sep.cos(theta4-theta1) - bkm[4]*sep.sin(theta4-theta1)) +  


        # LINHA 6 - BARRA 4 COM 3

         gkm[6]*(v4**2) - v4*v3*(gkm[6]*sep.cos(theta4-theta3)-bkm[6]*sep.sin(theta4-theta3))+

        # TRAFO 2 - BARRA 4 COM 5

        -(v4*v5/tap45)*-bkmt[2]*sep.sin(theta4-theta5)) == 0


            )



    ############################ BARRA 4 - REATIVA


    sep.Equation(


        qg[4] - qc[4] - 


        # LINHA 1 - BARRA 4 COM 0

        ((-(v4**2)*(-bkm[1]+bsh[1]/2)) + v4*v0*(-bkm[1]*sep.cos(theta4-theta0) - gkm[1]*sep.sin(theta4-theta0)) +

        # LINHA 4 - BARRA 4 COM 1

        (-(v4**2)*(-bkm[4]+bsh[4]/2)) + v4*v1*(-bkm[4]*sep.cos(theta4-theta1) - gkm[4]*sep.sin(theta4-theta1)) +

        # LINHA 6 - BARRA 4 COM 3

        (-(v4**2)*(-bkm[6]+bsh[6]/2)) + v3*v4*(-bkm[6]*sep.cos(theta4-theta3) - gkm[6]*sep.sin(theta4 - theta3)) +

        # TRAFO 2 - BARRA 4 COM 5

        -(-bkmt[2]/tap45**2)*v4**2 + (v4*v5/tap45)*-bkmt[2]*sep.cos(theta4-theta5)) == 0


        )


    ############################ BARRA 5 - ATIVA

    sep.Equation(


        pg[5] - pc[5] - 

        # LINHA 7 - BARRA 5 COM 10

        (gkm[7]*(v5**2) - v5*v10*(gkm[7]*sep.cos(theta5-theta10) - bkm[7]*sep.sin(theta5-theta10)) +  


        # LINHA 8 - BARRA 5 COM 11

        gkm[8]*(v5**2) - v5*v11*(gkm[8]*sep.cos(theta5-theta11) - bkm[8]*sep.sin(theta5-theta11)) +  


        # LINHA 9 - BARRA 5 COM 12

         gkm[9]*(v5**2) - v5*v12*(gkm[9]*sep.cos(theta5-theta12)-bkm[9]*sep.sin(theta5-theta12))+

        # TRAFO 2 - BARRA 5 COM 4

        -(v4*v5/tap45)*-bkmt[2]*sep.sin(theta5-theta4)) == 0


            )



    ############################ BARRA 5 - REATIVA


    sep.Equation(


        qg5 - qc[5] - 

        # LINHA 7 - BARRA 5 COM 10

        ((-(v5**2)*(-bkm[7]+bsh[7]/2)) + v5*v10*(-bkm[7]*sep.cos(theta5-theta10) - gkm[7]*sep.sin(theta5-theta10)) +

        # LINHA 8 - BARRA 5 COM 11

        (-(v5**2)*(-bkm[8]+bsh[8]/2)) + v5*v11*(-bkm[8]*sep.cos(theta5-theta11) - gkm[8]*sep.sin(theta5-theta11)) +

        # LINHA 9 - BARRA 5 COM 12

        (-(v5**2)*(-bkm[9]+bsh[9]/2)) + v5*v12*(-bkm[9]*sep.cos(theta5-theta12) - gkm[9]*sep.sin(theta5 - theta12)) +

        # TRAFO 2 - BARRA 4 COM 5

        -(-bkmt[2])*v5**2 + (v4*v5/tap45)*-bkmt[2]*sep.cos(theta5-theta4)) == 0


        )

    ############################ BARRA 6 - ATIVA

    sep.Equation(


        pg[6] - pc[6] - 

        # TRAFO 3 - BARRA 6 COM 7

        (-(v3*v6/tap36)*-bkmt[0]*sep.sin(theta6-theta3) + 

         # TRAFO 3 - BARRA 6 COM 7

        -(v6*v7/1)*-bkmt[3]*sep.sin(theta6-theta7) + 

         # TRAFO 4 - BARRA 6 COM 8

        -(v6*v8/1)*-bkmt[4]*sep.sin(theta6-theta8)) == 0


            )



    ############################ BARRA 6 - REATIVA


    sep.Equation(


        qg[6] - qc[6] - 


        # TRAFO 0 - BARRA 6 COM 3

       (-(-bkmt[0])*v6**2 + (v6*v3/tap36)*-bkmt[0]*sep.cos(theta6-theta3) +

        # TRAFO 3 - BARRA 6 COM 7

       -(-bkmt[3])*v6**2 + (v6*v7/1)*-bkmt[3]*sep.cos(theta6-theta7) +

        # TRAFO 4 - BARRA 6 COM 8

        -(-bkmt[4])*v6**2 + (v6*v8/1)*-bkmt[4]*sep.cos(theta6-theta8)) == 0


        )

    ############################ BARRA 7 - ATIVA

    sep.Equation(

        pg[7] - pc[7] - 


         # TRAFO 3 - BARRA 6 COM 7

        (-(v6*v7/1)*-bkmt[3]*sep.sin(theta7-theta6) ) == 0


            )



    ############################ BARRA 7 - REATIVA


    sep.Equation(


        qg7 - qc[7] - 

        # TRAFO 3 - BARRA 6 COM 7

       (-(-bkmt[3])*v7**2 + (v6*v7/1)*-bkmt[3]*sep.cos(theta7-theta6) ) == 0


        )


    ############################ BARRA 8 - ATIVA

    sep.Equation(


        pg[8] - pc[8] - 


         # TRAFO 1 - BARRA 3 COM 8

        (-(v3*v8/tap38)*-bkmt[1]*sep.sin(theta8-theta3) +


         # TRAFO 4 - BARRA 6 COM 8

        -(v6*v8/1)*-bkmt[4]*sep.sin(theta8-theta6) +

        # Linha 11 - BARRA 8 COM 13

        gkm[11]*(v8**2) - v8*v13*(gkm[11]*sep.cos(theta8-theta13) - bkm[11]*sep.sin(theta8-theta13)) +  

        # Linha 10 - BARRA 8 COM 9

        gkm[10]*(v8**2) - v8*v9*(gkm[10]*sep.cos(theta8-theta9) - bkm[10]*sep.sin(theta8-theta9))) == 0


            )


    ############################ BARRA 8 - REATIVA


    sep.Equation(


       qg[8] - qc[8] - sh8*v8*v8 -

        # TRAFO 1 - BARRA 3 COM 8

       (-(-bkmt[1])*v8**2 + (v3*v8/tap38)*-bkmt[1]*sep.cos(theta8-theta3)+

        # TRAFO 4 - BARRA 6 COM 8

       -(-bkmt[4])*v8**2 + (v6*v8/1)*-bkmt[4]*sep.cos(theta8-theta6) +

        # LINHA 11 - BARRA 8 COM 13

        (-(v8**2)*(-bkm[11]+bsh[11]/2)) + v8*v13*(-bkm[11]*sep.cos(theta8-theta13) - gkm[11]*sep.sin(theta8 - theta13)) +

        # LINHA 10 - BARRA 8 COM 9

        (-(v8**2)*(-bkm[10]+bsh[10]/2)) + v9*v8*(-bkm[10]*sep.cos(theta8-theta9) - gkm[10]*sep.sin(theta8 - theta9))) == 0


        )



    ############################ BARRA 9 - ATIVA

    sep.Equation(


        pg[9] - pc[9] - 


        # Linha 10 - BARRA 8 COM 9

        (gkm[10]*(v9**2) - v8*v9*(gkm[10]*sep.cos(theta9-theta8) - bkm[10]*sep.sin(theta9-theta8)) +  

        # Linha 12 - BARRA 9 COM 10

        gkm[12]*(v9**2) - v10*v9*(gkm[12]*sep.cos(theta9-theta10) - bkm[12]*sep.sin(theta9-theta10))) == 0

    )


    ############################ BARRA 9 - REATIVA


    sep.Equation(


        qg[9] - qc[9] -

         # LINHA 10 - BARRA 8 COM 9

        ((-(v9**2)*(-bkm[10]+bsh[10]/2)) + v8*v9*(-bkm[10]*sep.cos(theta9-theta8) - gkm[10]*sep.sin(theta9 - theta8)) +

        # LINHA 12 - BARRA 9 COM 10

        (-(v9**2)*(-bkm[12]+bsh[12]/2)) + v9*v10*(-bkm[12]*sep.cos(theta9-theta10) - gkm[12]*sep.sin(theta9 - theta10))) == 0


        )


    ############################ BARRA 10 - ATIVA

    sep.Equation(


        pg[10] - pc[10] - 


        # Linha 7 - BARRA 5 COM 10

        (gkm[7]*(v10**2) - v5*v10*(gkm[7]*sep.cos(theta10-theta5) - bkm[7]*sep.sin(theta10-theta5)) +  

        # Linha 12 - BARRA 9 COM 10

        gkm[12]*(v10**2) - v10*v9*(gkm[12]*sep.cos(theta10-theta9) - bkm[12]*sep.sin(theta10-theta9))) == 0

    )


    ############################ BARRA 10 - REATIVA


    sep.Equation(


        qg[10] - qc[10] -

        # Linha 7 - BARRA 5 COM 10

        ((-(v10**2)*(-bkm[7]+bsh[7]/2)) + v5*v10*(-bkm[7]*sep.cos(theta10-theta5) - gkm[7]*sep.sin(theta10 - theta5)) +

        # Linha 12 - BARRA 9 COM 10

        (-(v10**2)*(-bkm[12]+bsh[12]/2)) + v9*v10*(-bkm[12]*sep.cos(theta10-theta9) - gkm[12]*sep.sin(theta10 - theta9))) == 0


        )




    ############################ BARRA 11 - ATIVA

    sep.Equation(


        pg[11] - pc[11] - 


        # Linha 8 - BARRA 5 COM 11

        (gkm[8]*(v11**2) - v5*v11*(gkm[8]*sep.cos(theta11-theta5) - bkm[8]*sep.sin(theta11-theta5)) +  

        # Linha 13 - BARRA 11 COM 12

        gkm[13]*(v11**2) - v11*v12*(gkm[13]*sep.cos(theta11-theta12) - bkm[13]*sep.sin(theta11-theta12))) == 0

    )


    ############################ BARRA 11 - REATIVA


    sep.Equation(


        qg[11] - qc[11] -

        # Linha 8 - BARRA 5 COM 11

        ((-(v11**2)*(-bkm[8]+bsh[8]/2)) + v11*v5*(-bkm[8]*sep.cos(theta11-theta5) - gkm[8]*sep.sin(theta11 - theta5)) +

        # Linha 13 - BARRA 11 COM 12

        (-(v11**2)*(-bkm[13]+bsh[13]/2)) + v11*v12*(-bkm[13]*sep.cos(theta11-theta12) - gkm[13]*sep.sin(theta11 - theta12))) == 0


        )

    ############################ BARRA 12 - ATIVA

    sep.Equation(


        pg[12] - pc[12] - 


        # Linha 9 - BARRA 5 COM 12

        (gkm[9]*(v12**2) - v5*v12*(gkm[9]*sep.cos(theta12-theta5) - bkm[9]*sep.sin(theta12-theta5)) +  

        # Linha 13 - BARRA 11 COM 12

         gkm[13]*(v12**2) - v12*v11*(gkm[13]*sep.cos(theta12-theta11) - bkm[13]*sep.sin(theta12-theta11)) +  

        # Linha 14 - BARRA 12 COM 13

        gkm[14]*(v12**2) - v12*v13*(gkm[14]*sep.cos(theta12-theta13) - bkm[14]*sep.sin(theta12-theta13))) == 0

    )


    ############################ BARRA 12 - REATIVA


    sep.Equation(


        qg[12] - qc[12] -


        # Linha 9 - BARRA 5 COM 12

        ((-(v12**2)*(-bkm[9]+bsh[9]/2)) + v12*v5*(-bkm[9]*sep.cos(theta12-theta5) - gkm[9]*sep.sin(theta12 - theta5)) +

        # Linha 13 - BARRA 11 COM 12

        (-(v12**2)*(-bkm[13]+bsh[13]/2)) + v11*v12*(-bkm[13]*sep.cos(theta12-theta11) - gkm[13]*sep.sin(theta12 - theta11)) +

        # Linha 14 - BARRA 12 COM 13

        (-(v12**2)*(-bkm[14]+bsh[14]/2)) + v13*v12*(-bkm[14]*sep.cos(theta12-theta13) - gkm[14]*sep.sin(theta12 - theta13))) == 0 )



    ############################ BARRA 13 - ATIVA

    sep.Equation(


        pg[13] - pc[13] - 


        # Linha 11 - BARRA 8 COM 13

        (gkm[11]*(v13**2) - v8*v13*(gkm[11]*sep.cos(theta13-theta8) - bkm[11]*sep.sin(theta13-theta8)) +  

        # Linha 14 - BARRA 12 COM 13

        gkm[14]*(v13**2) - v12*v13*(gkm[14]*sep.cos(theta13-theta12) - bkm[14]*sep.sin(theta13-theta12))) == 0

    )


    ############################ BARRA 13 - REATIVA


    sep.Equation(


        qg[13] - qc[13] -

          # Linha 11 - BARRA 8 COM 13

        ((-(v13**2)*(-bkm[11]+bsh[11]/2)) + v8*v13*(-bkm[11]*sep.cos(theta13-theta8) - gkm[11]*sep.sin(theta13 - theta8)) +

        # Linha 14 - BARRA 12 COM 13

        (-(v13**2)*(-bkm[14]+bsh[14]/2)) + v13*v12*(-bkm[14]*sep.cos(theta13-theta12) - gkm[14]*sep.sin(theta13 - theta12))) == 0

    )
    
    
    ################################################## FUNÇÃO OBJETIVO (PERDAS DE POTÊNCIA ATIVA)       

               
        
    sep.Obj( 




        # 0 -> LINHA BARRA 0 - BARRA 1

       ( gkm[0]*((v0**2) + (v1**2) - 2*v0*v1*sep.cos(0-theta1)) +

        # 1 -> LINHA BARRA 0 - BARRA 4


        gkm[1]*((v0**2) + (v4**2) - 2*v0*v4*sep.cos(0-theta4)) +  


        # 2 -> LINHA BARRA 1 - BARRA 2


        gkm[2]*((v1**2) + (v2**2) - 2*v1*v2*sep.cos(theta1 - theta2)) + 


        # 3 -> LINHA BARRA 1 - BARRA 3


        gkm[3]*((v1**2) + (v3**2) - 2*v1*v3*sep.cos(theta1 - theta3)) + 


        # 4 -> LINHA BARRA 1 - BARRA 4


        gkm[4]*((v1**2) + (v4**2) - 2*v1*v4*sep.cos(theta1 - theta4)) + 


        # 5 -> LINHA BARRA 2 - BARRA 3


        gkm[5]*((v2**2) + (v3**2) - 2*v2*v3*sep.cos(theta2 - theta3)) + 

        # 6 -> LINHA BARRA 3 - BARRA 4


        gkm[6]*((v3**2) + (v4**2) - 2*v4*v3*sep.cos(theta3 - theta4)) + 

        # 7 -> LINHA BARRA 5 - BARRA 10


        gkm[7]*((v5**2) + (v10**2) - 2*v5*v10*sep.cos(theta5 - theta10)) + 

        # 8 -> LINHA BARRA 5 - BARRA 11


        gkm[8]*((v5**2) + (v11**2) - 2*v5*v11*sep.cos(theta5 - theta11)) + 

        # 9 -> LINHA BARRA 5 - BARRA 12


        gkm[9]*((v5**2) + (v12**2) - 2*v5*v12*sep.cos(theta5 - theta12)) + 


        # 10 -> LINHA BARRA 8 - BARRA 9


        gkm[10]*((v8**2) + (v9**2) - 2*v8*v9*sep.cos(theta8 - theta9)) + 

        # 11 -> LINHA BARRA 8 - BARRA 13


        gkm[11]*((v8**2) + (v13**2) - 2*v8*v13*sep.cos(theta8 - theta13)) + 

        # 12 -> LINHA BARRA 9 - BARRA 10


        gkm[12]*((v9**2) + (v10**2) - 2*v9*v10*sep.cos(theta9 - theta10)) + 


        # 13 -> LINHA BARRA 11 - BARRA 12


        gkm[13]*((v12**2) + (v11**2) - 2*v11*v12*sep.cos(theta11 - theta12)) + 



        # 14 -> LINHA BARRA 12 - BARRA 13


        gkm[14]*((v12**2) + (v13**2) - 2*v13*v12*sep.cos(theta12 - theta13)))


        )
            
   
    sep.options.SOLVER = solver
    
    sep.options.MAX_ITER = max_iter
    
    sep.options.RTOL = rtol
    
    sep.options.RTOL = otol
      
    sep.solve(disp=relatorio)
      
    if travado == False:
    
        solution = np.abs(np.array([v1.value[0], v2.value[0], v5.value[0], v7.value[0], tap36.value[0], tap38.value[0], tap45.value[0], sh8.value[0]]))

        if arredondado == True:

            solution[4:4+3] = discreto_tap(solution,3,4,1,sep14)
            solution[4+3:4+3+1]=discreto_bshunt(solution,3,4,1,sep14)
    else:
        
        solution = np.abs(np.array([v1.value[0], v2.value[0], v5.value[0], v7.value[0], tap36, tap38, tap45, sh8]))

    return np.abs(solution), sep.options.SOLVETIME, sep.options.OBJFCNVAL, v0.value[0]




def otimizacao_pso_discreto_sengi(sep, zeta, psi, sigma, omega, max_iter, n_particles,c1,c2,v_amp,valor_inicial,relatorio=True,inicial=True):
        
    enxame_fit = cria_enxame(sep,n_particles)
    
    if inicial == True:
        
        enxame_fit[0,:]=valor_inicial     
    
    if len(sep.bus) == 14:
        
        n_vgen = 4
        n_tap = 3
        n_bshunt = 1
    
    if len(sep.bus) == 30:
        
        n_vgen = 5
        n_tap = 4
        n_bshunt = 2
        
    
    if len(sep.bus) == 118:
        
        n_vgen = 53
        n_tap = 9
        n_bshunt = 14
        
        
    if len(sep.bus) == 300:
        
        n_vgen = 68
        n_tap = 62
        n_bshunt = 29
    
    
    w_max=0.9
    
    w_min=0.4
    
    
    j = []
    
    
    tempo = []
        
    perdas = []
    
    pen_v = []
    
    pen_gq = []
    
    pen_tap = []
    
    pen_bsh = []

    
    v_lim_superior = np.repeat(sep.bus['max_vm_pu'][0], len(sep.gen))
    
    v_lim_inferior = np.repeat(sep.bus['min_vm_pu'][0], len(sep.gen))
    
    tap_pos, tap_neutral, tap_step_percent,valores_taps = coleta_dados_trafo(sep,relatorio=False)
    
    tap_max = np.repeat(valores_taps[-1], len(tap_pos))
    
    tap_min = np.repeat(valores_taps[0], len(tap_pos))
    
    bsh,b=coleta_dados_bshunt(sep)
    bsh_max=[]
    
    bsh_min=[]
    

    for bs in bsh:
        bsh_max.append([np.max(bs)])
        bsh_min.append([np.min(bs)])


    maximo = np.expand_dims(np.concatenate((v_lim_superior, tap_max, bsh_max), axis = None), 0)
    minimo = np.expand_dims(np.concatenate((v_lim_inferior, tap_min, bsh_min), axis = None), 0)
     
    
    lim_sup = np.tile(maximo, (n_particles,1))
    lim_inf = np.tile(minimo, (n_particles,1))
    
    v_anterior = v_amp*cria_enxame_v(sep,n_particles)

    for i in range(0,max_iter):
        
        start = time.time()
        
        mu, sigma = 0.5, 0.152 # mean and standard deviation

        r1 = np.random.normal(mu, sigma, size = (n_particles,enxame_fit.shape[1]))
        r2 = np.random.normal(mu, sigma, size = (n_particles,enxame_fit.shape[1]))
        
        #r1 = np.random.random_sample(size = (n_particles,enxame_fit.shape[1]))
        
        #r2 = np.random.random_sample(size = (n_particles,enxame_fit.shape[1]))
    
       
        enxame_fit = fluxo_de_pot(enxame_fit,sep)
        
        enxame_fit = fitness(enxame_fit,zeta,psi,sigma,omega)

        if i==0:
            
            best_particles = enxame_fit.copy()

            global_best = best_particles[np.argsort(best_particles[:, -1])][0,:].copy()
            
            global_matriz = np.tile(global_best, (n_particles,1))
        
            globais = []
            globais.append(0)
           
        for t in range(0,n_particles):
                
            if enxame_fit[t,-1] < best_particles[t,-1]:
                    
                best_particles[t,:] = enxame_fit[t,:].copy()
        
        global_best = best_particles[np.argsort(best_particles[:, -1])][0,:].copy()
        
        globais.append(global_best[-1])
        
        if global_best[-1] - global_best[-6] < 1e-9:
            print('Entrou: ')
            print(global_best[-1])
            print(global_best[-6])
            
            
        
            
            if (globais[-2] - global_best[-1]) > 0.00001:
                print(globais[-2])
                if len(sep.bus) == 14:

                    n_vgen = 4
                    n_tap = 3
                    n_bshunt = 1

                    validacao(sep, global_best,relatorio=False)
                    solucao_travada14, tempo_travada14, objetivo_travada14 = ipm_ieee14(sep, solver = 3, rtol = 1e-6, otol = 1e-6, max_iter = 100, relatorio = False, remoto = True, arredondado = True, travado = True, minlp=False)
                    global_best[0:n_vgen+n_tap+n_bshunt]=solucao_travada14
                    global_best[-6]=objetivo_travada14
                    global_best[-1]=objetivo_travada14
                    globais.append(global_best[-1])

                if len(sep.bus) == 30:

                    n_vgen = 5
                    n_tap = 4
                    n_bshunt = 2
                    validacao(sep,global_best,relatorio=False)
                    solucao_travada30, tempo_travada30, objetivo_travada30 = ipm_ieee30(sep, solver = 3, rtol = 1e-6, otol = 1e-6, max_iter = 100, relatorio = False, remoto = True, arredondado = True, travado = True, minlp=False)
                    global_best[0:n_vgen+n_tap+n_bshunt]=solucao_travada30
                    global_best[-6]=objetivo_travada30
                    global_best[-1]=objetivo_travada30
                    globais.append(global_best[-1])
                    print('Saiu: ', global_best[-6])
                    print(global_best)
        
       
        global_matriz = np.tile(global_best, (n_particles,1))   
       
        enxame_fit[np.argsort(best_particles[:, -1])[0],:] = global_best.copy()
        
        enxame_fit_anterior = enxame_fit.copy()
        
        w_novo = w_max-(w_max-w_min)*(i+1)/max_iter
        
        enxame_fit_novo = enxame_fit_anterior  + v_novo
        
        v_anterior = v_novo.copy()
        
        
        for linha in range(n_particles):
          
            enxame_fit_novo[linha][n_vgen:n_vgen+n_tap] = discreto_tap(enxame_fit_novo[linha],n_tap,n_vgen,n_bshunt,sep)
            enxame_fit_novo[linha][n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(enxame_fit_novo[linha],n_tap,n_vgen,n_bshunt,sep)
            

        enxame_estat = enxame_fit_novo[:,-6:]

        enxame_fit = np.concatenate(( np.clip(enxame_fit_novo[:,0:-6], a_min = lim_inf, a_max = lim_sup, out = enxame_fit_novo[:,0:-6]),enxame_estat),axis=1)   

        enxame_fit[np.argsort(best_particles[:, -1])[0],:] = global_best.copy()
        
        end = time.time()

        elapsed = end - start

        j.append(global_best[-1])

        perdas.append(global_best[-6])

        pen_v.append(global_best[-5])

        pen_gq.append(global_best[-4])

        pen_tap.append(global_best[-3])

        pen_bsh.append(global_best[-2])
        
        tempo.append(elapsed)
      

        if relatorio == True:
            
            print(' ')

            print('Melhor Global da Iteração:',i)

            print('Perdas (pu):', global_best[-6])

            print('Penalização de Tensão:', global_best[-5])

            print('Penalização de Geração de Reativo:', global_best[-4])

            print('Penalização do Tap:', global_best[-3])

            print('Penalização do Bshunt:', global_best[-2])

            print('Fitness:', global_best[-1])
            
            print('Tempo: ', elapsed)

            print(' ')

            print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
            
            
    
    if relatorio == True:
        
            plt.figure(figsize=(18,10))
            plt.plot(perdas)
            
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Perdas de Potência Ativa (pu)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(j)
            
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Fitness (J)')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_v)
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização de Tensão')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_gq)
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização de Geração de Reativo')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização do TAP')
            plt.xlabel('Número da Iteração')
            
            plt.figure(figsize=(18,10))
            plt.plot(pen_bsh)
            plt.title('Otimização Por Enxame de Partículas')
            plt.ylabel('Penalização do BShunt')
            plt.xlabel('Número da Iteração')
                       
            
    return j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,global_best, tempo


