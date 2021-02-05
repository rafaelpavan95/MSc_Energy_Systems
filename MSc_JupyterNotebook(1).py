#!/usr/bin/env python
# coding: utf-8

# # <center> <img src="figs/logounesp" alt="Logo UNESP" width="110" align="left"/> <font size="20"> <br><center>Mestrado Acadêmico<br/><br><font size="4"> <br><center>Programa de Pós-Graduação em Engenharia Elétrica<br/><br><center>Faculdade de Engenharia de Bauru (FEB)<br/><br><center>Faculdade de Ciências (FC)<br/>

# # <center>Método Híbrido Para Solução do Problema de Fluxo de Potência Ótimo Reativo Com Variáveis Discretas<br/>

# #### <center><font size="4">Aluno:</center></font><br><center>Rafael Pavan</center></br>

# #### <center><font size="4">Orientador:</center></font> <br><center>Profª Drª Edilaine Martins Soler</center></br>

# # <center>Implementação Computacional</center>
# 
# 
# 
# 

# #### 1. Visualização de Dados do Sistema Elétrico de Potência
# 
#     1.1 IEEE 14 Barras;
#     1.2 IEEE 30 Barras;
#     1.3 IEEE 118 Barras;
#     1.4 IEEE 300 Barras;
#     
# #### 2. Funções Para Manipulação e Extração de Dados do Sistema Elétrico de Potência
# 
#     2.1 Função Para Inicalizar o Sistema Elétrico;
#     2.2 Função Para Calcular As Condutâncias de Linha do Sistema;
#     2.3 Função Para Coletar Tensões Das Barras e Seus Respectivos Limites Inferiores e Superiores;
#     2.4 Função Para Coletar Tensões e Potências Das Barras de Geração/Controle de Reativo e Seus Respectivos               Limites Inferiores e Superiores;
#     2.5 Função Objetivo Para Calcular as Perdas de Potência Ativa nas Linhas de Transmissão;
#     2.6 Função Para Calcular a Penalização por Ultrapassagem de Tensão;
#     2.7 Função Para Calcular a Penalização por Ultrapassagem de Geração de Reativo dos Geradores;
#     2.8 Função Para Coletar Dados de Carregamento e TAP dos Trafos
#     2.9 Função Para Coletar Dados dos Reativos Shunt;
#     2.10 Função Para Calcular a Penalização do Bshunt;
#     2.11 Função Para Converter TAPs;
#     2.12 Função Para Criar Alcateia;
#     2.13 Função Para Criar Enxame;
#     2.14 Função Para Calcular o Fluxo de Potência;
#     2.15 Função Para Calcular o Fitness;
#     2.16 Função Para Validação (Salva Dados no SEP);
#     
# #### 3. Implementação de Métodos de Otimização Meta-Heurísticos
#     
#     3.1 Método Alcateia de Lobos Cinzentos (Gray Wolf Optimizer);
#     3.2 Método Enxame de Partículas (Particle Swarm Optimization);    
#     3.3 Em Andamento: Algoritmo Genético, Algoritmo GWO/PSO Híbrido
#         .
#         .
#         .
#         .
#         .
#         
#     
# #### 4. Protocolo Experimental Com Métodos Meta-Heurísticos
# 
#     4.1 Em Andamento
#         .
#         .
#         .
#         .
#         .
#         .
#        

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandapower as pp
import time
import random
from pandapower.networks import case14, case_ieee30, case118, case300, case4gs
import pandapower.plotting as pplot
import tabulate
import numba
from numba import njit

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"
font = {'size'   : 13}
plt.rc('font', **font)
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 


# ##  1. Visualização dos Dados dos Sistemas Elétricos de Potência
# 

# In[2]:


sep14 = case14()
sep30 = case_ieee30()
sep118 = case118()
sep300 = case300()
sep4 = case4gs()


# ## 1.1 Sistema de 14 Barras

# In[3]:


pplot.simple_plot(sep14,plot_loads=True)


# In[4]:


pp.runpp(sep14, algorithm = 'nr')
sep14


# In[5]:


sep14.res_bus


# In[6]:


sep14.res_bus


# In[7]:


sep14.line


# In[8]:


sep14.ext_grid


# In[9]:


sep14.res_ext_grid


# In[10]:


sep14.gen


# In[11]:


sep14.res_gen


# In[12]:


sep14.res_line


# In[13]:


sep14.line


# In[14]:


sep14.shunt


# In[15]:


sep14.trafo.columns


# In[16]:


sep14.trafo


# ## 1.2 Sistema de 30 Barras

# In[17]:


pplot.simple_plot(sep30,plot_loads=True)


# In[18]:


pp.runpp(sep30, algorithm = 'nr')
sep30


# In[19]:


sep30.bus


# In[20]:


sep30.res_bus


# In[21]:


sep30.res_gen


# In[22]:


sep30.gen


# In[23]:


sep30.line


# In[24]:


sep30.gen


# In[25]:


sep30.shunt


# In[26]:


sep30.trafo


# In[27]:


sep30.res_trafo


# ## 1.3 Sistema de 118 Barras

# In[28]:


pplot.simple_plot(sep118,plot_loads=True)


# In[29]:


pp.runpp(sep118, algorithm = 'nr')

sep118


# In[30]:


sep118.bus


# In[31]:


sep118.res_bus


# In[32]:


sep118.gen


# In[33]:


sep118.line


# In[34]:


sep118.shunt


# In[35]:


sep118.trafo


# ## 1.4 Sistema de 300 Barras

# In[36]:


pplot.simple_plot(sep300,plot_loads=True)


# In[37]:


pp.runpp(sep300, algorithm = 'nr')
sep300


# In[38]:


sep300.bus


# In[39]:


sep300.res_bus


# In[40]:


sep300.res_trafo


# In[41]:


sep300.res_gen


# In[42]:


sep300.gen


# In[43]:


sep300.line


# In[44]:


sep300.res_line


# In[45]:


sep300.shunt


# In[46]:


sep300.trafo


# ## 2. Funções Para Manipulação e Extração de Dados do Sistema Elétrico de Potência

# ## 2.1 Função Para Inicializar o Sistema Elétrico de Potência

# Para se coletar alguns dados iniciais do sistema elétrico é necessário realizar um cálculo de fluxo de carga preliminar neste.

# In[47]:


def inicializa_sep(sep, algorithm, relatorio):
    
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
    
    sep14 = case14()
    sep30 = case_ieee30()
    sep118 = case118()
    sep300 = case300()

    if len(sep.bus)==14:
        
        sep.bus['min_vm_pu'] = 0.95
        sep.bus['max_vm_pu'] = 1.05
        sep.ext_grid['vm_pu'] = 1.05
        
      
        
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

        
    if len(sep.bus)==30:
        
        sep.bus['min_vm_pu'] = 0.94
        sep.bus['max_vm_pu'] = 1.06
        sep.ext_grid['vm_pu'] = 1.06
        sep.gen['max_q_mvar']=np.array([50,40,40,24,24])
        sep.gen['min_q_mvar']=np.array([-40,-40,-10,-6,-6])
        sep.ext_grid['max_q_mvar'] = 10
        sep.ext_grid['min_q_mvar'] = 0
        
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
    


# ### Teste

# In[48]:


inicial_30 = inicializa_sep(sep30,algorithm='nr', relatorio=False)
inicial_14 = inicializa_sep(sep14,algorithm='nr', relatorio=False)
inicial_118 = inicializa_sep(sep118,algorithm='nr', relatorio=False)
inicial_300 = inicializa_sep(sep300,algorithm='fdbx', relatorio=False)

inicial_30 = inicializa_sep(sep30,algorithm='nr', relatorio=True)


# ## 2.2 Função Para Calcular as Condutâncias de Linha do Sistema

# A matriz de condutâncias do sistema é importante para, posteriormente, calcular as perdas nas linhas.

# In[49]:


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


# ### Teste

# In[50]:


matriz_condutancia(sep300,relatorio=True)


# ## 2.3 Função Para Coletar Tensões Das Barras e Seus Respectivos Limites Inferiores e Superiores
# 
# As tensões nas barras são importantes para se calcular as perdas nas linhas, e os limites inferiores e superiores para garantir que as estas estejam nos níveis adequados.

# In[51]:


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
        plt.grid()
        plt.title('Módulo da Tensão por Barra do Sistema')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Tensão [pu]')
        plt.legend(loc='best')
        plt.figure(figsize=(16,10))
        sns.scatterplot(x=np.arange(0,len(theta),1),y=theta,color='green',label='Ângulo da Tensão',s=75)
        plt.grid()
        plt.title('Ângulo da Tensão por Barra do Sistema')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Theta [°]')
        plt.legend(loc='best')
        
    
    if relatorio==False:
        
        return vbus, theta, v_lim_superior, v_lim_inferior
    
    


# ### Teste

# In[52]:


coleta_dados_vbus(sep14,relatorio=True)


# ## 2.4 Função Para Coletar Tensões e Potências Das Barras de Geração/Controle de Reativo e Seus Respectivos Limites Inferiores e Superiores

# In[53]:


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
        plt.grid()
        plt.title('Potência Reativa Gerada')
        plt.xlabel('Barra do Sistema')
        plt.ylabel('Potência Reativa (pu)')
        plt.legend(loc='best')
        
    
    if relatorio==False:
        
        return vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra
    
    


# ### Teste

# In[54]:


coleta_dados_gen(sep14,relatorio=True)


# ## 2.5 Função Objetivo Para Calcular as Perdas de Potência Ativa nas Linhas de Transmissão

# In[55]:


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
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        
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


# ### Teste

# In[56]:


tensao,angulo, vlims ,vlimi = coleta_dados_vbus(sep300,relatorio=False)
matrizg,matrizz = matriz_condutancia(sep300,relatorio=False)


func_objetivo(tensao,angulo,matrizg,matrizz,relatorio=True)


# ## 2.6 Função Para Calcular a Penalização por Ultrapassagem de Tensão

# In[57]:


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


# ### Teste

# In[58]:


pen_tensao(tensao,vlims,vlimi,relatorio=True)


# ## 2.7 Função Para Calcular a Penalização por Ultrapassagem de Geração de Reativo

# In[59]:


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


# ### Teste

# In[60]:


vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra = coleta_dados_gen(sep14,relatorio=False)


# In[61]:


pen_ger_reativo(qgen, q_lim_superior, q_lim_inferior,sep14,relatorio=True)

sep14.res_ext_grid


# In[62]:


print('Resultado Esperado:\n')
print(0.0)


# ## 2.8 Função Para Coletar Dados de Carregamento e TAP dos Trafos

# In[63]:


def coleta_dados_trafo(sep, relatorio=True):
    
    '''    
    
    
    Valores dos TAPs Retirados de:
    
    - REFORMULAÇÃO DAS RESTRIÇÕESDE COMPLEMENTARIDADE EM PROBLEMAS DE FLUXO DE POTÊNCIA ÓTIMO
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
        
        step = 0.00625
        valores_taps = np.arange(start = 0.9, stop = 1.1, step = step)
        
        
    if len(sep.bus)==30:
        
        step = 0.00625
        valores_taps = np.arange(start = 0.9, stop = 1.1, step = step)
        
                
    if len(sep.bus)==118:
        
        step = 0.00625
        valores_taps = np.arange(start = 0.9, stop = 1.1, step = step)

                    
    if len(sep.bus)==300:
        
        step = 0.00625
        valores_taps = np.arange(start = 0.9, stop = 1.1, step = step)
        
        
    if relatorio == True:
        
    
        tap_pos = sep.trafo[~pd.isnull(sep.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)

        tap_neutral = sep.trafo[~pd.isnull(sep.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)

        tap_step_percent = sep.trafo[~pd.isnull(sep.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)
        
        valor_percentual= (tap_pos-tap_neutral)*(tap_step_percent/100) + 1
        
        plt.figure(figsize=(20,10))
        sns.scatterplot(x=np.arange(0,len(sep.trafo),1),y=carregamento,label='Carregamento',color='b',s=75)
        sns.lineplot(x=np.arange(0,len(sep.trafo),1),y=np.ones((len(sep.trafo))),label='Limite Máximo de Carregamento',color='r')
        plt.grid()
        
        plt.xlabel('Nº do Trafo')
        plt.ylabel('Carregamento %')
        plt.title('Carregamento dos Transformadores')
        
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
    


# ### Teste

# In[64]:


coleta_dados_trafo(sep300,relatorio=True)


# In[65]:


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
    
    step = 0.00625

    linha[-3] = np.sum(np.square(np.sin((linha[n_vgen:n_vgen+n_tap]*np.pi/step))))
    
    return linha


# ### Teste

# In[66]:


pen_trafo(np.array([0,0,0,0,0.95,0.97,1.003,0,0,0,0,0,0,0]), n_tap=3, n_vgen=4)


# ## 2.9 Função Para Coletar Dados dos Reativos Shunt

# In[67]:


def coleta_dados_bshunt(sep):
    
    '''    
    
    
    Valores dos Shunt Retirados de:
    
    - REFORMULAÇÃO DAS RESTRIÇÕESDE COMPLEMENTARIDADE EM PROBLEMAS DE FLUXO DE POTÊNCIA ÓTIMO
      Marina Valença Alencar - Dissertação de Mestrado

    - FUNÇÕES PENALIDADE PARA O TRATAMENTO DAS VARIÁVEIS DISCRETAS DO PROBLEMA DE FLUXO DE POTÊNCIA ÓTIMO REATIVO
      Daisy Paes Silva - Dissertação de Mestrado
          

    ''' 
    
    ieee14 = np.arange(0.00,0.40,0.01)
    ieee30 = np.arange(0.00,0.40,0.01)
    ieee118 = np.arange(0.00,0.40,0.01)
    
    bus = sep.shunt['bus'].sort_values().to_numpy()
    sep.shunt=sep.shunt.sort_index()
  
    
    if len(sep.bus)==14:
        
        #bsh = np.array([[0,0.05,0.15,0.19,0.20,0.24,0.34,0.39]])
        bsh = np.array([ieee14])
        
        
    if len(sep.bus)==30:
        
        #bsh = np.array([[0,0.05,0.15,0.19,0.20,0.24,0.34,0.39],[0, 0.04, 0.05, 0.09]])
        bsh = np.array([ieee30,ieee30])
        
                
    if len(sep.bus)==118:
        
       # bsh = np.array([[-0.40, -0.20, 0]
        #                ,
         #               [0, 0.06, 0.07, 0.13, 0.14, 0.20],
          #              [-0.25, -0.20, 0],
           #             [0, 0.10],
            #            [0, 0.10],
             #           [0, 0.10],
              #          [0, 0.15],
               #         [0.08, 0.12, 0.20],
                #        [0.10, 0.20],
                 #       [0.10, 0.20],
                  #      [0.10, 0.20],
                   #     [0.10, 0.20],
                    #    [0, 0.06, 0.07, 0.13, 0.14, 0.20],
                     #   [0, 0.06, 0.07, 0.13, 0.14, 0.20]])
                        
                                
        bsh = np.array([ieee118
                        ,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118,
                        ieee118])
                            
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
               ])
    
    
    
    return bsh, bus


# In[68]:


bsh,bus=coleta_dados_bshunt(sep118)


# In[69]:


bsh


# ## 2.10 Função para Calcular a Penalização Bshunt

# In[70]:


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
    


# ### Teste

# In[71]:


# Valor presente no conjunto de discretos
pen_bshunt(np.array([0,0,0,0,0,0,0,-5,0,0,0,0,0,0]),3,4,1,sep14)


# In[72]:


# Valor não presente no conjunto de discretos

pen_bshunt(np.array([0,0,0,0,0,0,0,0.12,0,0,0,0,0,0]),3,4,1,sep14)


# ## 2.11 Função Para Converter TAPS

# In[73]:


def converte_trafo(tap_pos, tap_neutral, tap_step_percent,valores_taps):
    
    '''
    Converte TAPS conforme equação disponibilizada pelo pandapower.
    
    https://pandapower.readthedocs.io/en/v2.1.0/elements/trafo.html
    
    '''
    
    taps_convertido = tap_neutral + ((valores_taps - 1.0)*(100/tap_step_percent))
    
    
    return taps_convertido


# ### Teste

# In[74]:


tap_pos, tap_neutral, tap_step_percent,valores_taps = coleta_dados_trafo(sep14,relatorio=False)

converte_trafo(np.array([0.98,0.95,1.1]), tap_neutral, tap_step_percent,valores_taps[0])


# ## 2.12 Função Para Criar Alcateia

# In[75]:


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
    
    
    


# ### Teste

# In[76]:


cria_alcateia(sep300,1)


# ## 2.13 Função Para Criar o Enxame de Partículas

# In[77]:


def cria_enxame(sep,n_particulas):
    
    """"
    
    Cria o enxame de partículas.
    
    
    linhas = partículas
    
    colunas = tensões geradores, tap transformadores, susceptâncias shunt, perdas, penalização de tensão, penalização de reativo, penalização de trafo, penalização shunt, fitness
    
    """
    
    
    vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra = coleta_dados_gen(sep,relatorio=False)
    
    n_vgen=len(vgen)
    
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


# ### Teste

# In[78]:


cria_enxame(sep14,5)


# ## 2.14 Função para Calcular o Fluxo de Potência

# In[79]:


def fluxo_de_pot(grupo, sep):
    
    n_bshunt = len(sep.shunt)
    n_vgen = len(sep.gen)
    n_tap = np.abs(sep.trafo['tap_pos']).count()
    
    matrizg = matriz_condutancia(sep,relatorio=False)
    
    for linha in range(grupo.shape[0]):
        
        sep.gen['vm_pu']=grupo[linha,0:n_vgen]
        
        tap_pos, tap_neutral, tap_step_percent,valores_taps=coleta_dados_trafo(sep,relatorio=False)
        
        sep.trafo['tap_pos'][~pd.isnull(sep.trafo['tap_pos'])]=converte_trafo(tap_pos, tap_neutral, tap_step_percent,grupo[linha,n_vgen:n_vgen+n_tap])
        
        sep.shunt['q_mvar']=grupo[linha,n_vgen+n_tap:n_vgen+n_tap+n_bshunt]*-100
        
        if len(sep.bus)==300:
        
            pp.runpp(sep,algorithm='fdbx',numba=True, init = 'results', tolerance_mva = 1e-4,max_iteration=100000)
        
        else:
        
            pp.runpp(sep,algorithm='nr',numba=True, init = 'results', tolerance_mva = 1e-5,max_iteration=10000)
        
        vbus, theta, v_lim_superior, v_lim_inferior=coleta_dados_vbus(sep,relatorio=False)
        
        grupo[linha,-6] = sep.res_line['pl_mw'].sum()/100 #func_objetivo(vbus,theta,matrizg,relatorio=False)

        grupo[linha,-5] = pen_tensao(vbus, v_lim_superior, v_lim_inferior,relatorio=False)
        
        vgen, thetagen, pgen, qgen, p_lim_superior, p_lim_inferior, q_lim_superior, q_lim_inferior,barra = coleta_dados_gen(sep,relatorio=False)
        
        grupo[linha,-4] = pen_ger_reativo(qgen, q_lim_superior, q_lim_inferior,sep,relatorio=False)
        
        grupo[linha,:] = pen_trafo(grupo[linha,:],n_tap,n_vgen)
        
        
        grupo[linha,-2] = pen_bshunt(grupo[linha,:],n_tap,n_vgen,n_bshunt,sep)
  
        
        
    
    return grupo
    


# ### Teste

# In[80]:


#for i in range(100):
enxame=cria_enxame(sep300,1)
fluxo_de_pot(enxame,sep300)


# ## 2.15 Função para Calcular o Fitness (J)

# In[81]:


def fitness (grupo,zeta,psi,sigma,omega):
    
# fitness J       perdas         pen tensão         pen q mvar          pen trafo           pen bshunt       
    grupo[:,-1]=(grupo[:,-6])+(zeta*grupo[:,-5])+(psi*grupo[:,-4])+(sigma*grupo[:,-3])+(omega*grupo[:,-2])

    return grupo


# ### Teste

# In[82]:


fitness(enxame,100,100,100,100)


# ## 2.16 Função para Validação

# In[83]:


def validacao (sep, best_solution):
       
    valida = fluxo_de_pot(np.array([best_solution]), sep)
           
    print('Sistema Simulado Para a Solução:\n')
    print(valida[0][:-6])
    print(' ')
    
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
    
    print('Fitness:\n')
    print(fitness(np.array([best_solution]),1000,1000,10,10)[0][-1])
    print(' ')
    


# ## 3. Implementação de Métodos de Otimização Meta-Heurísticos

# ## 3.1 Método Alcateia de Lobos Cinzentos 

# In[157]:


def otimizacao_alcateia_de_lobos_cinzentos(sep, zeta, psi, sigma, omega, max_iter, n_lobos,valor_inicial,relatorio=True,inicial=True):
        
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

    


# ## 3.2 Método Enxame de Partículas

# In[158]:


def otimizacao_enxame_de_particulas(sep, zeta, psi, sigma, omega, max_iter, n_particles,c1,c2,v_amp,valor_inicial,relatorio=True,inicial=True):
        
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

      


# ### Teste

# In[91]:


penalizacao_v = 10000
penalizacao_q = 10000
penalizacao_tap = 10000
penalizacao_bshunt = 10000

max_iter = 30

n_wolfs = 15

sistema = sep30

if len(sistema.bus) == 14:

    v_inicial = inicial_14

if len(sistema.bus) == 30:

    v_inicial = inicial_30
    

if len(sistema.bus) == 118:

    v_inicial = inicial_118


if len(sistema.bus) == 300:

    v_inicial = inicial_300


   
j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,alcateias,lobo_alfa, lobo_beta, lobo_delta, elapsed = otimizacao_alcateia_de_lobos_cinzentos(sistema, penalizacao_v, penalizacao_q, penalizacao_tap, penalizacao_bshunt, max_iter, n_wolfs, valor_inicial=v_inicial,relatorio=True,inicial=True)


# In[92]:


glob_30 = np.array([1.0251,0.9749,0.965,0.969,1.06,1.12,0.88,0.88,0.88,0.39,0.09,0,0,0,0,0,0])
glob_14 = np.array([1.0376,1.00956,1.05,1.02,1.12,1.12,0.88,0.39,0,0,0,0,0,0])


penalizacao_v = 10000
penalizacao_q = 10000
penalizacao_tap = 10000
penalizacao_bshunt = 10000

max_iter = 30

n_particles = 15

sistema = sep30

if len(sistema.bus) == 14:

    v_inicial = inicial_14

if len(sistema.bus) == 30:

    v_inicial = inicial_30
    

if len(sistema.bus) == 118:

    v_inicial = inicial_118


if len(sistema.bus) == 300:

    v_inicial = inicial_300

j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,global_best, tempo = otimizacao_enxame_de_particulas(sistema, penalizacao_v, penalizacao_q, penalizacao_tap, penalizacao_bshunt, max_iter, n_particles, 2, 2, 0.1, valor_inicial = v_inicial,relatorio=True,inicial=True)

print(global_best)


# In[165]:


print(lobo_alfa)


# In[ ]:


jota = np.zeros([50,50])
per = np.zeros([50,50])
penv = np.zeros([50,50])
peng = np.zeros([50,50])
penbsh = np.zeros([50,50])
tim = np.zeros([50,50])
pentap = np.zeros([50,50])
global_b = []

for i in range(50):
    
    penalizacao_v = 1000
    penalizacao_q = 1000
    penalizacao_tap = 0
    penalizacao_bshunt = 0

    max_iter = 50

    n_particles = 30

    sistema = sep118

    if len(sistema.bus) == 14:

        v_inicial = inicial_14

    if len(sistema.bus) == 30:

        v_inicial = inicial_30


    if len(sistema.bus) == 118:

        v_inicial = inicial_118


    if len(sistema.bus) == 300:

        v_inicial = inicial_300

    j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,global_best, tempo = otimizacao_enxame_de_particulas2(sistema, penalizacao_v, penalizacao_q, penalizacao_tap, penalizacao_bshunt, max_iter, n_particles, 2, 2, 0.1, valor_inicial= v_inicial,relatorio=False,inicial=True)


    jota[i,:] = j
    per[i,:] = perdas
    penv[i,:] = pen_v
    peng[i,:] = pen_gq
    penbsh[i,:] = pen_bsh
    tim[i,:] = tempo
    pentap[i,:] = pen_tap
    global_b.append(global_best)
    
    print('Iteração: ')
    print(i)



# In[ ]:


jota = np.zeros([50,50])
per = np.zeros([50,50])
penv = np.zeros([50,50])
peng = np.zeros([50,50])
penbsh = np.zeros([50,50])
tim = np.zeros([50,50])
pentap = np.zeros([50,50])
global_b = []

for i in range(50):
    
    penalizacao_v = 1000
    penalizacao_q = 1000
    penalizacao_tap = 0
    penalizacao_bshunt = 0

    max_iter = 109

    n_particles = 50
    
    sistema = sep118

    if len(sistema.bus) == 14:

        v_inicial = inicial_14

    if len(sistema.bus) == 30:

        v_inicial = inicial_30


    if len(sistema.bus) == 118:

        v_inicial = inicial_118


    if len(sistema.bus) == 300:

        v_inicial = inicial_300

    j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,alcateias,lobo_alfa, lobo_beta, lobo_delta, elapsed = otimizacao_alcateia_de_lobos_cinzentos2(sistema, penalizacao_v, penalizacao_q, penalizacao_tap, penalizacao_bshunt, max_iter, n_wolfs, valor_inicial=v_inicial,relatorio=False,inicial=True)


    jota[i,:] = j
    per[i,:] = perdas
    penv[i,:] = pen_v
    peng[i,:] = pen_gq
    penbsh[i,:] = pen_bsh
    tim[i,:] = tempo
    pentap[i,:] = pen_tap
    global_b.append(lobo_alfa)
    
    print('Iteração: ')
    print(i)





# In[ ]:


jota300 = np.zeros([50,100])
per300 = np.zeros([50,100])
penv300 = np.zeros([50,100])
peng300 = np.zeros([50,100])
penbsh300 = np.zeros([50,100])
tim300 = np.zeros([50,100])
pentap300 = np.zeros([50,100])
global_b300 = []

for i in range(50):
    
    j, perdas, pen_v, pen_gq, pen_tap, pen_bsh, global_best, tempo = otimizacao_enxame_de_particulas(sep300, 1000, 1000, 10, 10, 100, 100, 2, 2,0.1, valor_inicial=glob_30,relatorio=False,inicial=False)

    jota300[i,:] = j
    per300[i,:] = perdas
    penv300[i,:] = pen_v
    peng300[i,:] = pen_gq
    penbsh300[i,:] = pen_bsh
    tim300[i,:] = tempo
    pentap300[i,:] = pen_tap
    global_b300.append(global_best)
    
    print('Iteração: ')
    print(i)
    


# In[ ]:


plt.plot(np.median(per,axis=0),label='Mediana',color='blue')
plt.plot(np.mean(per,axis=0),label='Média',color='red')
plt.fill_between(np.arange(30), np.mean(per,axis=0) + np.std(per,axis=0), np.mean(per,axis=0) - np.std(per,axis=0),alpha=0.1)
plt.fill_between(np.arange(30), np.median(per,axis=0) + np.std(per,axis=0), np.median(per,axis=0) - np.std(per,axis=0),alpha=0.1)

plt.legend()
plt.grid()

    
plt.title('IEEE 30: PSO (50 EXECUÇÕES / 30 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Média e Mediana das Perdas')


# In[ ]:


plt.style.use('seaborn-white')

for i in range(len(jota)):
    plt.plot(jota[i,:])
    
plt.title('IEEE 30: PSO (50 EXECUÇÕES / 30 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Fitness')

plt.grid()
plt.show()


# In[ ]:


plt.plot(np.median(jota,axis=0),label='Mediana',color='blue')
plt.plot(np.mean(jota,axis=0),label='Média',color='red')
plt.fill_between(np.arange(30), np.mean(jota,axis=0) + np.std(jota,axis=0), np.mean(jota,axis=0) - np.std(jota,axis=0),alpha=0.1)
plt.fill_between(np.arange(30), np.median(jota,axis=0) + np.std(jota,axis=0), np.median(jota,axis=0) - np.std(jota,axis=0),alpha=0.1)

plt.legend()
plt.grid()

    
plt.title('IEEE 30: GWO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Média e Mediana do Fitness')


# In[ ]:


plt.style.use('seaborn-white')

for i in range(len(per)):
    plt.plot(per[i,:])
    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Perdas')

plt.grid()
plt.show()


# In[ ]:


plt.plot(np.median(per,axis=0),label='Mediana',color='blue')
plt.plot(np.mean(per,axis=0),label='Média',color='red')
plt.fill_between(np.arange(100), np.mean(per,axis=0) + np.std(per,axis=0), np.mean(per,axis=0) - np.std(per,axis=0),alpha=0.1)
plt.fill_between(np.arange(100), np.median(per,axis=0) + np.std(per,axis=0), np.median(per,axis=0) - np.std(per,axis=0),alpha=0.1)

plt.legend()
plt.grid()

    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Média e Mediana das Perdas')


# In[ ]:


plt.style.use('seaborn-white')

for i in range(len(penv)):
    plt.plot(penv[i,:])
    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Penalização de Tensão')

plt.grid()
plt.show()


# In[ ]:


plt.plot(np.median(penv,axis=0),label='Mediana',color='blue')
plt.plot(np.mean(penv,axis=0),label='Média',color='red')
plt.fill_between(np.arange(100), np.mean(penv,axis=0) + np.std(penv,axis=0), np.mean(penv,axis=0) - np.std(penv,axis=0),alpha=0.1)
plt.fill_between(np.arange(100), np.median(penv,axis=0) + np.std(penv,axis=0), np.median(penv,axis=0) - np.std(penv,axis=0),alpha=0.1)

plt.legend()
plt.grid()

    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Média e Mediana da Penalização de Tensão')


# In[ ]:


plt.plot(np.median(peng,axis=0),label='Mediana',color='blue')
plt.plot(np.mean(peng,axis=0),label='Média',color='red')
plt.fill_between(np.arange(100), np.mean(peng,axis=0) + np.std(peng,axis=0), np.mean(peng,axis=0) - np.std(peng,axis=0),alpha=0.1)
plt.fill_between(np.arange(100), np.median(peng,axis=0) + np.std(peng,axis=0), np.median(peng,axis=0) - np.std(peng,axis=0),alpha=0.1)

plt.legend()
plt.grid()

    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Média e Mediana da Penalização de Geração')


# In[ ]:


plt.style.use('seaborn-white')

for i in range(len(peng)):
    plt.plot(peng[i,:])
    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Penalização de Geração')

plt.grid()
plt.show()


# In[ ]:


plt.style.use('seaborn-white')

for i in range(len(penbsh)):
    plt.plot(penbsh[i,:])
    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Penalização Bshunt')

plt.grid()
plt.show()


# In[ ]:


plt.plot(np.median(penbsh,axis=0),label='Mediana',color='blue')
plt.plot(np.mean(penbsh,axis=0),label='Média',color='red')
plt.fill_between(np.arange(100), np.mean(penbsh,axis=0) + np.std(penbsh,axis=0), np.mean(penbsh,axis=0) - np.std(penbsh,axis=0),alpha=0.1)
plt.fill_between(np.arange(100), np.median(penbsh,axis=0) + np.std(penbsh,axis=0), np.median(penbsh,axis=0) - np.std(penbsh,axis=0),alpha=0.1)

plt.legend()
plt.grid()

    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Média e Mediana da Penalização Bshunt')


# In[ ]:


plt.style.use('seaborn-white')

for i in range(len(pentap)):
    plt.plot(pentap[i,:])
    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Penalização do TAP')

plt.grid()
plt.show()


# In[ ]:


plt.plot(np.median(pentap,axis=0),label='Mediana',color='blue')
plt.plot(np.mean(pentap,axis=0),label='Média',color='red')
plt.fill_between(np.arange(100), np.mean(pentap,axis=0) + np.std(pentap,axis=0), np.mean(pentap,axis=0) - np.std(pentap,axis=0),alpha=0.1)
plt.fill_between(np.arange(100), np.median(pentap,axis=0) + np.std(pentap,axis=0), np.median(pentap,axis=0) - np.std(pentap,axis=0),alpha=0.1)

plt.legend()
plt.grid()

    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES)')
plt.xlabel('Iterações')
plt.ylabel('Média e Mediana da Penalização de Geração')


# In[ ]:


pena = jota[:,-1] - per[:,-1]

plt.hist(pena)
plt.grid()

    
plt.title('IEEE 118: PSO (50 EXECUÇÕES / 100 ITERAÇÕES) Histograma de Frequência')
plt.xlabel('Somatório das Penalizações')
plt.ylabel('Frequências')


# In[ ]:


len(pena[pena<0.001])


# In[ ]:


np.argmin(jota[:,-1])


# In[ ]:


init


# In[ ]:


validacao(sep118,lobo_alfa)


# In[ ]:


coleta_dados_vbus(sep118,relatorio=True) #
#[ 0.98798055  1.04575647  1.00950339  1.04036462  1.02704805  1.00893791
#  1.00858986  1.0047482   1.00063139  1.0582289   1.03505602  1.03624014
#  1.03453512  1.04453844  1.03849237  1.02522917  1.02045642  1.05513686
#  1.05194047  1.05526188  1.03000704  1.03018907  1.02471901  1.02548059
#  1.02621844  1.00970307  1.01223421  1.04322709  1.04918653  1.02507344
#  0.95218199  1.02663763  1.00753379  0.99740693  1.04214262  1.06
#  1.04121107  1.02544117  1.0570769   0.9448963   1.03090518  1.03910908
#  1.03006572  1.04678375  1.04564313  1.03527807  1.0351827   1.03107075
#  1.03075305  1.04996945  0.9994698   1.04178034  1.02796504  1.025
#  1.00625     1.00625     0.99375     1.          1.          1.01875
#  0.96875     1.1         0.          0.07       -0.2         0.1
#  0.1         0.1         0.15        0.12        0.1         0.1
#  0.2         0.2         0.07        0.2  ]



#Sistema Simulado Para a Solução:

#[ 0.97110395  0.99301498  0.99341324  0.98897927  1.01125619  0.99420506
#  0.9786915   0.98293981  0.97761344  0.98380786  1.00963681  0.98948012
#  1.01182371  0.97741193  0.99793626  0.98490286  0.97902733  0.98614388
#  0.99949246  0.98722291  1.01493092  1.00423279  0.99958708  1.00036178
#  1.00328008  1.01783387  1.01326557  1.01327623  1.0187207   1.00724629
#  0.9942187   0.99986967  0.98480014  0.96191013  0.99234626  0.99629653
#  0.9802574   0.99601756  0.99116616  0.97520822  0.9943048   0.98149934
#  0.97362485  0.9933737   0.98631997  0.97837746  0.98311451  0.9879563
#  1.01136539  1.02460243  1.01716511  0.99922554  0.9750293   0.95
#  0.975       0.9625      1.03125     0.96875     0.9875      0.99375
#  0.99375     0.9625      0.          0.07       -0.2         0.1
#  0.          0.          0.          0.12        0.1         0.2
#  0.2         0.2         0.13        0.14      ]


# In[ ]:


coleta_dados_gen(sep118,relatorio=True)


# In[ ]:


coleta_dados_trafo(sep118,relatorio=True)


# In[ ]:


temp = 0

tempo_acumulado = np.zeros(len(tempo))

for i in range(len(tempo)):
    
    if i == 0:
        
        tempo_acumulado[i] = tempo[i]
    
    else:
        
    
        tempo_acumulado[i] = tempo_acumulado[i-1] + tempo[i]
    


# In[ ]:


plt.plot(tempo_acumulado)
plt.grid()


# In[ ]:


np.mean(tempo)


# In[ ]:


np.std(tempo)


# In[95]:


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
    


# In[96]:


def discreto_tap(grupo,n_tap,n_vgen,n_bshunt,sep):
    
    b = grupo[n_vgen:n_vgen+n_tap]
    
    ref = np.arange(0.9,1.1,0.00625)
    
    discretizatap = np.zeros(len(b))
    
    i = 0

    
    for i in range(len(b)):
        
        discretizatap[i]=(ref[np.argmin(np.abs(ref-b[i]))])
                   
    return discretizatap
    
    


# In[135]:


def otimizacao_enxame_de_particulas2(sep, zeta, psi, sigma, omega, max_iter, n_particles,c1,c2,v_amp,valor_inicial,relatorio=True,inicial=True):
        
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
    
    v_anterior = v_amp*cria_enxame(sep,n_particles)

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
        
        
        for linha in range(n_particles):
          
            enxame_fit_novo[linha][n_vgen:n_vgen+n_tap] = discreto_tap(enxame_fit_novo[linha],n_tap,n_vgen,n_bshunt,sep)
            enxame_fit_novo[linha][n_vgen+n_tap:n_vgen+n_tap+n_bshunt] = discreto_bshunt(enxame_fit_novo[linha],n_tap,n_vgen,n_bshunt,sep)
            

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

      


# In[145]:


glob_30 = np.array([1.0251,0.9749,0.965,0.969,1.06,1.12,0.88,0.88,0.88,0.39,0.09,0,0,0,0,0,0])
glob_14 = np.array([1.0376,1.00956,1.05,1.02,1.12,1.12,0.88,0.39,0,0,0,0,0,0])


penalizacao_v = 10000
penalizacao_q = 10000
penalizacao_tap = 0
penalizacao_bshunt = 0

max_iter = 30

n_particles = 15

sistema = sep30

if len(sistema.bus) == 14:

    v_inicial = inicial_14

if len(sistema.bus) == 30:

    v_inicial = inicial_30
    

if len(sistema.bus) == 118:

    v_inicial = inicial_118

if len(sistema.bus) == 300:

    v_inicial = inicial_300

j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,global_best, tempo = otimizacao_enxame_de_particulas2(sistema, penalizacao_v, penalizacao_q, penalizacao_tap, penalizacao_bshunt, max_iter, n_particles, 2, 2, 0.1, valor_inicial = v_inicial,relatorio=True,inicial=True)

print(global_best)


# In[146]:


def otimizacao_alcateia_de_lobos_cinzentos2(sep, zeta, psi, sigma, omega, max_iter, n_lobos,valor_inicial,relatorio=True,inicial=True):
        
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

      
        
        for linha in range(n_wolfs):
          
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

    


# In[156]:


penalizacao_v = 10000
penalizacao_q = 10000
penalizacao_tap = 0
penalizacao_bshunt = 0

max_iter = 30

n_wolfs = 20

sistema = sep30

if len(sistema.bus) == 14:

    v_inicial = inicial_14

if len(sistema.bus) == 30:

    v_inicial = inicial_30
    

if len(sistema.bus) == 118:

    v_inicial = inicial_118


if len(sistema.bus) == 300:

    v_inicial = inicial_300


j,perdas,pen_v,pen_gq,pen_tap,pen_bsh,alcateias,lobo_alfa, lobo_beta, lobo_delta, elapsed = otimizacao_alcateia_de_lobos_cinzentos2(sistema, penalizacao_v, penalizacao_q, penalizacao_tap, penalizacao_bshunt, max_iter, n_wolfs, valor_inicial=v_inicial,relatorio=True,inicial=True)


# In[ ]:


sep14.gen


# In[ ]:


sep14.res_gen


# In[ ]:


sep14.ext_grid


# In[ ]:


cria_alcateia(sep118,1)


# In[ ]:


sep118.ext_grid


# In[ ]:


sep118.res_ext_grid


# In[ ]:


sep14.poly_cost


# In[ ]:


sep30.poly_cost


# In[ ]:


np.min(jota[:,-1])


# In[ ]:


global_b[49]


# In[ ]:


np.mean(tim)


# In[ ]:


pena = jota[:,-1] - per[:,-1]


# In[ ]:


len(pena[pena<0.0001])


# In[ ]:


sep14.res_ext_grid


# In[ ]:


sep118.ext_grid


# In[ ]:


sep118.gen


# In[ ]:


sep14.trafo


# In[ ]:


sep5.line


# In[ ]:


sep5.gen


# In[ ]:


pp.runpp(sep4, algorithm = 'nr')


# In[ ]:


sep4.line


# In[ ]:


sep4.res_ext_grid


# In[ ]:


sep4.ext_grid


# In[ ]:


sep4.load


# In[ ]:


sep30.ext_grid


# In[ ]:


sep30.gen


# In[167]:


lobo_alfa


# In[ ]:




