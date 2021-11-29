#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandapower as pp
from pandapower.networks import case14, case_ieee30, case118, case300, case4gs
from gekko import GEKKO
from pandapower.plotting.plotly import pf_res_plotly
import pandapower.plotting as pplot
from msc_rafael_pavan import otimizacao_pso_discreto_sengi
from msc_rafael_pavan import inicializa_sep
from msc_rafael_pavan import matriz_condutancia
from msc_rafael_pavan import coleta_dados_vbus
from msc_rafael_pavan import coleta_dados_gen
from msc_rafael_pavan import func_objetivo
from msc_rafael_pavan import pen_tensao
from msc_rafael_pavan import pen_ger_reativo
from msc_rafael_pavan import coleta_dados_trafo
from msc_rafael_pavan import pen_trafo
from msc_rafael_pavan import coleta_dados_bshunt
from msc_rafael_pavan import converte_trafo
from msc_rafael_pavan import cria_alcateia
from msc_rafael_pavan import cria_enxame
from msc_rafael_pavan import pen_bshunt
from msc_rafael_pavan import fluxo_de_pot
from msc_rafael_pavan import fitness
from msc_rafael_pavan import validacao
from msc_rafael_pavan import validacao_q
from msc_rafael_pavan import fluxo_de_pot_q

from msc_rafael_pavan import otimizacao_gwo_continuo
from msc_rafael_pavan import otimizacao_pso_continuo
from msc_rafael_pavan import discreto_bshunt
from msc_rafael_pavan import discreto_tap
from msc_rafael_pavan import otimizacao_gwo_discreto
from msc_rafael_pavan import otimizacao_pso_discreto
from msc_rafael_pavan import ipm_ieee30
from msc_rafael_pavan import ipm_ieee14
from msc_rafael_pavan import cria_enxame_v
import copy

from numba import jit
import time

sep_14 = case14()
sep_118 = case118()
sep_30 = case_ieee30()
sep_300 = case300()
v_init30 = inicializa_sep(sep_30,algorithm='fdbx', relatorio=False)
v_init14 = inicializa_sep(sep_14,algorithm='fdbx', relatorio=False)
sep_118 = case118()
v_init118 = inicializa_sep(sep_118,algorithm='fdbx', relatorio=False)

v_init300 = inicializa_sep(sep_300,algorithm='fdbx', relatorio=False)


# In[7]:


def balanco_potencia_ativa(sep,pg_sgen,pg, pc, barras_origem, barras_destino, barra_atual, gkm_linhas, bkm_linhas, tensoes, angulos, to, td, tap, bkmt,gkmt,pshunt):
    
    soma = 0
    
    fluxos = []
    
    linhas = np.arange(0,len(barras_origem),1)
    
    baux = []
    baux.append(33333333)
    baux.append(33333331)
    
    for bd in barras_destino[barras_origem==barra_atual]:
        
        baux.append(bd)
        posi = linhas[(barras_destino==bd) & (barras_origem==barra_atual)][0]
        
        if baux[-2]==bd:
            
#             print(linhas[(barras_destino==bd) & (barras_origem==barra_atual)],'aaaaaaaaaa')
            posi = linhas[(barras_destino==bd) & (barras_origem==barra_atual)][1]
            
#         print(bd,'vai linha')
#         print(posi)
        soma = soma + gkm_linhas[posi]*(tensoes[barra_atual]**2) - tensoes[barra_atual]*tensoes[bd]*(gkm_linhas[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-bkm_linhas[posi]*sep.sin(angulos[barra_atual]-angulos[bd]))
        fluxos.append(gkm_linhas[posi]*(tensoes[barra_atual]**2) - tensoes[barra_atual]*tensoes[bd]*(gkm_linhas[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-bkm_linhas[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
        
    for bd in barras_origem[barras_destino==barra_atual]:
        
        posi = linhas[(barras_destino==barra_atual) & (barras_origem==bd)][0]
        
        baux.append(bd)
        
        if baux[-2]==bd:
#             print(linhas[(barras_destino==barra_atual) & (barras_origem==bd)][1],'aaaaaaaaaa')
            posi = linhas[(barras_destino==barra_atual) & (barras_origem==bd)][1]
        
#         print(bd,'volta linha')
#         print(posi)
        soma = soma + gkm_linhas[posi]*(tensoes[barra_atual]**2) - tensoes[barra_atual]*tensoes[bd]*(gkm_linhas[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-bkm_linhas[posi]*sep.sin(angulos[barra_atual]-angulos[bd]))
        fluxos.append(gkm_linhas[posi]*(tensoes[barra_atual]**2) - tensoes[barra_atual]*tensoes[bd]*(gkm_linhas[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-bkm_linhas[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
    
    linhas = np.arange(0,len(to),1)
    
    for bd in td[to==barra_atual]:
        
        posi = linhas[(td==bd) & (to==barra_atual)][0]
#         print(bd,'vai trafo')
#         print(posi)
        soma = soma + (gkmt[posi]*tensoes[barra_atual]*tensoes[barra_atual]/tap[posi]**2 - (tensoes[barra_atual]*tensoes[bd]/tap[posi])*(gkmt[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-bkmt[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
        
        fluxos.append(gkmt[posi]*tensoes[barra_atual]*tensoes[barra_atual]/tap[posi]**2 - (tensoes[barra_atual]*tensoes[bd]/tap[posi])*(gkmt[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-bkmt[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
    
    for bd in to[td==barra_atual]:
        
        posi = linhas[(td==barra_atual) & (to==bd)][0]
#         print(bd,'volta trafo')
#         print(posi)
        soma = soma + (gkmt[posi]*tensoes[barra_atual]*tensoes[barra_atual]  - (tensoes[barra_atual]*tensoes[bd]/tap[posi])*(gkmt[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-bkmt[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
    
        fluxos.append(gkmt[posi]*tensoes[barra_atual]*tensoes[barra_atual] - (tensoes[barra_atual]*tensoes[bd]/tap[posi])*(gkmt[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-bkmt[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))

    return  pg[barra_atual] - pc[barra_atual] - soma + pg_sgen[barra_atual] - pshunt[barra_atual]*tensoes[barra_atual]**2


# In[8]:


def balanco_potencia_reativa(sep,qg, qc, barras_origem, barras_destino, barra_atual, gkm_linhas, bkm_linhas, tensoes, angulos, to, td, tap, bkmt,gkmt, bshl, bsht,qshunt):
    
    soma = 0
    
    fluxos = []
    
    linhas = np.arange(0,len(barras_origem),1)
    
    baux = []
    baux.append(10101010101)
    baux.append(1000101010)
    
    for bd in barras_destino[barras_origem==barra_atual]:
        
        baux.append(bd)
        posi = linhas[(barras_destino==bd) & (barras_origem==barra_atual)][0]
        
        if baux[-2]==bd:
            
#             print(linhas[(barras_destino==bd) & (barras_origem==barra_atual)],'aaaaaaaaaa')
            posi = linhas[(barras_destino==bd) & (barras_origem==barra_atual)][1]
            
#         print(bd,'vai linha')
#         print(posi)
        
        soma = soma + -(-bkm_linhas[posi]+bshl[posi]/2)*tensoes[barra_atual]**2+tensoes[barra_atual]*tensoes[bd]*(-bkm_linhas[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-gkm_linhas[posi]*sep.sin(angulos[barra_atual]-angulos[bd]))
        fluxos.append(-(-bkm_linhas[posi]+bshl[posi]/2)*tensoes[barra_atual]**2+tensoes[barra_atual]*tensoes[bd]*(-bkm_linhas[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-gkm_linhas[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
        
    for bd in barras_origem[barras_destino==barra_atual]:
        
        posi = linhas[(barras_destino==barra_atual) & (barras_origem==bd)][0]
        
        baux.append(bd)
        
        if baux[-2]==bd:
#             print(linhas[(barras_destino==barra_atual) & (barras_origem==bd)][1],'aaaaaaaaaa')
            posi = linhas[(barras_destino==barra_atual) & (barras_origem==bd)][1]
        
#         print(bd,'volta linha')
#         print(posi)
        
        soma = soma + -(-bkm_linhas[posi]+bshl[posi]/2)*tensoes[barra_atual]**2+tensoes[barra_atual]*tensoes[bd]*(-bkm_linhas[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-gkm_linhas[posi]*sep.sin(angulos[barra_atual]-angulos[bd]))
        fluxos.append(-(-bkm_linhas[posi]+bshl[posi]/2)*tensoes[barra_atual]**2+tensoes[barra_atual]*tensoes[bd]*(-bkm_linhas[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-gkm_linhas[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
        
    linhas = np.arange(0,len(to),1)
    
    for bd in td[to==barra_atual]:
        
        posi = linhas[(td==bd) & (to==barra_atual)][0]
#         print(bd,'vai trafo')
#         print(posi)
        soma = soma + -(-bkmt[posi]/(tap[posi]**2)+bsht[posi]/2)*tensoes[barra_atual]**2+tensoes[barra_atual]*(1/tap[posi])*tensoes[bd]*(-bkmt[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-gkmt[posi]*sep.sin(angulos[barra_atual]-angulos[bd]))
        fluxos.append(-(-bkmt[posi]/(tap[posi]**2)+bsht[posi]/2)*tensoes[barra_atual]**2+tensoes[barra_atual]*(1/tap[posi])*tensoes[bd]*(-bkmt[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-gkmt[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
        
    for bd in to[td==barra_atual]:
        
        posi = linhas[(td==barra_atual) & (to==bd)][0]
#         print(bd,'volta trafo')
#         print(posi)
        soma = soma + -(-bkmt[posi]+bsht[posi]/2)*tensoes[barra_atual]**2+tensoes[barra_atual]*(1/tap[posi])*tensoes[bd]*(-bkmt[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-gkmt[posi]*sep.sin(angulos[barra_atual]-angulos[bd]))
        fluxos.append(-(-bkmt[posi]+bsht[posi]/2)*tensoes[barra_atual]**2+tensoes[barra_atual]*(1/tap[posi])*tensoes[bd]*(-bkmt[posi]*sep.cos(angulos[barra_atual]-angulos[bd])-gkmt[posi]*sep.sin(angulos[barra_atual]-angulos[bd])))
        
    return  qg[barra_atual] - qc[barra_atual] - soma - qshunt[barra_atual]*tensoes[barra_atual]**2


# In[9]:


def perdas(sep,gkml, gkmt, angulos, tensoes, tap, origem, destino, hv, lv):
    
    i = 0
    
    eq = []
 
    for bus in zip(origem,destino):
        
        
        perdas = gkml[i]*(tensoes[bus[0]]**2 + tensoes[bus[1]]**2 - 2*tensoes[bus[1]]*tensoes[bus[0]]*sep.cos(angulos[bus[0]]-angulos[bus[1]]))
        i=i+1
        
        eq.append(perdas)
    
    j = 0
    

    
    for bus in zip(hv,lv):
        
        perdas = gkmt[j]*((tensoes[bus[0]]/tap[j])**2 + tensoes[bus[1]]**2 - 2*tensoes[bus[1]]*tensoes[bus[0]]*(1/tap[j])*sep.cos(angulos[bus[0]]-angulos[bus[1]]))
        
        eq.append(perdas)
            
        j=j+1
        
    return perdas, eq
    


# In[10]:


def voltage_dev(tensoes):
    
    
    sum_dev = 0
    
    for tensao in tensoes:
        dev = (1-tensao)**2

        sum_dev = dev + sum_dev
        
    return sum_dev


# In[11]:


def automatiza_ipm_c(sep_teste, verbose=True, travado=False):
    
    
    sep14 = copy.copy(sep_teste) 
    sep14.res_line = sep14.res_line.sort_index()
    sep14.line = sep14.line.sort_index()
    origem = sep14.line[['from_bus']].values
    destino = sep14.line[['to_bus']].values
    
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
    

    
    sep14.trafo = sep14.trafo.sort_index()

    barras = sep14.trafo['hv_bus'].to_numpy()
    
    zkm = (sep14.trafo['vk_percent'].to_numpy()/100)*(1000/sep14.trafo['sn_mva'].to_numpy())
    
    rkm = (sep14.trafo['vkr_percent'].to_numpy()/100)*(1000/sep14.trafo['sn_mva'].to_numpy())
    
    #a = (sep14.trafo['vn_lv_kv'].to_numpy()*sep14.trafo['vn_lv_kv'].to_numpy()*1000/sep14.trafo['sn_mva'].to_numpy())/(sep14.trafo['vn_lv_kv'].to_numpy()*sep14.trafo['vn_lv_kv'].to_numpy()/1000)
    
    a = 1
    
    zkm=zkm/10
    
    rkm=rkm/10
    
    xkm = np.sqrt(zkm**2-rkm**2)
    

#     xkm[91] = 0.0231
        
    gkmt = (rkm*a/((a*rkm)**2+(a*xkm)**2))
    
    bkmt = (xkm*a/((a*rkm)**2+(a*xkm)**2))
    
    bsht = np.sqrt(np.power(sep14.trafo['i0_percent'].to_numpy()/100,2))
    
    bsht = bsht*99
    
    print(bsht)
    

        ########################################################################### Vetor de tap

    tap_pos = sep14.trafo[~pd.isnull(sep14.trafo['tap_pos'])]['tap_pos'].to_numpy(dtype=np.float64)

    tap_neutral = sep14.trafo[~pd.isnull(sep14.trafo['tap_neutral'])]['tap_neutral'].to_numpy(dtype=np.float64)

    tap_step_percent = sep14.trafo[~pd.isnull(sep14.trafo['tap_step_percent'])]['tap_step_percent'].to_numpy(dtype=np.float64)

    valor_percentual = (tap_pos-tap_neutral)*(tap_step_percent/100) + 1

#     valor_percentual = np.resize(valor_percentual,(len(sep14.trafo)))


    to = np.zeros(len(sep14.trafo))
    td = np.zeros(len(sep14.trafo))
    
    for i in range(len(sep14.trafo)):
        
        if sep14.trafo['tap_side'].iloc[i] == None or sep14.trafo['tap_side'].iloc[i] == 'hv':
        
            to[i] = int(sep14.trafo['hv_bus'].iloc[i])


            td[i] = int(sep14.trafo['lv_bus'].iloc[i])

        if sep14.trafo['tap_side'].iloc[i] == 'lv':
        
            to[i] = int(sep14.trafo['lv_bus'].iloc[i])


            td[i] = int(sep14.trafo['hv_bus'].iloc[i])
            
    to = to.astype(int)
    td = td.astype(int)

    i = 0

    for i in range(len(valor_percentual)):

        if i < len(tap_pos):

            valor_percentual[i] = valor_percentual[i]

        else:

            valor_percentual[i] = 1


    tap = valor_percentual
    
    sep14.trafo['tap_pos'][~pd.isnull(sep14.trafo['tap_pos'])] = valor_percentual
    sep14.trafo['tap_pos'][pd.isnull(sep14.trafo['tap_pos'])] = 1
    
    tap = sep14.trafo['tap_pos'].values
    
    
    ########################################################################### Vetor de tensões das barras
    sep14.line = sep14.line.sort_index()

    sep14.res_bus = sep14.res_bus.sort_index()
    
    sep14.sgen = sep14.sgen.sort_index()

    v = sep14.res_bus['vm_pu'].to_numpy()
    Sbase=100
    ########################################################################### Vetor de ângulos das barras

    theta = np.radians(sep14.res_bus['va_degree'].to_numpy())

    ########################################################################### Vetor de potência ativa gerada
    sep14.gen = sep14.gen.sort_index()
    
    pg = np.zeros(len(sep14.bus))
    
    pgs_max = np.zeros(len(sep14.bus))
    
    pgs_min = np.zeros(len(sep14.bus))
    
    pg_sgen = np.zeros(len(sep14.bus))
    qg_sgen = np.zeros(len(sep14.bus))
    
    i = 0

    sep14.gen = sep14.gen.sort_index()

    sep14.res_gen = sep14.res_gen.sort_index()

    for bus in sep14.gen['bus'].to_numpy():

        pg[bus] = sep14.gen['p_mw'].to_numpy()[i]/Sbase
        pgs_max[bus] = sep14.gen['max_p_mw'].to_numpy()[i]/Sbase
        pgs_min[bus] = sep14.gen['min_p_mw'].to_numpy()[i]/Sbase
        i=i+1
        

    i=0
    
    qg = np.zeros(len(pg))
   
    for bus in sep14.sgen['bus'].to_numpy():
        
        pg_sgen[bus] = sep14.sgen['p_mw'].to_numpy()[i]/Sbase
        
        qg_sgen[bus] = sep14.sgen['q_mvar'].to_numpy()[i]/Sbase

        i = i+1
    
    if len(sep14.bus)==118:

        pg[68] = sep14.res_ext_grid['p_mw'].to_numpy()/100
        qg[68] = sep14.res_ext_grid['q_mvar'].to_numpy()/100
        pgs_max[68] = sep14.ext_grid['max_p_mw'].to_numpy()/100
        pgs_min[68] = sep14.ext_grid['min_p_mw'].to_numpy()/100
        
    if len(sep14.bus)==300:

        pg[256] = sep14.res_ext_grid['p_mw'].to_numpy()/100
        qg[256] = sep14.res_ext_grid['q_mvar'].to_numpy()/100
        

    
    if len(sep14.bus)==14 or len(sep14.bus)==30 :

        pg[0] = sep14.res_ext_grid['p_mw'].to_numpy()/100
        pgs_max[0] = sep14.ext_grid['max_p_mw'].to_numpy()/100
        pgs_min[0] = sep14.ext_grid['min_p_mw'].to_numpy()/100
        
    pg_ls = sep14.ext_grid['max_p_mw'].to_numpy()/100

    pg_li = sep14.ext_grid['min_p_mw'].to_numpy()/100
    
    sep14.load = sep14.load.sort_index()
    pc = np.zeros(len(sep14.bus))

    i = 0

    sep14.load = sep14.load.sort_index()

    for bus in sep14.load['bus'].to_numpy():

        pc[bus] = sep14.load['p_mw'].to_numpy()[i]/Sbase

        i=i+1

    qc = np.zeros(len(sep14.bus))

    i = 0

    for bus in sep14.load['bus'].to_numpy():

        qc[bus] = sep14.load['q_mvar'].to_numpy()[i]/Sbase

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
    
    qg = np.zeros(len(pg))

    if len(sep14.bus)==14 or len(sep14.bus)==30:

        qg[0] = sep14.res_ext_grid['q_mvar'].values/100

    if len(sep14.bus)==118:

        qg[68] = sep14.res_ext_grid['q_mvar'].values/100


    if len(sep14.bus)==300:

        qg[256] = sep14.res_ext_grid['q_mvar'].values/100


    sepaux = sep14.gen['bus']

    for barra in sepaux:

        qg[barra]=sep14.res_gen[sep14.gen['bus']==barra]['q_mvar'].values/100


    barras = sep14.shunt['bus'].to_numpy()
    qshunt = np.zeros(np.shape(qg))
    pshunt = np.zeros(np.shape(pg))

    for barra in barras:

        qshunt[barra]=sep14.shunt[sep14.shunt['bus']==barra]['q_mvar'].values/100
        pshunt[barra]=sep14.shunt[sep14.shunt['bus']==barra]['p_mw'].values/100

    hv=sep14.trafo['hv_bus'].values
    lv=sep14.trafo['lv_bus'].values

    if len(sep14.bus) == 118:
        


        tensoes = []
        angulos = []

        sep14.res_bus = sep14.res_bus.sort_index()


        v = sep14.res_bus['vm_pu'].to_numpy()


        theta = np.radians(sep14.res_bus['va_degree'].to_numpy())
        sep = GEKKO()

        for bus in range(len(sep14.bus)):
        
            if len(sep14.bus)==118:
                
                tensoes.append(sep.Var(v[bus],0.94,1.06))
                
            else: tensoes.append(sep.Var(v[bus],0.95,1.05))
            
            angulos.append(sep.Var(theta[bus],-np.pi,np.pi))


        shunt = np.zeros(len(sep14.bus)).tolist()

#         shunt[4]= sep.Var(qshunt[4],0,0.40)
#         shunt[33]= sep.Var(qshunt[33],-0.2,0)
#         shunt[36]= sep.Var(qshunt[36],0,0.25)
#         shunt[43]= sep.Var(qshunt[43],-0.1,0)
#         shunt[44]= sep.Var(qshunt[44],-0.1,0)
#         shunt[45]= sep.Var(qshunt[45],-0.1,0)
#         shunt[47]= sep.Var(qshunt[47],-0.15,0)
#         shunt[73]= sep.Var(qshunt[73],-0.2,0)
#         shunt[78]= sep.Var(qshunt[78],-0.2,0)
#         shunt[81]= sep.Var(qshunt[81],-0.2,0)
#         shunt[82]= sep.Var(qshunt[82],-0.2,0)
#         shunt[104]= sep.Var(qshunt[104],-0.2,0)
#         shunt[106]= sep.Var(qshunt[106],-0.2,0)
#         shunt[109]= sep.Var(qshunt[109],-0.2,0)


        shunt[4]= sep.Var(qshunt[4],0,0.40)
        shunt[33]= sep.Var(qshunt[33],-0.14,0)
        shunt[36]= sep.Var(qshunt[36],0,0.25)
        shunt[43]= sep.Var(qshunt[43],-0.1,0)
        shunt[44]= sep.Var(qshunt[44],-0.1,0)
        shunt[45]= sep.Var(qshunt[45],-0.1,0)
        shunt[47]= sep.Var(qshunt[47],-0.15,0)
        shunt[73]= sep.Var(qshunt[73],-0.12,0)
        shunt[78]= sep.Var(qshunt[78],-0.2,0)
        shunt[81]= sep.Var(qshunt[81],-0.2,0)
        shunt[82]= sep.Var(qshunt[82],-0.1,0)
        shunt[104]= sep.Var(qshunt[104],-0.2,0)
        shunt[106]= sep.Var(qshunt[106],-0.06,0)
        shunt[109]= sep.Var(qshunt[109],-0.06,0)


#         shunt[4]= sep.sos1([0,0.4])
#         shunt[33]= sep.sos1([0,-0.06,-0.07,-0.13,-0.14,-0.2])
#         shunt[36]= sep.sos1([0,0.25])
#         shunt[43]= sep.sos1([0,-0.1])
#         shunt[44]= sep.sos1([0,-0.1])
#         shunt[45]= sep.sos1([0,-0.1])
#         shunt[47]= sep.sos1([0,-0.15])

#         shunt[73]= sep.sos1([0,-0.08,-0.12,-0.2])

#         shunt[78]= sep.sos1([0,-0.1,-0.2])

#         shunt[81]= sep.sos1([0,-0.1,-0.2])

#         shunt[82]= sep.sos1([0,-0.1,-0.2])

#         shunt[104]= sep.sos1([0,-0.1,-0.2])

#         shunt[106]= sep.sos1([0,-0.06,-0.07,-0.13,-0.14,-0.2])
#         shunt[109]= sep.sos1([0,-0.06,-0.07,-0.13,-0.14,-0.2])


        

        sep14.trafo['tap_pos'][sep14.trafo['tap_pos']==np.nan] = 1

        tap = sep14.trafo['tap_pos'].to_numpy()

        taps = []

        for valor in tap:

            if valor !=1:
                
                if len(sep14.bus)==118:
                    taps.append(sep.Var(valor,0.9,1.1))
                else: taps.append(sep.Var(valor,0.95,1.05))
#                 taps.append(sep.sos1([0.88,0.8875,0.895,0.9025,0.91,0.9175,0.925,0.9325,0.94,0.9475,0.955,0.9625,0.97,0.9775,0.985,0.9925,1.0,1.0075,1.015,1.0225,1.03,1.0375,1.045,1.0525,1.06,1.0675,1.075,1.0825,1.09,1.0975,1.105,1.1125,1.12]))

            else: 
                taps.append(1)

        if travado == True:
            taps=tap
            shunt = qshunt

    


        qgs = np.zeros(len(pg))

        qg = qgs.tolist()

        pgs = np.zeros(len(pg))
        
        pgs = pgs.tolist()

        for bus in sep14.gen['bus'].to_numpy():

            qg[bus] = sep.Var((sep14.res_gen[sep14.gen['bus']==bus]['q_mvar'].to_numpy()/100)[0], (sep14.gen[sep14.gen['bus']==bus]['min_q_mvar'].to_numpy()/100)[0],(sep14.gen[sep14.gen['bus']==bus]['max_q_mvar'].to_numpy()/100)[0] )  
            pgs[bus] = sep.Var(pg[bus],pgs_min[bus],pgs_max[bus])

        qg[68] = sep.Var((sep14.res_ext_grid['q_mvar'].to_numpy()/100)[0],(sep14.ext_grid['min_q_mvar'].to_numpy()/100)[0],(sep14.ext_grid['max_q_mvar'].to_numpy()/100)[0])
        
        pgs[68] = sep.Var(pg[68],pgs_min[68],pgs_max[68])

        qg[68] = sep.Var((sep14.res_ext_grid['q_mvar'].to_numpy()/100)[0],(sep14.ext_grid['min_q_mvar'].to_numpy()/100)[0],(sep14.ext_grid['max_q_mvar'].to_numpy()/100)[0])
        
       
        for barra in range(0,len(sep14.bus)):


            sep.Equation(balanco_potencia_reativa(sep,qg, qc, origem.ravel(), destino.ravel(), barra, gkm, bkm, tensoes, angulos, to, td, taps, bkmt,gkmt, bsh, bsht,shunt)==0)
        
        for barra in range(0,len(sep14.bus)):

    
            sep.Equation(balanco_potencia_ativa(sep,pg_sgen,pgs,pc,origem.ravel(), destino.ravel(), barra, gkm, bkm, tensoes, angulos, to,td,taps,bkmt,gkmt,pshunt)==0)
    
        a, equations = perdas(sep,gkm, gkmt, angulos, tensoes, taps, origem.ravel(), destino.ravel(), to, td)
        
        
#         custo

        polycosts = sep14.poly_cost[sep14.poly_cost['et'] != 'ext_grid']

        index_list = sep14.gen.index.values.tolist()

        a_k = []
        b_k = []
        c_k = []

        for val in index_list:

            a_k.append(polycosts['cp2_eur_per_mw2'][polycosts['element']==val].values[0])
            b_k.append(polycosts['cp1_eur_per_mw'][polycosts['element']==val].values[0])
            c_k.append(polycosts['cp0_eur'][polycosts['element']==val].values[0])

        pmax0 = sep14.ext_grid['max_p_mw'].values
        pmin0 = sep14.ext_grid['min_p_mw'].values

        a_k0=sep14.poly_cost['cp2_eur_per_mw2'][sep14.poly_cost['et']=='ext_grid'].values
        b_k0=sep14.poly_cost['cp1_eur_per_mw'][sep14.poly_cost['et']=='ext_grid'].values
        c_k0=sep14.poly_cost['cp0_eur'][sep14.poly_cost['et']=='ext_grid'].values
        e_k0 = (5/100)*((a_k0*pmax0**2 + b_k0*pmax0 + c_k0 + a_k0*pmin0**2 + b_k0*pmin0 + c_k0 )/2)
        f_k0 = (4*np.pi/(sep14.ext_grid['max_p_mw'].values-sep14.ext_grid['min_p_mw'].values))

        e_k = (5/100)*((a_k*sep14.gen['max_p_mw'].values**2 + b_k*sep14.gen['max_p_mw'].values + c_k + a_k*sep14.gen['min_p_mw'].values**2 + b_k*sep14.gen['max_p_mw'].values + c_k )/2)
        f_k = (4*np.pi/(sep14.gen['max_p_mw'].values-sep14.gen['min_p_mw'].values))

        a_k = np.concatenate((a_k0, np.array(a_k)))
        b_k = np.concatenate((b_k0, np.array(b_k)))
        c_k = np.concatenate((c_k0, np.array(c_k)))
        e_k = np.concatenate((e_k0, np.array(e_k)))
        f_k = np.concatenate((f_k0, np.array(f_k)))
            
        alfa = np.zeros(len(sep14.gen['bus'])+1)
        
        alfa = alfa.tolist()
        
        
        alfa[0] = sep.Var(sep.abs(e_k[0]*sep.sin(f_k[0]*(100*pgs_min[68]-100*pgs[68]))),0,1e15)
        
        u=1
        for bus in sep14.gen['bus'].to_numpy():
            alfa[u]=sep.Var(sep.abs(e_k[u]*sep.sin(f_k[u]*(100*pgs_min[u]-100*pgs[u]))),0,1e15)
            sep.Equation(-alfa[u]<=e_k[u]*sep.sin(f_k[u]*(100*pgs_min[bus]-100*pgs[bus])))
            sep.Equation(e_k[u]*sep.sin(f_k[u]*(100*pgs_min[bus]-100*pgs[bus]))<=alfa[u])
            u=u+1
            
        sep.Equation(e_k[0]*sep.sin(f_k[0]*(100*pgs_min[68]-100*pgs[68]))<=alfa[0])
        
        lista_custos = []
        
        u=1
        for bus in sep14.gen['bus'].to_numpy():
            
            lista_custos.append(((100*pgs[bus])**2)*a_k[u]+pgs[bus]*b_k[u]*100+c_k[u]+alfa[u])
            u=u+1
        
        lista_custos.append(((100*pgs[68])**2)*a_k[0]+100*pgs[68]*b_k[0]+c_k[0]+alfa[0])
            
        sep.Obj(sep.sum(lista_custos))

            
            
#         sep.Obj(sep.sum(equations)) # perdas
        
#         sep.Obj(voltage_dev(tensoes)) # desvio
        
        
        sep.options.SOLVER = 3
        sep.options.RTOL = 1e-6
        
        sep.solver_options = ['tol 1e-5',
                              'constr_viol_tol 1e-5',
                              'bound_push 1e-16',
                              'bound_frac 1e-16',
                              'warm_start_init_point yes',
                              'warm_start_bound_push 1e-16',
                              'warm_start_slack_bound_push 1e-16',
                              'warm_start_slack_bound_frac 1e-16',
                              'warm_start_bound_frac 1e-16']  
        
        sep.solve(disp=verbose)
        
        tensao = np.zeros(len(sep14.gen['bus'].to_numpy())+1)
        i=1
        tensao[0] = tensoes[68][0]

        for bus in sep14.gen['bus'].to_numpy():

            tensao[i] =  tensoes[bus][0]
            i=i+1
        
        if travado == False:
            t = np.array([taps[0],taps[1],taps[2],taps[3],taps[4],taps[5],taps[6],taps[8],taps[10]])
        else:
            t = np.array([taps[0],taps[1],taps[2],taps[3],taps[4],taps[5],taps[6],taps[8],taps[10]])

        s = np.zeros(len(sep14.shunt['bus'].to_numpy()))

        i=0
        for bus in sep14.shunt['bus'].to_numpy():


            
            if travado == True:
                s[i] =  shunt[bus]
            else:
                s[i] =  shunt[bus][0]

            i=i+1

        s = s*-1
        
        sep14.res_bus= sep14.res_bus.sort_index()

        thetas = np.zeros(len(angulos))

        voltages = np.zeros(len(angulos))

        pot_reativas = np.zeros(len(qg))


        for i in range(len(angulos)):

            thetas[i]=angulos[i][0]

            voltages[i]=tensoes[i][0]



        sep14.res_bus['vm_pu'] = voltages


        sep14.res_ext_grid['p_mw'] = pgs[68][0]*100
        
        u=0
        for bus in sep14.gen['bus'].to_numpy():
            
            sep14.gen['p_mw'][int(u)] = pgs[int(bus)][0]*100
            u=u+1



        sep14.res_ext_grid['q_mvar'] = qg[68][0]*100

        sep14.shunt['q_mvar'] = s*100
        
        gbest = [tensao,t,s,np.array([0,0,0,0,0,0])]
        
        sep14.trafo['tap_pos'][pd.isnull(sep14.trafo['tap_step_percent'])]=np.nan
        
        sep14.bus['max_vm_pu'] = 1.06

        sep14.bus['min_vm_pu'] = 0.94
        sep14.gen['min_vm_pu'] = 0.94

        sep14.gen['max_vm_pu'] = 1.06
        
        sep14.ext_grid['min_vm_pu'] = 0.94

        sep14.ext_grid['max_vm_pu'] = 1.06
        
    elif len(sep14.bus) == 14:
        
        print('ENTROU 14')
        
        tensoes = []
        angulos = []

        sep14.res_bus = sep14.res_bus.sort_index()


        v = sep14.res_bus['vm_pu'].to_numpy()


        theta = np.radians(sep14.res_bus['va_degree'].to_numpy())
        sep = GEKKO()
        

        for bus in range(len(sep14.bus)):

            tensoes.append(sep.Var(v[bus],0.95,1.05))
            angulos.append(sep.Var(theta[bus],0,2*np.pi))


        shunt = np.zeros(len(sep14.bus)).tolist()
            
        shunt[8]= sep.Var(qshunt[8],-0.05,0)
        
        sep14.trafo['tap_pos'][sep14.trafo['tap_pos']==np.nan] = 1

        tap = sep14.trafo['tap_pos'].to_numpy()

        taps = []

        for valor in tap:

            if valor !=1:

                taps.append(sep.Var(valor,0.95,1.05))
#                 taps.append(sep.sos1([0.88,0.8875,0.895,0.9025,0.91,0.9175,0.925,0.9325,0.94,0.9475,0.955,0.9625,0.97,0.9775,0.985,0.9925,1.0,1.0075,1.015,1.0225,1.03,1.0375,1.045,1.0525,1.06,1.0675,1.075,1.0825,1.09,1.0975,1.105,1.1125,1.12]))

            else: 
                taps.append(1)

        if travado == True:
            taps=tap
            shunt = qshunt

    

        qgs = np.zeros(len(pg))

        qg = qgs.tolist()

        pgs = np.zeros(len(pg))
        
        pgs = pgs.tolist()

        for bus in sep14.gen['bus'].to_numpy():

            qg[bus] = sep.Var((sep14.res_gen[sep14.gen['bus']==bus]['q_mvar'].to_numpy()/100)[0], (sep14.gen[sep14.gen['bus']==bus]['min_q_mvar'].to_numpy()/100)[0],(sep14.gen[sep14.gen['bus']==bus]['max_q_mvar'].to_numpy()/100)[0] )  
            pgs[bus] = sep.Var(pg[bus],pgs_min[bus],pgs_max[bus])

        qg[0] = sep.Var((sep14.res_ext_grid['q_mvar'].to_numpy()/100)[0],(sep14.ext_grid['min_q_mvar'].to_numpy()/100)[0],(sep14.ext_grid['max_q_mvar'].to_numpy()/100)[0])
        
        pgs[0] = sep.Var(pg[0],pgs_min[0],pgs_max[0])
        
        
         
        
        for barra in range(0,len(sep14.bus)):


            sep.Equation(balanco_potencia_reativa(sep,qg, qc, origem.ravel(), destino.ravel(), barra, gkm, bkm, tensoes, angulos, to, td, taps, bkmt,gkmt, bsh, bsht,shunt)==0)
        
        for barra in range(0,len(sep14.bus)):

    
            sep.Equation(balanco_potencia_ativa(sep,pg_sgen,pgs,pc,origem.ravel(), destino.ravel(), barra, gkm, bkm, tensoes, angulos, to,td,taps,bkmt,gkmt,pshunt)==0)
    
        a, equations = perdas(sep,gkm, gkmt, angulos, tensoes, taps, origem.ravel(), destino.ravel(), to, td)
        
#         custo

        polycosts = sep14.poly_cost[sep14.poly_cost['et'] != 'ext_grid']

        index_list = sep14.gen.index.values.tolist()

        a_k = []
        b_k = []
        c_k = []

        for val in index_list:

            a_k.append(polycosts['cp2_eur_per_mw2'][polycosts['element']==val].values[0])
            b_k.append(polycosts['cp1_eur_per_mw'][polycosts['element']==val].values[0])
            c_k.append(polycosts['cp0_eur'][polycosts['element']==val].values[0])

        pmax0 = sep14.ext_grid['max_p_mw'].values
        pmin0 = sep14.ext_grid['min_p_mw'].values

        a_k0=sep14.poly_cost['cp2_eur_per_mw2'][sep14.poly_cost['et']=='ext_grid'].values
        b_k0=sep14.poly_cost['cp1_eur_per_mw'][sep14.poly_cost['et']=='ext_grid'].values
        c_k0=sep14.poly_cost['cp0_eur'][sep14.poly_cost['et']=='ext_grid'].values
        e_k0 = (5/100)*((a_k0*pmax0**2 + b_k0*pmax0 + c_k0 + a_k0*pmin0**2 + b_k0*pmin0 + c_k0 )/2)
        f_k0 = (4*np.pi/(sep14.ext_grid['max_p_mw'].values-sep14.ext_grid['min_p_mw'].values))

        e_k = (5/100)*((a_k*sep14.gen['max_p_mw'].values**2 + b_k*sep14.gen['max_p_mw'].values + c_k + a_k*sep14.gen['min_p_mw'].values**2 + b_k*sep14.gen['max_p_mw'].values + c_k )/2)
        f_k = (4*np.pi/(sep14.gen['max_p_mw'].values-sep14.gen['min_p_mw'].values))

        a_k = np.concatenate((a_k0, np.array(a_k)))
        b_k = np.concatenate((b_k0, np.array(b_k)))
        c_k = np.concatenate((c_k0, np.array(c_k)))
        e_k = np.concatenate((e_k0, np.array(e_k)))
        f_k = np.concatenate((f_k0, np.array(f_k)))
            
        
        alfa = np.zeros(len(sep14.gen['bus'])+1)
        
        alfa = alfa.tolist()
        
        
        alfa[0] = sep.Var(sep.abs(e_k[0]*sep.sin(f_k[0]*(100*pgs_min[0]-100*pgs[0]))),0,1e15)
        
        u=1
        for bus in sep14.gen['bus'].to_numpy():
            alfa[u]=sep.Var(sep.abs(e_k[u]*sep.sin(f_k[u]*(100*pgs_min[u]-100*pgs[u]))),0,1e15)
            sep.Equation(-alfa[u]<=e_k[u]*sep.sin(f_k[u]*(100*pgs_min[bus]-100*pgs[bus])))
            sep.Equation(e_k[u]*sep.sin(f_k[u]*(100*pgs_min[bus]-100*pgs[bus]))<=alfa[u])
            u=u+1
            
        sep.Equation(e_k[0]*sep.sin(f_k[0]*(100*pgs_min[0]-100*pgs[0]))<=alfa[0])
        
        lista_custos = []
        
        u=1
        for bus in sep14.gen['bus'].to_numpy():
            
            lista_custos.append(((100*pgs[bus])**2)*a_k[u]+pgs[bus]*b_k[u]*100+c_k[u]+alfa[u])
            u=u+1
        
        lista_custos.append(((100*pgs[0])**2)*a_k[0]+100*pgs[0]*b_k[0]+c_k[0]+alfa[0])
            
        sep.Obj(sep.sum(lista_custos))

        
        
        
        
#         sep.Obj(sep.sum(equations)) # perdas
        
        
#         sep.Obj(voltage_dev(tensoes)) # desvio de Tensão
        sep.options.SOLVER = 3
        sep.options.RTOL = 1e-6
        
        sep.solver_options = ['tol 1e-5',
                              'constr_viol_tol 1e-5',
                              'bound_push 1e-16',
                              'bound_frac 1e-16',
                              'warm_start_init_point yes',
                              'warm_start_bound_push 1e-16',
                              'warm_start_slack_bound_push 1e-16',
                              'warm_start_slack_bound_frac 1e-16',
                              'warm_start_bound_frac 1e-16',
                              'bound_frac 1e-6']  
        sep.solve(disp=verbose)
        
        tensao = np.zeros(len(sep14.gen['bus'].to_numpy())+1)
        i=1
        tensao[0] = tensoes[0][0]

        for bus in sep14.gen['bus'].to_numpy():

            tensao[i] =  tensoes[bus][0]
            i=i+1
        
        
        if travado == False:
            t =np.array([taps[0][0],taps[1][0],taps[2][0]])
        else:
            t =np.array([taps[0],taps[1],taps[2]])

        s = np.zeros(len(sep14.shunt['bus'].to_numpy()))

        i=0
        for bus in sep14.shunt['bus'].to_numpy():


            
            if travado == True:
                s[i] =  shunt[bus]
            else:
                s[i] =  shunt[bus][0]

            i=i+1

        s = s*-1
        
        sep14.res_bus= sep14.res_bus.sort_index()

        thetas = np.zeros(len(angulos))

        voltages = np.zeros(len(angulos))

        pot_reativas = np.zeros(len(qg))


        for i in range(len(angulos)):

            thetas[i]=angulos[i][0]

            voltages[i]=tensoes[i][0]



        sep14.res_bus['vm_pu'] = voltages


        sep14.res_ext_grid['p_mw'] = pgs[0][0]*100
        
        u=0
        for bus in sep14.gen['bus'].to_numpy():
            
            sep14.gen['p_mw'][u] = pgs[bus][0]*100
            u=u+1


        sep14.res_ext_grid['q_mvar'] = qg[0][0]*100

        sep14.shunt['q_mvar'] = s*100
        
        gbest = np.concatenate((tensao,t,s,np.array([0,0,0,0,0,0])),axis=0)
        
        sep14.trafo['tap_pos'][pd.isnull(sep14.trafo['tap_step_percent'])]=np.nan
        
        sep14.bus['max_vm_pu'] = 1.05

        sep14.bus['min_vm_pu'] = 0.95
        sep14.gen['min_vm_pu'] = 0.95

        sep14.gen['max_vm_pu'] = 1.05
        
        sep14.ext_grid['min_vm_pu'] = 0.95

        sep14.ext_grid['max_vm_pu'] = 1.05

    elif len(sep14.bus) == 30:
        
        
        tensoes = []
        angulos = []

        sep14.res_bus = sep14.res_bus.sort_index()


        v = sep14.res_bus['vm_pu'].to_numpy()


        theta = np.radians(sep14.res_bus['va_degree'].to_numpy())
        sep = GEKKO()

        for bus in range(len(sep14.bus)):

            tensoes.append(sep.Var(v[bus],0.95,1.05))
            angulos.append(sep.Var(theta[bus],-np.pi,np.pi))

        shunt = np.zeros(len(sep14.bus)).tolist()
            
        shunt[9]= sep.Var(qshunt[9],-0.05,0)
        shunt[23]= sep.Var(qshunt[23],-0.05,0)
        
        sep14.trafo['tap_pos'][sep14.trafo['tap_pos']==np.nan] = 1

        tap = sep14.trafo['tap_pos'].to_numpy()

        taps = []

        for valor in tap:

            if valor !=1:

                taps.append(sep.Var(valor,0.95,1.05))
#                 taps.append(sep.sos1([0.88,0.8875,0.895,0.9025,0.91,0.9175,0.925,0.9325,0.94,0.9475,0.955,0.9625,0.97,0.9775,0.985,0.9925,1.0,1.0075,1.015,1.0225,1.03,1.0375,1.045,1.0525,1.06,1.0675,1.075,1.0825,1.09,1.0975,1.105,1.1125,1.12]))

            else: 
                taps.append(1)

        if travado == True:
            taps=tap
            shunt = qshunt


        qgs = np.zeros(len(pg))

        qg = qgs.tolist()

        pgs = np.zeros(len(pg))
        
        pgs = pgs.tolist()

        for bus in sep14.gen['bus'].to_numpy():

            qg[bus] = sep.Var((sep14.res_gen[sep14.gen['bus']==bus]['q_mvar'].to_numpy()/100)[0], (sep14.gen[sep14.gen['bus']==bus]['min_q_mvar'].to_numpy()/100)[0],(sep14.gen[sep14.gen['bus']==bus]['max_q_mvar'].to_numpy()/100)[0] )  
            pgs[bus] = sep.Var(pg[bus],pgs_min[bus],pgs_max[bus])

        qg[0] = sep.Var((sep14.res_ext_grid['q_mvar'].to_numpy()/100)[0],(sep14.ext_grid['min_q_mvar'].to_numpy()/100)[0],(sep14.ext_grid['max_q_mvar'].to_numpy()/100)[0])
        
        pgs[0] = sep.Var(pg[0],pgs_min[0],pgs_max[0])
        
        
         
        
        for barra in range(0,len(sep14.bus)):


            sep.Equation(balanco_potencia_reativa(sep,qg, qc, origem.ravel(), destino.ravel(), barra, gkm, bkm, tensoes, angulos, to, td, taps, bkmt,gkmt, bsh, bsht,shunt)==0)
        
        for barra in range(0,len(sep14.bus)):

    
            sep.Equation(balanco_potencia_ativa(sep,pg_sgen,pgs,pc,origem.ravel(), destino.ravel(), barra, gkm, bkm, tensoes, angulos, to,td,taps,bkmt,gkmt,pshunt)==0)
    
        a, equations = perdas(sep,gkm, gkmt, angulos, tensoes, taps, origem.ravel(), destino.ravel(), to, td)
        
#         custo

        polycosts = sep14.poly_cost[sep14.poly_cost['et'] != 'ext_grid']

        index_list = sep14.gen.index.values.tolist()

        a_k = []
        b_k = []
        c_k = []

        for val in index_list:

            a_k.append(polycosts['cp2_eur_per_mw2'][polycosts['element']==val].values[0])
            b_k.append(polycosts['cp1_eur_per_mw'][polycosts['element']==val].values[0])
            c_k.append(polycosts['cp0_eur'][polycosts['element']==val].values[0])

        pmax0 = sep14.ext_grid['max_p_mw'].values
        pmin0 = sep14.ext_grid['min_p_mw'].values

        a_k0=sep14.poly_cost['cp2_eur_per_mw2'][sep14.poly_cost['et']=='ext_grid'].values
        b_k0=sep14.poly_cost['cp1_eur_per_mw'][sep14.poly_cost['et']=='ext_grid'].values
        c_k0=sep14.poly_cost['cp0_eur'][sep14.poly_cost['et']=='ext_grid'].values
        e_k0 = (5/100)*((a_k0*pmax0**2 + b_k0*pmax0 + c_k0 + a_k0*pmin0**2 + b_k0*pmin0 + c_k0 )/2)
        f_k0 = (4*np.pi/(sep14.ext_grid['max_p_mw'].values-sep14.ext_grid['min_p_mw'].values))

        e_k = (5/100)*((a_k*sep14.gen['max_p_mw'].values**2 + b_k*sep14.gen['max_p_mw'].values + c_k + a_k*sep14.gen['min_p_mw'].values**2 + b_k*sep14.gen['max_p_mw'].values + c_k )/2)
        f_k = (4*np.pi/(sep14.gen['max_p_mw'].values-sep14.gen['min_p_mw'].values))

        a_k = np.concatenate((a_k0, np.array(a_k)))
        b_k = np.concatenate((b_k0, np.array(b_k)))
        c_k = np.concatenate((c_k0, np.array(c_k)))
        e_k = np.concatenate((e_k0, np.array(e_k)))
        f_k = np.concatenate((f_k0, np.array(f_k)))
                
        alfa = np.zeros(len(sep14.gen['bus'])+1)
        
        alfa = alfa.tolist()
        
        
        alfa[0] = sep.Var(sep.abs(e_k[0]*sep.sin(f_k[0]*(100*pgs_min[0]-100*pgs[0]))),0,1e15)
        
        u=1
        for bus in sep14.gen['bus'].to_numpy():
            alfa[u]=sep.Var(sep.abs(e_k[u]*sep.sin(f_k[u]*(100*pgs_min[u]-100*pgs[u]))),0,1e15)
            sep.Equation(-alfa[u]<=e_k[u]*sep.sin(f_k[u]*(100*pgs_min[bus]-100*pgs[bus])))
            sep.Equation(e_k[u]*sep.sin(f_k[u]*(100*pgs_min[bus]-100*pgs[bus]))<=alfa[u])
            u=u+1
            
        sep.Equation(e_k[0]*sep.sin(f_k[0]*(100*pgs_min[0]-100*pgs[0]))<=alfa[0])
        
        lista_custos = []
        
        u=1
        for bus in sep14.gen['bus'].to_numpy():
            
            lista_custos.append(((100*pgs[bus])**2)*a_k[u]+pgs[bus]*b_k[u]*100+c_k[u]+alfa[u])
            u=u+1
        
        lista_custos.append(((100*pgs[0])**2)*a_k[0]+100*pgs[0]*b_k[0]+c_k[0]+alfa[0])
            
        sep.Obj(sep.sum(lista_custos))

        
        
        
        
#         sep.Obj(sep.sum(equations)) # perdas
        
        
#         sep.Obj(voltage_dev(tensoes)) # desvio de Tensão
        
        sep.options.SOLVER = 3
#         sep.options.RTOL = 1e-6
        

        sep.solver_options = ['tol 1e-5',
                              'constr_viol_tol 1e-5',
                              'bound_push 1e-16',
                              'bound_frac 1e-16',
                              'warm_start_init_point yes',
                              'warm_start_bound_push 1e-16',
                              'warm_start_slack_bound_push 1e-16',
                              'warm_start_slack_bound_frac 1e-16',
                              'warm_start_bound_frac 1e-16']  
        
        
        sep.solve(disp=verbose)
        
        tensao = np.zeros(len(sep14.gen['bus'].to_numpy())+1)
        i=1
        tensao[0] = tensoes[0][0]

        for bus in sep14.gen['bus'].to_numpy():

            tensao[i] =  tensoes[bus][0]
            i=i+1
     

        
        if travado == False:
            print(taps)
            
            t =np.array([taps[0],taps[1],taps[4],taps[6]])
        else:
            t =np.array([taps[0],taps[1],taps[4],taps[6]])

        s = np.zeros(len(sep14.shunt['bus'].to_numpy()))

        i=0
        for bus in sep14.shunt['bus'].to_numpy():


            
            if travado == True:
                s[i] =  shunt[bus]
            else:
                s[i] =  shunt[bus][0]

            i=i+1

        s = s*-1
        
        sep14.res_bus= sep14.res_bus.sort_index()

        thetas = np.zeros(len(angulos))

        voltages = np.zeros(len(angulos))

        pot_reativas = np.zeros(len(qg))


        for i in range(len(angulos)):

            thetas[i]=angulos[i][0]

            voltages[i]=tensoes[i][0]



        sep14.res_bus['vm_pu'] = voltages


        sep14.res_ext_grid['p_mw'] = pgs[0][0]*100
        
        u=0
        for bus in sep14.gen['bus'].to_numpy():
            
            sep14.gen['p_mw'][u] = pgs[bus][0]*100
            u=u+1


        sep14.res_ext_grid['q_mvar'] = qg[0][0]*100

        sep14.shunt['q_mvar'] = s*100
        
        gbest = [tensao,t,s,np.array([0,0,0,0,0,0])]
        
        sep14.trafo['tap_pos'][pd.isnull(sep14.trafo['tap_step_percent'])]=np.nan
        
        sep14.bus['max_vm_pu'] = 1.05

        sep14.bus['min_vm_pu'] = 0.95
        sep14.gen['min_vm_pu'] = 0.95

        sep14.gen['max_vm_pu'] = 1.05
        
        sep14.ext_grid['min_vm_pu'] = 0.95

        sep14.ext_grid['max_vm_pu'] = 1.05
        

    
    return gbest, sep14, tensao, t, s, pgs, sep.options.OBJFCNVAL 

#     return sep.options.OBJFCNVAL

# In[12]:


# solucao_continua,sep_atualizado,tensao,t,s, pgs = automatiza_ipm_c(sep_118, verbose=True, travado=False)


# In[13]:


# gen = []
# for bus in sep_atualizado.gen['bus'].to_numpy():
#     gen.append(sep_atualizado.gen['p_mw'][sep_atualizado.gen['bus']==bus].to_numpy()[0]/100)

# initial = np.concatenate((solucao_continua[:-6],gen,np.array([0,0,0,0,0,0,0])))

# initial


# In[14]:


# gbest=np.copy(solucao_continua)

# gbest[len(tensao):len(t)+len(tensao)] = discreto_tap(gbest,len(t),len(tensao),len(s),sep_atualizado)

# gbest[len(t)+len(tensao):len(t)+len(tensao)+len(s)] = discreto_bshunt(gbest,len(t),len(tensao),len(s),sep_atualizado)

# validacao(sep_atualizado,gbest)

# # solucao_continua,sep_atualizado,tensao,t,s, pgs = automatiza_ipm_c(sep_atualizado, verbose=True, travado=True)


# In[16]:


# solucao_continua2,sep_atualizado2,tensao,t,s,pgs = automatiza_ipm_c(sep_atualizado, verbose=True, travado=True)

