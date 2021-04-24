clc
clear all
format long 
 
% Passo 1 - Especifica Demanda
%demanda = [250 350 500 700 900 1100 1175 1250];
 
demanda = [250 350 500 700 900 1100 1175 1250];

% ESPECIFIQUE QUAL TESTE DESEJA REALIZAR:
% TESTE = 1 -> SIMULA CASO COM 2 GERADORES
% TESTE = 2 -> SIMULA CASO COM 4 GERADORES
 
TESTE = 1;
 
% parâmetros de entrada
 
% caso de teste 1
 
if TESTE==2
    
a = [0.0080 0.0096 0.0080 0.0096];
 
b = [8 6.4 8 6.4];
 
c = [0 0];
 
pmin = [50 50 50 50];
 
pmax = [625/2 625/2 625/2 625/2];
 
B = [8.383183 -0.049448 8.383183 -0.049448 0.375082; -0.049448 5.9635668 -0.049448 5.9635668 0.194971; 8.383183 -0.049448 8.383183 -0.049448 0.375082; -0.049448 5.9635668 -0.049448 5.9635668 0.194971; 0.375082 0.194971 0.375082 0.194971 0.090121]*0.001;
 
end
 
 
if TESTE==1
   
 
pmin = [100 100];
 
pmax = [625 625];
 
a = [0.0080 0.0096];
 
b = [8 6.4];
 
c = [0 0];
 
B = [8.383183 -0.049448 0.375082; -0.049448 5.9635668 0.194971; 0.375082 0.194971 0.090121]*0.001;
 
end
 
% Passo 2 - Calcula Preço Inicial
 
% calcular at, bt e alfa
 
pg = zeros(size(a));
 
PotenciasGeradas = [];
 
alfat = [];
 
Pbase = 100;
 
cip=[];
perdasf=[];
cig=[];
potenciasgera=[];
 
for i=1:length(demanda)
    
  at=0;
  for j=1:length(a)
      at=at+(1/a(j));
  end
  at=1/(at);
  bt=0;
  for j=1:length(a)
      bt=bt+(b(j)/a(j));
  end
  
  bt=bt*at;
  
  alfa = at*demanda(i)+bt;     
 
  for k=1:length(pg)
     pg(k) = (alfa-b(k))/a(k);
  end
 
limitemin =[];
limitemax =[];
normal = [];
 
% verifica quais geradores tiveram limites ultrapassados
 
for k=1:length(pg)
 
    if pg(k) < pmin(k)
        limitemin = [limitemin, k];
    end
          
    if  pg(k) > pmax(k)
        limitemax = [limitemax, k];
    end
     
    if (pg(k) > pmin(k)) && (pg(k) < pmax(k))
         normal = [normal, k];
    end
 
end
 
 
pacomulada = 0;
 
for u=1:length(limitemin)
    pacomulada= pacomulada + pmin(limitemin(u));
end
 
for u=1:length(limitemax)
    pacomulada= pacomulada + pmax(limitemax(u));
end
 
pgt = demanda(i)-pacomulada;
 
% recalcula at, bt e alfa para as normais
 
for j=1:length(normal)
      at=at+(1/a(normal(j)));
end
 
at=1/(at);
bt=0;
  
for j=1:length(normal)
    bt=bt+(b(normal(j)))/a(normal(j));
end
  
bt=bt*at;
 
alfa = at*pgt+bt;
alfat = [alfat, alfa];
% recalcula as potências com o novo alfa
 
for k=1:length(normal)
     pg(normal(k)) = (alfa-b(normal(k)))/a(normal(k));
end
 
% substitui potências pelos limites dos que ultrapassaram
 
for k=1:length(limitemin)
     pg(limitemin(k)) = pmin(limitemin(k));
end
 
for k=1:length(limitemax)
     pg(limitemax(k)) = pmax(limitemax(k));
end
bdim = size(B);
an=a*Pbase*Pbase;
bn=b*Pbase;
alfan=alfa*Pbase;
matriz = [];
vetor = [];
 
demanda2 = demanda(i);
 
% Passo 3 - Calcula Matriz e Potências para o Preço Inicial
 
for linha=1:length(a)
 
    for coluna=1:length(a)
 
        if linha==coluna
        
            matriz(linha,coluna) = 2*B(linha,coluna)+an(linha)/alfan;
    
        else
            matriz(linha,coluna) = 2*B(linha,coluna);
        end
    end
end
 
vetor = 1-(2*B(1:end-1,bdim(2)))-(transpose(bn)/alfan);
 
pgeradas = [];
 
pgeradas = transpose(matriz\vetor);

 
if TESTE==1

    if pgeradas(1)<1
    
        calcula=0;
        calcula=1-pgeradas(1);
        pgeradas(1)=1;
        pgeradas(2)=pgeradas(2)-calcula;
    
    end

     if pgeradas(2)<1
    
        calcula=0;
        calcula=1-pgeradas(2);
        pgeradas(2)=1;
        pgeradas(1)=pgeradas(1)-calcula;
    
     end
    
    if pgeradas(2)>6.25
    
        calcula=0;
        calcula=pgeradas(2)-6.25;
        pgeradas(2)=6.25;
        pgeradas(1)=pgeradas(1)+calcula;
    
     end
     
    if pgeradas(1)>6.25
    
        calcula=0;
        calcula=pgeradas(1)-6.25;
        pgeradas(1)=6.25;
        pgeradas(2)=pgeradas(2)+calcula;
    
     end
end

        

if TESTE==2
    for i=1:length(pgeradas)
    if pgeradas(i)<0.001
        pgeradas(i)=pmin(i)/Pbase;
    end
    end
    
     if pgeradas(4)>3.25
        pgeradas(4)=pmax(4)/Pbase;
    end
    if pgeradas(2)>3.25
        pgeradas(2)=pmax(2)/Pbase;
    end
    
end

        
% Passo 4 - Cálculo das Perdas
 
pp = [pgeradas 1];
 
perdas = pp*B*transpose(pp);
 
% Passo 5 - Cálculo do mismatch de potência ativa e Critério de Parada
 
DELTAP = sum(pgeradas) - demanda2/Pbase - perdas;
 
pgeradas = [zeros(size(pgeradas)); [pgeradas]];
 
p = [0 alfan];
 
perdas = [0 perdas];
 
contador=2;
 
while abs(DELTAP)>0.000001
    
    deltap = ((p(contador)-p(contador-1))/(sum(pgeradas(contador,1:end))-sum(pgeradas(contador-1,1:end))))*((demanda2/Pbase)+perdas(contador)-sum(pgeradas(contador,1:end)));
 
    novo_p = p(contador)+deltap;
 
    for linha=1:length(a)
 
        for coluna=1:length(a)
 
            if linha==coluna
        
                matriz(linha,coluna) = 2*B(linha,coluna)+an(linha)/novo_p;
    
            else
                matriz(linha,coluna) = 2*B(linha,coluna);
            end
        end
    end
 
    matriz3=matriz;
    vetor = 1-(2*B(1:end-1,bdim(2)))-(transpose(bn)/novo_p);
 
    pgr = transpose(matriz\vetor);
 
    if TESTE==2
    for i=1:length(pgr)
    if pgr(i)<0.001
        pgr(i)=pmin(i)/Pbase;
    end
    
    end
    
    
    if pgr(4)>3.25
        pgr(4)=pmax(4)/Pbase;
    end
    if pgr(2)>3.25
        pgr(2)=pmax(2)/Pbase;
    end
    
    end
if TESTE==1

    if pgr(1)<1
    
        calcula=0;
        calcula=1-pgr(1);
        pgr(1)=1;
        pgr(2)=pgr(2)-calcula;
    
    end
    
    if pgr(2)<1
    
        calcula=0;
        calcula=1-pgr(2);
        pgr(2)=1;
        pgr(1)=pgr(1)-calcula;
    
    end
    
    if pgr(2)>6.25
    
        calcula=0;
        calcula=pgr(2)-6.25;
        pgr(2)=6.25;
        pgr(1)=pgr(1)+calcula;
    
     end
     
    if pgr(1)>6.25
    
        calcula=0;
        calcula=pgr(1)-6.25;
        pgr(1)=6.25;
        pgr(2)=pgr(2)+calcula;
    
    end
    
end


    pp = [pgr 1];
 
    perds = pp*B*transpose(pp);
 
    DELTAP = sum(pgr) - demanda2/Pbase - perds;
 
    pgeradas = [[pgeradas]; pgr];
 
    p = [p novo_p];
 
    perdas = [perdas perds];
 
    contador = contador + 1;
 
end

pprov = [pgr,1];
 
perdasincrementais = [];
 
for t=1:length(pgr)
 
    bprov = B(t,1:end);
    perdasincrementais(t) = sum(2*pprov.*bprov);
    
end
 
 
fatordepenalidade=[];
 
for t=1:length(pgr)
 
fatordepenalidade(t) = (1/(1-perdasincrementais(t)));
    
end
 
custosincrementaisdeger = [];
 
for t=1:length(pgr)
 
custosincrementaisdeger(t) = a(t)*pgr(t)*Pbase+b(t);
    
end
 
custoincrementaldepotencia = custosincrementaisdeger(1)*fatordepenalidade(1);
 
 
display('Para a Demanda [pu] de:')
display(demanda2/Pbase)
display('Potências Geradas [pu] de:')
display(pgr)
display('Perdas [pu] de:')
display(perds)
display('Custo Incremental de Potência [$/MW]:')
display(custoincrementaldepotencia)
display('Fatores de Penalidade:')
display(fatordepenalidade)
display('Custos Incrementais de Geração [$/MWh]:')
display(custosincrementaisdeger)
 
cip = [cip,custoincrementaldepotencia];
perdasf = [perdasf,perds*Pbase]; 
cig = [cig;custosincrementaisdeger];
potenciasgera=[potenciasgera; pgr];
 
end
 
figure
 
subplot(2,2,1)
plot(demanda,perdasf);
title('Gráfico de Perdas por Demanda')
xlabel('Demanda [MW]')
ylabel('Perdas [MW]')
grid on
grid minor
 
subplot(2,2,2)
plot(demanda,cip);
title('Gráfico de Custo Incremental de Potência por Demanda')
xlabel('Demanda [MW]')
ylabel('CIP [$/MW]')
ylim([0 15])
grid on
grid minor
 
cigdim=size(cig);
 
subplot(2,2,3)
 
for ay=1:cigdim(2)
    
    plot(demanda,cig(1:end,ay))
    hold on
 
end
 
title('Gráfico de Custo Incremental de Geração por Demanda')
xlabel('Demanda [MW]')
ylabel('CIG [$/MWh]')
grid on
grid minor
 
 
if TESTE==2
legend('Gerador 1','Gerador 2','Gerador 3','Gerador 4')
end
 
if TESTE==1
legend('Gerador 1','Gerador 2')
end
 
subplot(2,2,4)
 
for ay=1:cigdim(2)
    
    plot(demanda,transpose(potenciasgera(1:end,ay))*Pbase)
    hold on
 
end
 
title('Gráfico de Potência Gerada por Demanda Para Cada Gerador')
xlabel('Demanda [MW]')
ylabel('Potência Gerada [MW]')
grid on
grid minor
 
if TESTE==2
legend('Gerador 1','Gerador 2','Gerador 3','Gerador 4')
end
 
if TESTE==1
legend('Gerador 1','Gerador 2')
end
