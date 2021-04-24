% parâmetros de entrada

% caso de teste 1

%a = [0.0080 0.0096];

%b = [8 6.4];

%c = [0 0];

%demanda = [250 350 500 700 900 1100 1175 1250];

%pmin = [100 100];

%pmax = [625 625];

% caso de teste 2

a = [0.0080 0.0080 0.0096 0.0096];

b = [8 8 6.4 6.4];

c = [0 0 0 0];

demanda = [250 350 500 700 900 1100 1175 1250];

pmin = [50 50 50 50];

pmax = [625/2 625/2 625/2 625/2];


% calcular at, bt e alfa

pg = zeros(size(a));

PotenciasGeradas = [];

alfat = [];

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

PotenciasGeradas = [PotenciasGeradas;pg];

end


disp('Potências Geradas:')
PotenciasGeradas

% plota curva da potência dos geradores
figure
for g=1:length(pg)
plot(demanda,PotenciasGeradas(:,g),'-x')
hold on
end
hold off
grid on
grid minor
title('Gráfico de Potência Gerada x Demanda')
ylabel('Potência Gerada por Cada Gerador [MW]')
xlabel('Demanda [MW]')


% plota curva do custo marginal incremental
figure
plot(demanda,alfat,'-x')
grid on
grid minor
title('Curva de Custo Marginal Incremental Por Demanda')
ylabel('Custo Marginal Incremental[$/MWh]')
xlabel('Demanda [MW]')

% plota curva do custo

% calcula custo
custo=[];

for dem=1:length(demanda)
    custo(dem) = sum((a/2).*PotenciasGeradas(dem,:).*PotenciasGeradas(dem,:)+b.*PotenciasGeradas(dem,:)+c);
end

disp('Custo Marginal:')
alfat

disp('Custo:')
custo

figure
plot(demanda,custo,'-x')
grid on
grid minor
title('Curva de Custo Por Demanda')
ylabel('Custo [$/h]')
xlabel('Demanda [MW]')
