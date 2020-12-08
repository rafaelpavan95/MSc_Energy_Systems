% Matriz de Imped�ncias S�rie

Zkm = [0 (0.2 + 0.4i) (0.2 + 0.4i);(0.2 +0.4i) 0 (0.1 + 0.2i); (0.2+0.4i) (0.1+0.2i) 0 ];

% Vetor de Suscept�ncias Shunt

Bsh = imag([0, 0.01i, 0.01i; 0.01i, 0, 0.01i; 0.01i, 0.01i,0]);

% Calculando a Matriz de Admit�ncias

Ykm = 1./Zkm;

% Separando o Vetor de Admit�ncias em Dois Vetores: Condut�ncias [Gkm] e
% Suscept�ncias [Ykm]

Gkm = real(Ykm);

Bkm = imag(Ykm);

% Criando as Vari�veis Simb�licas (Inc�gnitas)

syms V2 fi2 fi3 Pg1 Qg1 Qg3;

incognitas = [V2, fi2, fi3];

% Criando Vetor de �ngulos

fi = [0, fi2, fi3];

% Criando Vetor de Tens�es

V = [1.05, V2, 1];

% Criando Vetor de Pot�ncias Geradas

Pg = [Pg1, 0, 0.3];
Qg = [Qg1, 0, Qg3];

% Criando Vetor de Pot�ncias Consumidas

Pc = [0, 0.5, 0];
Qc = [0, 0.2, 0];

%Define N�mero Total de Barras

nbarras = [3];

% Define Quais s�o Barras de Gera��o

bger = [3];

% Define Quais s�o Barras de Carga

bcarga = [2];

% Define Qual a Barra de Refer�ncia

bref= [1];

% Inicializa a vari�vel que vai armazenar as equa��oes

eqqc = sym(zeros(1,numel(bcarga)));
eqpc = sym(zeros(1,numel(bcarga)));
eqpg = sym(zeros(1,numel(bger)));

% Inicializa Vari�veis de Controle dos Loops

k=1;
j=1;
l=1;
m=1;

% Equa��es Barras de Carga

for j=1:numel(bcarga)  
    
    eqqc(j) = Qg(bcarga(j))-Qc(bcarga(j));
    eqpc(j) = Pg(bcarga(j))-Pc(bcarga(j));
    
       while (k<nbarras)
           
            if k == bcarga(j)
                k=k+1;
            end
            
            eqpc(j) = eqpc(j) + ((-V(bcarga(j))*V(bcarga(j))*(Gkm(bcarga(j),k)))+(V(bcarga(j))*V(k)*Gkm(bcarga(j),k)*cos(fi(bcarga(j))-fi(k)))+(V(bcarga(j))*V(k)*Bkm(bcarga(j),k)*sin(fi(bcarga(j))-fi(k))));
            eqqc(j) = eqqc(j) -((-V(bcarga(j))*V(bcarga(j))*(Bkm(bcarga(j),k)+Bsh(bcarga(j),k)))+(V(bcarga(j))*V(k)*Bkm(bcarga(j),k)*cos(fi(bcarga(j))-fi(k)))-(V(bcarga(j))*V(k)*Gkm(bcarga(j),k)*sin(fi(bcarga(j))-fi(k))));          
            k=k+1;
       
       end
    
       
end


% Equa��es Barras de Gera��o

for l=1:numel(bger)
    
    eqpg(l) = Pg(bger(l))-Pc(bger(l));
            
      while (m<nbarras)
           
            if m == bger(l)
                 m=m+1;
            end
            
        eqpg(l) = eqpg(l) -((V(bger(l))*V(bger(l))*(Gkm(bger(l),m)))-(V(bger(l))*V(m)*Gkm(bger(l),m)*cos(((fi(bger(l)))-(fi(m)))))-(V(bger(l))*V(m)*Bkm(bger(l),m)*sin((fi(bger(l))-fi(m)))));
        m=m+1;

      end    

end

% Salva as Equa��es em um Vetor 

equations = [eqpc, eqqc, eqpg]


% Script para Resolver o Sistema N�o-Linear de N Equa��es pelo M�todo de Newton

% Par�metros necess�rios: Vetor de Equa��es, Vetor de Valores Iniciais e
% Vetor de Inc�gnitas

equations = [eqpc, eqqc, eqpg];
incognitas = [V2, fi2, fi3];
vetordevalores = [1, 1, 1];

% Inicializa Vari�veis de Controle dos Loops

iteracoes = 1;
t=1;
u=1;

% Realiza C�lculo da Matriz Jacobiana com as Equa��es

for t = 1:length(equations)
 
       for u = 1:length(equations)
     
          J(u,t) = diff(equations(u),incognitas(t));
    
       end
end

% Substitui o vetor de valores iniciais nas equa��es

F = vpa(subs(equations, incognitas, vetordevalores));

% Calcula a Norma Infinito

error(iteracoes)=max(abs(F));

% Enquanto a Norma Infinito for Superior ao Crit�rio de Parada
% Estabelecido, Realiza o Procedimento Abaixo

while(error(iteracoes)>0.00001)
    
    iteracoes=iteracoes+1;

    % Calcula a Matriz Jacobiana e Sua Inversa, e M�ltiplica por -1
    
    Jsub = -1*vpa(subs(J, incognitas, vetordevalores));
    IJsub = inv(Jsub);
    
    % Calcula o Valor da Norma Infinito
    
    F = vpa(subs(equations, incognitas, vetordevalores));
    error(iteracoes)=max(abs(F));
    
    % Calcula o Vetor de Acr�scimo
    
    dx=IJsub*transpose(F);
    
    % Atualiza o Vetor de Valores Inicial
    vetordevalores=vetordevalores+transpose(dx); 
    resultado = vetordevalores;
    
end
 
% Realiza a Plotagem do Gr�fico com o Valor da Norma Infinito para Cada
% Itera��o

plot(1:iteracoes,error);
 
title('M�todo de Newton - Sistema N�o-Linear - ||F(x)|| - Norma Infinito');
xlabel('N�mero da Itera��o');
ylabel('Valor da Norma');
grid on;
grid minor;

% Solu��o do Sistema

v2=resultado(1)
fi2=resultado(2)
fi3=resultado(3)

% Script para Calcular o Fluxo de Pot�ncia Ativa e Reativa Entre as Barras
% Necessita dos Vetores de Tens�es e �ngulos de Fase como Par�metros

VN = subs(V, [incognitas(1)],[resultado(1)])
fin = subs(fi, [incognitas(2),incognitas(3)],[resultado(2),resultado(3)])

y = 1;
c = 1;
d = 1;

for y=1:nbarras
            
    for d=1:nbarras
     
         if (y == d)
            
            Peq(y,d)=0.0;
            Qeq(y,d)=0.0;
 
         end
     
     Peq(y,d) = (VN(y)*VN(y))*Gkm(y,d)- VN(y)*VN(d)*Gkm(y,d)*cos(fin(y)-fin(d)) - VN(y)*VN(d)*Bkm(y,d)*sin(fin(y)-fin(d));
     Qeq(y,d) = -(VN(y)^2)*(Bkm(y,d)+Bsh(y,d)) + VN(y)*VN(d)*Bkm(y,d)*cos(fin(y)-fin(d)) - VN(y)*VN(d)*Gkm(y,d)*sin(fin(y)-fin(d));
     d=d+1;
     
    end
      
      y=y+1;
    
end

fluxoentrelinhas = Peq+Qeq*i;
fluxo12=fluxoentrelinhas(1,2)
fluxo21=fluxoentrelinhas(2,1)
fluxo13=fluxoentrelinhas(1,3)
fluxo31=fluxoentrelinhas(3,1)
fluxo32=fluxoentrelinhas(3,2)
fluxo23=fluxoentrelinhas(2,3)



% Script que Calcula as Perdas nas Linhas

n=1;
p=1;

for n=1:nbarras
            
    for p=1:nbarras
     
     if (n == p)
     
         Peql(n,p)=0.0;
         Qeql(n,p)=0.0;
             
     end
    
     Peql(n,p) = Peq(n,p)+Peq(p,n);
     Qeql(n,p) = Qeq(n,p)+Qeq(p,n);
     p=p+1;

    end
      
      n=n+1;
    
end

perdasnalinha=Peql+Qeql*i;

% Perdas nas Linhas

perdas12 = perdasnalinha(1,2)
perdas13 = perdasnalinha(1,3)
perdas32 = perdasnalinha(3,2)

perdastotais = perdas12+perdas13+perdas32

% Calcula as Pot�ncias Ativas e Reativas Geradas na Barra 1, e a Pot�ncia
% Reativa Gerada na Barra 3

PG1=vpa(-((Pg(2)+Pg(3)) - (Pc(1)+Pc(2)+Pc(3)) - real((perdasnalinha(1,2)+perdasnalinha(1,3)+perdasnalinha(3,2)))))

QG1=vpa(-(-Qc(1)-Qeq(1,2)-Qeq(1,3)))

QG3=vpa(-(-Qc(3)-Qeq(3,1)-Qeq(3,2)))

% Novos Vetores de Pot�ncias Geradas nas Barras

PgN = subs(Pg,Pg(1),PG1);
QgN = subs(Qg,[Qg(1),Qg(3)],[QG1,QG3]);

% C�lculo do Balan�o de Pot�ncia Aparente nas Barras

SbalancoB3 = round(PgN(3)+QgN(3)*i-(Pc(3)+Qc(3)*i+Peq(3,1)+Peq(3,2)+Qeq(3,1)*i+Qeq(3,2)*i))

SbalancoB2 = round(PgN(2)+QgN(2)*i-(Pc(2)+Qc(2)*i+Peq(2,1)+Peq(2,3)+Qeq(2,1)*i+Qeq(2,3)*i))

SbalancoB1 = round(PgN(1)+QgN(1)*i-(Pc(1)+Qc(1)*i+Peq(1,2)+Peq(1,3)+Qeq(1,2)*i+Qeq(1,3)*i))



