clc
close all
clear all

% Rafael Pavan
% TC3 -Implementa��o do M�todo Primal-Afim Trajet�ria Central para PPQ
% Disciplina: Otimiza��o e M�todo de Pontos Interiores
% Professor: Antonio Roberto Balbo

% Exemplo 1:
% min 2x1^2+3x2^2+5x3^2+x1+2x2-3x3
% sujeito a x1+x2=5; x2+x3=10; xi>=0

%Entrada de par�metros

c = [1;2;-3];

A = [1 1 0;0 1 1];

b = [5;10];

x0 = [1.6; 3.4; 6.6];

k = 1;

u0 = 1.5;

alpha = 0.9995;

Q = [4 0 0;0 6 0;0 0 10];

epsilon = 10^-3;

e = [1;1;1];

%Testando se a solu��o otima esta no interior da regiao factivel

x_init = -inv(Q)*c;

if A*x_init == b
    if length(x_init(x_init<0)) <= length(xinicial)
        display('A solu��o �tima �:')
        xinicial
    end
    else
        display('A solu��o inicial n�o est� contida dentro da regi�o fact�vel.')
end

k = 0;

condicao_de_parada = 0;
   
xk = x0;

while(condicao_de_parada == 0)  
        
    % Cria Xk
    
    for linha=1:length(xk)
        
        Xk(linha,linha) = x0(linha);
        
    end
    
    %Calcula o vetor estimativa dual w1k
    
    Hk = inv(Q+(u0+1)*inv(Xk)^2);
    
    wk = inv(A*Hk*transpose(A))*A*Hk*(Q*xk+c-u0*inv(Xk)*e);
    
    %Calcula o vetor custo relativo sk
    
    sk = (Q*xk+c-u0*inv(Xk)*e)-transpose(A)*wk;
    
    %Teste de otimalidade
    
    if sk>0
        if transpose(x0)*sk <= epsilon
            display('A solu��o �tima �')
            xk
            sk
            break
        end
    end
    
    %Calcula a dire��o de transla��o dxk
    
    dxk = -Hk*sk;
    
    %Teste de ilimitariedade
    
    if length(dxk(dxk>0)) == length(dxk)
       display('O problema � ilimitado.')
       break
    end
       
    if length(dxk(dxk<10e-5)) == length(dxk)
       display('A Solu��o �tima �: ')
       xk
       xk
       break
    end
    
        
    %Calcula o comprimento do passo alfak
    alfa = [];
    
    for i=1:length(dxk)
    
        if dxk(i) < 0
        
            alfa = [alfa, alpha*xk(i)/(-dxk(i))];
       
        end
        
    end
    
        
    alfak1 = min(alfa);
    
    alfak2 = (- transpose(dxk)*(Q*x0+c))/(transpose(dxk)*Q*dxk);
    
    if alfak2 > 0
        alfak =  alpha * min(alfak1,alfak2);
    else 
        alfak = alpha * min(alfak1,1);
    end
    
    %Atualizando dados
    
    u0=u0/2;
    
    xk=xk+alfak*dxk
    
    k=k+1;
  
   %Limite de itera��es
    if k==2
        condicao_de_parada=1;
        break
    end
       
end

display('solu��o �tima primal:')

xk

display('solu��o �tima dual:')

sk

funcaoobjetivo=transpose(xk)*Q*xk/2+transpose(c)*xk
    
