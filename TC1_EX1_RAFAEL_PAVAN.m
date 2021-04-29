clear all
clc


% Exercicio Exemplo:

% Trabalho C1 - O Metodo Primal-Afim de Pontos Interiores
% Aluno: Rafael Pavan
% Programa de Pos-Graduacao em Engenharia Eletrica - UNESP Bauru
% Mestrando em Engenharia Eletrica

% Problema Exemplo:
% minimizar -2*x1 + x2
% sujeito a:
% x1 - x2 + x3 = 15
% x2 + x4 = 15
% x1,x2,x3,x4 >= 0

% Ponto Inicial X0:

f = [];

x0 = [10,6,11,8,9];

syms x1 x2 x3 x4 x5


eq = 101*x1^2 -2*x1 + 100*x2^4 -200*x1*x2^2 + 1 + 0*x3 + 0*x4 + 0*x5

gradiente=gradient(eq,[x1,x2,x3,x4,x5])

hessiana = hessian(eq,[x1,x2,x3,x4,x5])

x1 = x0(1)
x2 = x0(2)
x3 = x0(3)
x4 = x0(4)
x5 = x0(5)

gradientevpa = vpa(subs(gradiente))

hessianavpa = vpa(subs(hessiana))


% Inicia Valor de Alfa:

alfa = 0.9995;

% Define a Matriz A:

A = [1, -1, 1, 0, 0; 0, -1, 0, 1, 0; 0, 1, 0, 0, 1];

% Define vetor c:

c = [-2, 1, 0, 0];

% Define Vetor b:

b = [15, 15];

% Define valor de tolerancia episolon:

episolon = 0.01;

% Define matriz Xk diagonal:

Xk = zeros(length(x0),length(x0));

% Inicializa vetor de e [1,1,1....,1]

e = ones(length(x0),1);

condicao_de_parada = 0;

iteracoes = 0;

while (condicao_de_parada == 0)
    
% Inicializa matriz Xk

    for linha=1:length(x0)

        coluna = linha;

        Xk(linha,coluna) = x0(coluna);

    end

    % Calcula vetor estimativa dual Wk:

    Wk = inv(A*Xk*Xk*transpose(A))*A*Xk*Xk*transpose(c);

    % Calcula vetor custo relativo Rk:

    Rk = transpose(c) - transpose(A)*Wk;

    Rk = round(Rk,10);
    
    if length(Rk(Rk>=0)) == length(Rk)
            
        % E

        if transpose(e)*Xk*Rk <= episolon

            condicao_de_parada=1;

            display("1. Condicao de Parada Atingida")

        end
    end

    % OU
    
    
    if length(x0(x0>=0)) == length(x0)
        
        % E
        
        if length(Rk(Rk>=0)) == length(Rk)
            
            % E 
            
            [n, m] = size(Xk);
            
            valor = m-n;
            
            if length(x0(x0==0)) == valor
                
                condicao_de_parada = 1;
                display("2. Condicao de Parada Atingida")
                
            end
        end
    end


    % Calcula a direção dk:

    dk = -Xk*Rk;

    % Verifica Ilimitação


    if length(dk(dk<episolon)) == 0

        display("3. Problema Ilimitado. Algoritmo Interrompido")

    end

    
    if length(dk(dk==0)) == length(dk)

        display("4. Problema Resolvido. Otimo Primal Encontrado")    
        condicao_de_parada=1
    end

    % Calcula comprimento do passo alfak:

    lista = [];

    for i=1:length(dk)


        if dk(i) < 0


            lista = [lista, (alfa/(-dk(i)))];

        end

    end

    alfak = min(lista);

    % Calcula yk:


    yk = e + alfak*dk;

    xi = Xk*yk;

    x0 = xi;
    
    xfinal = x0;
    
    iteracoes=iteracoes+1;
    
   % Calcula funcao objetivo
    f = [f,-2*x0(1) + x0(2)];
    
    if iteracoes == 50000
        
        condicao_de_parada = 1
    end   
    

end

display("Solucao: ")
display(xfinal)

% Plota funcao objetivo 

plot(f(1:5))

title('Curva de Convergencia')
xlabel('Iteracao')
ylabel('Funcao Objetivo')
grid minor