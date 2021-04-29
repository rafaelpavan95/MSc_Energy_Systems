clear all
clc

% Inicia Valor de Alfa:

alfa = 0.9995;

% Define a Matriz A:

A = [1 1 0 ; 0 1 1];

% Define vetor c:

c = [1 2 -3];

% Define Vetor b:

b = [5 10];


% Define Matriz Q

Q = [4 0 0; 0 6 0; 0 0 10];

x0 = -inv(Q)*transpose(c);

if (A*x0) == b
    
    if length(x0(x0>=0)) == length(x0)
        
    display("Pare. Solucao otima encontrada")
    condicao_de_parada=1;

    end
end
 
condicao_de_parada = 0;

iteracoes = 0;

x0 = [1.6,3.4,6.6];

Xk = zeros(length(x0),length(x0));

while (condicao_de_parada==0)

    % pega o vetor x0 e coloca na diagonal da matriz Xk
    
    for linha=1:length(x0)

        coluna = linha;

        Xk(linha,coluna) = x0(coluna);

    end

    % Calculo do vetor estimativa dual

    Hk = inv((Q + inv(Xk)*inv(Xk)));

    wk = inv(A*Hk*transpose(A)) * A*Hk*(Q*transpose(x0)+transpose(c));

    % Cálculo do vetor custo relativo

    sk = (Q*transpose(x0) + transpose(c)) - transpose(A)*wk;

    episolon3 = 0.0001;

    % Teste de Otimalidade

    if length(x0(x0>0)) == length(x0)

        if length(sk(round(sk,2)>0)) == length(sk)

            if x0*sk < episolon3

                display('Solucao Otima Encontrada')
                condicao_de_parada=1;
                
                    % Teste de Factibilidade

                episolon1 = 0.00001;

                episolon2 = 0.1;


                if length(x0(x0>=0)) == length(x0)

                            if norm(A*transpose(x0) - transpose(b))/(norm(b)+1) < episolon1

                        display('Factibilidade Primal Atingida')

                    end
                end

                if length(sk(round(sk,2)>=0)) == length(sk)

                    if  norm(sk)/(norm(Q*transpose(x0)+transpose(c))+1) < episolon2

                        display('Factibilidade Dual Atingida')

                    end
                end


            end
            
        end
    end


    % Calculo da Direcao de Translacao

    dxk = -Hk*sk;

    % Teste de Ilimitariedade
    
    if length(dxk(dxk>0)) == length(dxk)

        display('Problema Ilimitado')
        condicao_de_parada=1;

    end
     
    % Teste de Otimalidade dxk
    
    if length(dxk(dxk<episolon3)) == length(dxk)

        display('Solucao Otima Encontrada')
        condicao_de_parada=1;
        

            % Teste de Factibilidade

            episolon1 = 0.00001;

            episolon2 = 0.3;


            if length(x0(x0>=0)) == length(x0)

                        if norm(A*transpose(x0) - transpose(b))/(norm(b)+1) < episolon1

                    display('Factibilidade Primal Atingida')

                end
            end

            if length(sk(round(sk,2)>=0)) == length(sk)

                if  norm(sk)/(norm(Q*transpose(x0)+transpose(c))+1) < episolon2

                    display('Factibilidade Dual Atingida')

                end
            end

    end


    alfa = 0.9995;

    % Calculo do comprimento do passo:

    alfak2 = -transpose(dxk)*(Q*transpose(x0)+transpose(c))/((transpose(dxk)*Q*dxk));

    lista = [];

    for i=1:length(dxk)

        if dxk(i) < 0

            lista = [lista, -alfa*x0(i)/dxk(i)];
        end
    end

    alfak1 = min(lista);

    alfak = alfa*min(alfak1,alfak2);

    if length(lista) == 0

        alfak = alfak2;

    end

    % Nova solucao:

    x0 = x0 + transpose(alfak*dxk);
    

    if iteracoes == 5
        
        condicao_de_parada=1;
    end
    
    iteracoes=iteracoes+1;

end

display('Solução Ótima')
x_otimo=x0