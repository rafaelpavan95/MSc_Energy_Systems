clc
x0 = [15,12,12,3,10];

syms x1 x2 x3 x4 x5


eq = 101*x1^2 -2*x1 + 100*x2^4 -200*x1*x2^2 + 1 + 0*x3 + 0*x4 + 0*x5;

gradiente=gradient(eq,[x1,x2,x3,x4,x5]);

%vars = [x1 x2 x3 x4 x5];
%solucao = solve(gradiente==0,vars);

%solucao.x2
%solucao.x3
%solucao.x4
%solucao.x5

hessiana = hessian(eq,[x1,x2,x3,x4,x5]);

x1 = x0(1);
x2 = x0(2);
x3 = x0(3);
x4 = x0(4);x1
x5 = x0(5);

gradientevpa = vpa(subs(gradiente));

hessianavpa = vpa(subs(hessiana));


% Inicia Valor de Alfa:

alfa = 0.9995;

% Define a Matriz A:

A = [1, -1, 1, 0, 0; 0, 1, 0, 1, 0; 0, -1, 0, 0, 1];

% Define Vetor b:

b = [15, 15,-2];

% Define valor de tolerancia episolon:

episolon = 0.01;

% Define matriz Xk diagonal:

Xk = zeros(length(x0),length(x0));

% Inicializa vetor de e [1,1,1....,1]

e = ones(length(x0),1);

condicao_de_parada = 0;

iteracoes = 0;


while (condicao_de_parada==0)

    % pega o vetor x0 e coloca na diagonal da matriz Xk
    
    for linha=1:length(x0)

        coluna = linha;

        Xk(linha,coluna) = x0(coluna);

    end

    % Calculo do vetor estimativa dual

    Hk = inv((hessianavpa + inv(Xk)*inv(Xk)));

    wk = inv(A*Hk*transpose(A)) * A*Hk*(gradientevpa)

    % Cálculo do vetor custo relativo

    sk = (gradientevpa) - transpose(A)*wk
    
    norm(sk)

    episolon3 = 0.0001;



    % Calculo da Direcao de Translacao

    dxk = -Hk*sk

    alfa = 0.9995;

    % Calculo do comprimento do passo:

    alfak2 = -transpose(dxk)*(gradientevpa)/((transpose(dxk)*hessianavpa*dxk))

    lista = [];

    for i=1:length(dxk)

        if dxk(i) < 0

            lista = [lista, -alfa*x0(i)/dxk(i)];
        end
    end

    alfak1 = min(lista)

    alfak = alfa*min(alfak1,alfak2)

    % Nova solucao:

    x0 = x0 + transpose(alfak*dxk)
    

    if iteracoes == 2
        
        condicao_de_parada=1;
    end
    
gradientevpa = vpa(subs(gradiente));

hessianavpa = vpa(subs(hessiana));

    iteracoes=iteracoes+1;

end


%a = 17:0.005:20;
%b = 3:0.005:5;
%[X1,X2] = meshgrid(a,b);
%Z = 100*(X1-X2.^2).^2 + (1 - X1).^2;

close all

%figure

%surf(X1,X2,Z)

%hold on

%contour(X1,X2,Z)

%figure
%contour(X1,X2,Z)
%hold on
%plot(x0(1),x0(2),'x')


    

