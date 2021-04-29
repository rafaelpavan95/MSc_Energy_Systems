clear all
clc


% Exercicio Exemplo:

% Trabalho C2 - O Metodo Dual-Afim Trajetória Central de Pontos Interiores
% Aluno: Rafael Pavan
% Programa de Pos-Graduacao em Engenharia Eletrica - UNESP Bauru
% Mestrando em Engenharia Eletrica

% Problema Exemplo:
% minimizar -2*x1 - x2
% sujeito a:
% -x1 + x2 + s1 = 1
% x1 + s2 = 3
% x2 + s3 = 2
% -x1 + s4 = 0
% -x2 + s5 =0
% s1,s2,s3,s4,s5 >= 0

% Ponto Inicial X0:

f = [];

iteracoes=0;

x0 = [1.6,0.75];


% Matriz A

A = [-1 1;1 0; 0 1; -1 0; 0 -1];

% Vetor b

b = [1 3 2 0 0];

% Vetor c

c = [-2 -1];

% episolon
episolon = 0.01;

condicao_de_parada=0;

% Vetor s0 inicial

s0 = transpose(b) -A*transpose(x0);

% inicializa Sk

Sk = zeros(length(s0),length(s0));

u0 = 1.5;

iteracoes=1;

while(condicao_de_parada==0)

for linha=1:length(s0)

    coluna = linha;

    Sk(linha,coluna) = s0(coluna);

end

e = ones(1,length(s0));

% Obtem direcao de translacao:

dxk = - inv((transpose(A)*inv(Sk)*inv(Sk)*A))*(transpose(c)+u0*transpose(A)*inv(Sk)*transpose(e));

dsk =  -A*dxk;


% Verifica ilimitariedade:

if length(abs(dsk(abs(dsk)<episolon))) == length(dsk)
    
    display('Otimo Primal Encontrado.');

end
if length(dsk(dsk>0)) == length(dsk)
    
    display('Problema Ilimitado.')

end

% Calculo do vetor estimativa dual:

wk = inv(Sk)*inv(Sk)*dsk;

% Teste de Otimalidade:

%Teste 1

if length(wk(wk>0)) == length(wk)
    
    % E
    
    if e*Sk*wk <= episolon
    
      condicao_de_parada = 1;
        
        display('A. Otimo Primal Encontrado.')
        
    end
    
    
end


%Teste 2

if length(wk(wk>0)) == length(x0)
    
    % E
    
    if abs(x0*transpose(c)-b*wk) <= episolon
    
        condicao_de_parada = 1;
        
        abs(x0*transpose(c)-b*wk)
        
        
        display('B. Otimo Primal Encontrado.');
        
        
    end
    
    
end

%Teste 3

    % E

if length(wk(wk>0)) == length(wk)

    
    [m,n] = size(A);
    
    if length(s0(s0<episolon)) == m-n

            condicao_de_parada = 1;

            display('C. Otimo Primal Encontrado.');

end
    
    
end    
     
        

% Calculo do Passo:

lista = [];
alfa = 0.9995;

for i=1:length(s0)
    
    if dsk(i)<0
        
        lista=[lista,-alfa*s0(i)/dsk(i)];

    end
    
end

u0 = u0/2;

bk = min(lista);


xk=x0
sk=s0
x0 = x0 + bk*transpose(dxk);
s0 = s0 + bk*dsk;

f = [f, -2*x0(1)-x0(2)];


 iteracoes = iteracoes+1 


if iteracoes==2
    
   condicao_de_parada=1
    
end

iteracoes
xfinal = x0;
end


plot(f)

title('Curva de Convergencia')
xlabel('Iteracao')
ylabel('Funcao Objetivo')
grid minor
display('Valor Otimo: ')
xfinal
