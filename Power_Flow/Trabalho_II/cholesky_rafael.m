
%CasodeTeste1 

% A = [16,-4,12,-4;-4,2,-1,1;12,-1,14,-2;-4,1,-2,83;];

%CasodeTeste2 

% A = [1 2 3; 4 5 6; 7 8 9 ];

%CasodeTeste3

% A= [2 -1 0; -1 2 -1; 0 -1 2]

%CasodeTeste4

% A= [-1 0;0 -1]

%CasodeTeste5

% A= [1 2 ; 2 1; 3 3]

A= [2 -1 0; -1 2 -1; 0 -1 2]; %<-INSIRA A MATRIZ AQUI

A=real(A);
 
[l c] = size(A);

quad=0;
pos=0;
sim=0;

% Verifica se � quadrada

if (l ~= c )
    quad=1;
else

% Verifica se � sim�trica

for i=1:l
    for j=1:c
       if (A(i,j) ~= A(j,i))
	    	sim=sim+1;
       end
	end
end


G=zeros(l,c);
Gt=zeros(l,c);
G(1,1) = sqrt(A(1,1));

% Aplica o algoritmo de Fatora��o de Cholesky (P�g. 153 - M�todos Num�ricos
% - Aspectos Te�ricos e Computacionais - M�rcia A. Gomes Ruggiero 

for k = 1:l
    soma=0;
    for j = 1:k-1
        soma=soma+G(k,j)*G(k,j);
    end
    R = A(k,k) - soma;
    pos=0;
    if R<=0;
        pos=pos+l;
    end
    
    G(k,k) = sqrt(R);
    
        for i=(k+1):l
            soma = 0;
                for j =1:k-1
                     soma = soma + G(i,j)*G(k,j);
                end
            G(i,k) = (A(i,k)-soma)/G(k,k);
        end 

end
end

% Emite Relat�rio com Resultados

disp('Relat�rio do Programa:')
disp('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _')
disp(' ')
disp('Matriz de Entrada:')
disp(' ')
A
disp('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _')
disp(' ')
if quad ~= 0
    disp('- A matriz n�o � quadrada.')
    disp('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _')
else
    disp('- A matriz � quadrada.')
  
    if sim ~= 0
        disp('- A matriz n�o � sim�trica.')
        
    else
        disp('- A matriz � sim�trica.')
        
        if pos ~=0
                disp('- A matriz n�o � definida positiva.')
                
        else
                disp('- A matriz � definida positiva.')
                disp('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _')
                disp(' ')
                disp('Fator de Cholesky:')
                disp(' ')

                G
                Gt=transpose(G)
        end
    end
end

    