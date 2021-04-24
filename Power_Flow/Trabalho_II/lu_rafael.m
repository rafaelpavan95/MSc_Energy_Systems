A = [16,-4,12,-4;-4,2,-1,1;12,-1,14,-2;-4,1,-2,83;];



A=real(A)


[l c]=size(A);

if (l ~= c )
    disp ( 'Erro, a matriz deve ser quadrada.' );
else
    disp ( 'A matriz é quadrada.' )

x=0;
for i=1:l
	for j=1:c
		if (A(i,j) ~= A(j,i))
	    	x=x+1;
		end
	end
end

if x ~= 0
	disp('Erro, A matriz não é simétrica.')
else
	disp('A matriz é simétrica.')


L=zeros(l,c);
U=zeros(l,c);
Ub=zeros(l,c);
D=zeros(l,c);
Db=zeros(l,c);

%Realiza Decomposição LU
for i=1:l
    for k=1:i-1
        L(i,k)=A(i,k);
        for j=1:k-1
            L(i,k)= L(i,k)-L(i,j)*U(j,k);
        end
        L(i,k) = L(i,k)/U(k,k);
    end
  
    for k=i:l
        U(i,k) = A(i,k);
        for j=1:i-1
            U(i,k)= U(i,k)-L(i,j)*U(j,k);
        end
    end
end

for i=1:l
    L(i,i)=1;
end
  
  L; % Matriz L
  
  U; % Matriz U
 
% Encontra a matriz Ubarrada

for z=1:l
    for t=1:c
        Ub(z,t) = U(z,t)/U(z,z);
    end
end
Ub;

% Encontra a matriz D diagonal
for u=1:l
    
D(u,u)=U(u,u);
  
end
% Verifica se a matriz possui elementos negativos, caso sim não é positiva
dp=0;
for i=1:l
    for j=1:c
        if D(i,j)<0
        dp=dp+1;
        end
    end
end

if dp == 0
    
    disp('A matriz é definida positiva.')
D;
  
Db = sqrt(D);
 % Encontra matriz fator de Cholesky
G = L*Db
 
Gt = Db*transpose(L)
else
    disp('A matriz não é definida positiva.')
end


end  
end
