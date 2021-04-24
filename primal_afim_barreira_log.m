clear all
clc

x0 = transpose([10,6,11,9]);

c = transpose([-2 1 0 0]);

e = [1; 1; 1;1];

A = [1 -1 1 0; 0 1 0 1];

mi0 = 1.5;

max_iter = 4;

xk = x0;

iter=1;

while(iter<max_iter)
    
    for i=1:4

       Xk(i,i) = xk(i);

    end

    gradf = c - inv(Xk)*mi0*e;

    wk = inv(A*Xk*Xk*transpose(A))*A*Xk*Xk*gradf;

    rk = gradf-transpose(A)*wk;

    dyk = -Xk*rk;

    alfa = 0.9995 / -min(dyk(dyk<0));
    
    if alfa > 10
        
        break
        
        display('Ponto Ótimo Encontrado')
    
    end
    
    yk = e+alfa*dyk;

    xk = xk.*yk;

    mi0 = mi0/2;

    iter=iter+1;
end

display('Solução')

xk