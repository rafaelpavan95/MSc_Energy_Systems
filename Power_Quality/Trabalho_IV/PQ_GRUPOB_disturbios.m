% Desenvolvido por Rafael Pavan
% Programa de P�s Gradua��o em Engenharia El�trica da UNESP
% Disciplina de Qualidade de Energia El�trica

clc
clear all

f = 60;
fsample = 1/48000;
t=[0 :fsample:0.2];
a=1;

% Tens�o Puramente Senoidal

y=a*sin(2*f*pi*t);
figure
plot(t,y)
title('Tens�o Puramente Senoidal (60 Hz)')
xlabel('Tempo [s]')
ylabel('Tens�o [pu]')
grid on
grid minor

% Afundamento de Tens�o

alpha=0.5;

y2=(1-alpha*((heaviside(t-0.04)-heaviside(t-0.1)))).*y;
figure(2)
plot(t,y2);
title('Afundamento de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [pu]')
grid on
grid minor


% Eleva��o de Tens�o

alpha=0.5;

y3=(1+alpha*((heaviside(t-0.04)-heaviside(t-0.1)))).*y;
figure
plot(t,y3);
title('Eleva��o de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [pu]')
grid on
grid minor

% Interrup��o de Tens�o

alpha=1;

y4=(1-alpha*((heaviside(t-0.04)-heaviside(t-0.1)))).*y;
figure
plot(t,y4);
title('Interrup��o de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [pu]')
grid on
grid minor

% Harm�nicas

h3=0.15;
h5=0.15;
h7=0.15;
h1= 1;
y5 = h1*y+ h3*a*sin(2*f*pi*3*t)+h5*a*sin(2*f*pi*5*t)+h7*a*sin(2*f*pi*7*t);
figure
plot(t,y5)
title('Harm�nicas');
xlabel('Tempo [s]')
ylabel('Tens�o [pu]')
grid on
grid minor

% Transit�rio

fn=1250;
t1=0.04; t2=0.1; 
tmedio = (t1+t2)/5;
amp=3;
t1=0.04; t2=0.1; 
ty= (t1+t2)/2;

y6 = y + amp*(heaviside(t-t1)-heaviside(t-t2)).*exp(-t/tmedio).*sin(2*3.14*fn*t);
figure
plot(t,y6)
title('Transit�rio');
xlabel('Tempo [s]')
ylabel('Tens�o [pu]')
grid on
grid minor

% Afundamento + Harm�nica

alpha=0.3;

y7=(1-alpha*((heaviside(t-0.04)-heaviside(t-0.1)))).*h1.*y+h3*a*sin(2*f*pi*3*t)+h5*a*sin(2*f*pi*5*t)+h7*a*sin(2*f*pi*7*t);
figure;
plot(t,y7);
title('Afundamento de Tens�o e Harm�nicas');
xlabel('Tempo [s]')
ylabel('Tens�o [pu]')
grid on
grid minor

% Eleva��o + Harm�nica

alpha=0.3;

y8=(1+alpha*((heaviside(t-0.04)-heaviside(t-0.1)))).*h1.*y+h3*a*sin(2*f*pi*3*t)+h5*a*sin(2*f*pi*5*t)+h7*a*sin(2*f*pi*7*t);
figure;
plot(t,y8);
title('Eleva��o de Tens�o e Harm�nicas');
xlabel('Tempo [s]')
ylabel('Tens�o [pu]')
grid on
grid minor

