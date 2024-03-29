
% Desenvolvido por Rafael Pavan
% Programa de P�s Gradua��o em Engenharia El�trica da UNESP

f = 60;

fsample = 1/48000;
t = [0 :fsample:0.2];

y=sin(2*pi*60*t);

s=1;

sref=1;

R1 = ((s./2).*exp(-i*pi/2)+(sref/2)*exp(i*pi/2)).*exp(i*2*pi*f*t)+((s./2).*exp(i*pi/2)-(sref/2)*exp(-i*pi/2)).*exp(-i*2*pi*f*t)

figure
subplot(2,1,1)
plot(t,s*y)
title('Onda Sem Dist�rbios');
xlabel('Tempo [s]')
ylabel('Tens�o [PU]')
grid on
grid minor

subplot(2,1,2)
plot(real(R1),imag(R1),'LineWidth',2,'Color','blue')
title('Onda Sem Dist�rbios - Space Vector Ellipse in Complex Plane');
xlabel('Tens�o - Real [pu]')
ylabel('Tens�o - Imag [pu]')
grid on
grid minor
ylim([-1.5 1.5])
xlim([-1.5 1.5])

alpha=0.5;

s=1*(1-alpha*((heaviside(t-0.04)-heaviside(t-0.1))))
sref=1;

R2 = ((s./2).*exp(-i*pi/2)+(sref/2)*exp(i*pi/2)).*exp(i*2*pi*f*t)+((s./2).*exp(i*pi/2)-(sref/2)*exp(-i*pi/2)).*exp(-i*2*pi*f*t)

figure
subplot(2,1,1)
plot(t,s.*y)
title('Afundamento de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [PU]')
grid on
grid minor

subplot(2,1,2)
plot(real(R2),imag(R2),'LineWidth',2,'Color','red')
title('Afundamento de Tens�o - Space Vector Ellipse in Complex Plane');
xlabel('Tens�o - Real [pu]')
ylabel('Tens�o - Imag [pu]')
grid on
grid minor
hold on
plot(real(R1),imag(R1),'LineWidth',2,'Color','blue')
ylim([-1.5 1.5])
xlim([-1.5 1.5])
legend('Afundamento 0.5 pu','Refer�ncia')


alpha=-0.5;

s=1*(1-alpha*((heaviside(t-0.04)-heaviside(t-0.1))))
sref=1;

R3 = ((s./2).*exp(-i*pi/2)+(sref/2)*exp(i*pi/2)).*exp(i*2*pi*f*t)+((s./2).*exp(i*pi/2)-(sref/2)*exp(-i*pi/2)).*exp(-i*2*pi*f*t)

figure
subplot(2,1,1)
plot(t,s.*y)
title('Eleva��o de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [PU]')
grid on
grid minor
subplot(2,1,2)
plot(real(R3),imag(R3),'LineWidth',2,'Color','red')
title('Eleva��o de Tens�o - Space Vector Ellipse in Complex Plane');
xlabel('Tens�o - Real [pu]')
ylabel('Tens�o - Imag [pu]')
grid on
grid minor
hold on
plot(real(R1),imag(R1),'LineWidth',2,'Color','blue')
ylim([-2 2])
xlim([-2 2])
legend('Eleva��o de 0.5 pu','Refer�ncia')

alfa=1
s=1*(1-alfa*((heaviside(t-0.04)-heaviside(t-0.1))))
sref=1;

R4 = ((s./2).*exp(-i*pi/2)+(sref/2)*exp(i*pi/2)).*exp(i*2*pi*f*t)+((s./2).*exp(i*pi/2)-(sref/2)*exp(-i*pi/2)).*exp(-i*2*pi*f*t)

figure
subplot(2,1,1)
plot(t,s.*y)
title('Interrup��o de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [PU]')
grid on
grid minor

subplot(2,1,2)
plot(real(R4),imag(R4),'LineWidth',2,'Color','red')
title('Interrup��o de Tens�o - Space Vector Ellipse in Complex Plane');
xlabel('Tens�o - Real [pu]')
ylabel('Tens�o - Imag [pu]')
grid on
grid minor
hold on
plot(real(R1),imag(R1),'LineWidth',2,'Color','blue')
ylim([-1.5 1.5])
xlim([-1.5 1.5])
legend('Interrup��o','Refer�ncia')

h3=0.15;
h5=0.15;
h7=0.15;
h1= sqrt(1-h3^2-h5^2-h7^2);
s = h1*sin(2*f*pi*t)+ h3*sin(2*f*pi*3*t)+h5*sin(2*f*pi*5*t)+h7*sin(2*f*pi*7*t);
sref=1;

R5 = ((s./2).*exp(-i*pi/2)+(sref/2)*exp(i*pi/2)).*exp(i*2*pi*f*t)+((s./2).*exp(i*pi/2)-(sref/2)*exp(-i*pi/2)).*exp(-i*2*pi*f*t)


figure
subplot(2,1,1)
plot(t,s)
title('Harm�nicas');
xlabel('Tempo [s]')
ylabel('Tens�o [PU]')
grid on
grid minor

subplot(2,1,2)

plot(real(R5),imag(R5),'LineWidth',2,'Color','red')
title('Harm�nicas - Space Vector Ellipse in Complex Plane');
xlabel('Tens�o - Real [pu]')
ylabel('Tens�o - Imag [pu]')
grid on
grid minor
hold on
plot(-real(R5),imag(R5),'LineWidth',2,'Color','red')
ylim([-1.5 1.5])
xlim([-1.5 1.5])
hold on
plot(real(R1),imag(R1),'LineWidth',2,'Color','blue')
ylim([-1.5 1.5])
xlim([-1.5 1.5])
legend('Harm�nicas')

fn=1250;
t1=0.04; t2=0.1; 
tmedio = (t1+t2)/5;
amp=3;
t1=0.04; t2=0.1; 
ty= (t1+t2)/2;

s = y + amp*(heaviside(t-t1)-heaviside(t-t2)).*exp(-t/tmedio).*sin(2*3.14*fn*t);
sref=1;

R6 = ((s./2).*exp(-i*pi/2)+(sref/2)*exp(i*pi/2)).*exp(i*2*pi*f*t)+((s./2).*exp(i*pi/2)-(sref/2)*exp(-i*pi/2)).*exp(-i*2*pi*f*t)
figure
subplot(2,1,1)
plot(t,s)
title('Transit�rio de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [PU]')
grid on
grid minor

subplot(2,1,2)
plot(real(R6),imag(R6),'LineWidth',1,'Color','red')
hold on
plot(-real(R6),imag(R6),'LineWidth',1,'Color','red')
title('Transit�rio - Space Vector Ellipse in Complex Plane');
xlabel('Tens�o - Real [pu]')
ylabel('Tens�o - Imag [pu]')
grid on
grid minor
ylim([-2 2])
xlim([-3 3])
hold on
plot(real(R1),imag(R1),'LineWidth',2,'Color','blue')
legend('Transit�rio')


alpha=0.3;

s=(1-alpha*((heaviside(t-0.04)-heaviside(t-0.1)))).*h1.*sin(2*f*pi*t)+h3*sin(2*f*pi*3*t)+h5*sin(2*f*pi*5*t)+h7*sin(2*f*pi*7*t);

sref=1;

R7 = ((s./2).*exp(-i*pi/2)+(sref/2)*exp(i*pi/2)).*exp(i*2*pi*f*t)+((s./2).*exp(i*pi/2)-(sref/2)*exp(-i*pi/2)).*exp(-i*2*pi*f*t)

figure
subplot(2,1,1)
plot(t,s)
title('Harm�nica com Afundamento de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [PU]')
grid on
grid minor

subplot(2,1,2)
plot(real(R7),imag(R7),'LineWidth',2,'Color','red')
hold on
plot(-1*real(R7),imag(R7),'LineWidth',2,'Color','red')
hold on
plot(real(R5),imag(R5),'LineWidth',2,'Color','blue')
plot(-real(R5),imag(R5),'LineWidth',2,'Color','blue')
grid on
grid minor 
title('Afundamento de Tens�o Com Harm�nica - Space Vector Ellipse in Complex Plane');
xlabel('Tens�o - Real [pu]')
ylabel('Tens�o - Imag [pu]')
legend('Harm�nica com Afundamento')

ylim([-2 2])
xlim([-2 2])


alpha=-0.3;

s=(1-alpha*((heaviside(t-0.04)-heaviside(t-0.1)))).*h1.*sin(2*f*pi*t)+h3*sin(2*f*pi*3*t)+h5*sin(2*f*pi*5*t)+h7*sin(2*f*pi*7*t);

sref=1;

R8 = ((s./2).*exp(-i*pi/2)+(sref/2)*exp(i*pi/2)).*exp(i*2*pi*f*t)+((s./2).*exp(i*pi/2)-(sref/2)*exp(-i*pi/2)).*exp(-i*2*pi*f*t)


figure
subplot(2,1,1)
plot(t,s)
title('Harm�nica com Eleva��o de Tens�o');
xlabel('Tempo [s]')
ylabel('Tens�o [PU]')
grid on
grid minor

subplot(2,1,2)
plot(real(R8),imag(R8),'LineWidth',2,'Color','red')
hold on
plot(-1*real(R8),imag(R8),'LineWidth',2,'Color','red')
hold on
plot(real(R5),imag(R5),'LineWidth',2,'Color','blue')
plot(-real(R5),imag(R5),'LineWidth',2,'Color','blue')
grid on
grid minor 
title('Eleva��o de Tens�o Com Harm�nica - Space Vector Ellipse in Complex Plane');
xlabel('Tens�o - Real [pu]')
ylabel('Tens�o - Imag [pu]')
legend('Harm�nica com Eleva��o')

ylim([-2 2])
xlim([-2 2])
