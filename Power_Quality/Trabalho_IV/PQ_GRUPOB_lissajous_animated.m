% Desenvolvido por Rafael Pavan
% Programa de P�s Gradua��o em Engenharia El�trica da UNESP
% Disciplina de Qualidade de Energia El�trica


curve = animatedline('LineWidth',0.5);
set(gca,'XLim',[-127*sqrt(2), 127*sqrt(2)],'YLim',[-4000 4000],'ZLim',[0 50]);

hold on

grid on
grid minor
title('Trajet�ria V-I: Carga Resistiva - Reativa 1.5|45�')
xlabel('Tens�o [V]')
ylabel('Corrente [A]')
zlabel('Tempo [s]')
view(-69,14)


for i=1:length(vas)
    addpoints(curve,vas(i),ias(i),1:length(ias(i)));
    drawnow;
    pause(0.001);
end
