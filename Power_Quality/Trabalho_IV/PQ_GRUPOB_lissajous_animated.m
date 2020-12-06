% Desenvolvido por Rafael Pavan
% Programa de Pós Graduação em Engenharia Elétrica da UNESP
% Disciplina de Qualidade de Energia Elétrica


curve = animatedline('LineWidth',0.5);
set(gca,'XLim',[-127*sqrt(2), 127*sqrt(2)],'YLim',[-4000 4000],'ZLim',[0 50]);

hold on

grid on
grid minor
title('Trajetória V-I: Carga Resistiva - Reativa 1.5|45º')
xlabel('Tensão [V]')
ylabel('Corrente [A]')
zlabel('Tempo [s]')
view(-69,14)


for i=1:length(vas)
    addpoints(curve,vas(i),ias(i),1:length(ias(i)));
    drawnow;
    pause(0.001);
end
