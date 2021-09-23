% %%%%%%%%% Particle Swarm Optimization
% %%%%%%%%% CASO TESTE IEEE 300
% %%%%%%%%% Desenvolvedor: Rafael Pavan
% clear all
% clc
startup 
% 
% sep = loadcase('case300');
sep = loadcase('case14');
% 
%sep = runpf(sep);
% 
% sep.bus(:,12)=1.1
% 
% sep.bus(:,13)=0.9
% 
% fit = [];
% 
% losses = [];
% %%%%%%%%% NÚMERO DE VARIÁVEIS DE CONTROLE
% 
% n_gen = 69;
% n_tap = 107;
% n_shunt = 14;
% S_base = 100;
% w_max = 0.9;
% w_min = 0.4;
% atenuacao = 0.01;
% 
% alfa = 10; % PENALIZAÇÃO TENSÃO
% 
% % gg = [1.00905107855947	0.964035308652737	1.01452013746182	0.964623369669479	0.957714984482626	1.03212548446613	1.03795297215737	1.01569285688411	1.03928345898037	0.989010825583057	1.06055959920714	0.950670913037730	0.961368956100686	1.00566185560185	1.02377553362306	1.03796469815123	1.00009461934625	1.00397777743047	1.03759006810946	1.06163528210851	1.06702882533057	1.04157356246014	0.997463678947823	0.955788777415943	1.04351236814180	0.978601031832252	1.06383786331800	1.03840472244315	1.04101390246409	1.01006361314043	1.10000000000000	1.03298027833548	0.952280602653974	0.965559307631808	1.00803205465057	1.00878327367171	0.968371591103022	0.987587530277715	1.01131024398774	1.02886854323717	0.961340996965163	1.02845031840093	0.991247771806834	0.959515614034709	0.980721894147692	1.03841966920901	0.996503035407688	0.996060175117965	0.981362984527196	1.03600931937675	0.989485782833823	1.00099666729834	1.05237766328319	1.04101225169891	1.02952624016048	1.02819608955548	1.02935558381564	1.02497841904143	1.02475510579308	0.987083107174962	1.00301682775214	1.01444264049173	1.03922227017849	1.03692694063185	1.00285044714938	0.953253991905232	0.989981865484023	1.02277303465681	0.993055728776455	0.987778516059029	0.969086109894971	0.957668936768631	1.09677389363529	0.970126400588137	1.04852378800148	0.980226679337799	0.992483113960740	1.06535290260279	1.03494208757182	1.06858843334916	0.995527101998934	1.01664560159408	0.997352488170912	0.986360586195613	1.02665928187442	0.982087156233659	0.972154511689284	0.986516154468309	1.06732330753892	1.00311088255122	0.983649695360602	1.03038558033817	0.994442947895189	0.973196565672362	0.987454636275395	0.991186123682528	1.01948212599907	1.00485927621761	0.989381669434984	0.989380493359159	1.04094052999687	0.965559489886998	1.06781582440554	1.01416387614158	0.941972140054076	1.03552036483522	0.993163945582640	1.00578144798120	0.941182588245496	1.03834117033744	0.977370254025068	0.978708748533380	1.05097980577576	0.971595619179971	1.05839037892846	1.00609221375407	1.00225850642530	0.979911576163325	1.00011630978344	1.00122518766182	1.01404805860083	0.997714217724128	1.02746711305383	0.997346921794255	1.03360839553741	1.05189589022086	0.992475431588225	1.01011004216500	1.00930023492468	1.00020240436480	0.971723189207028	0.993881726152056	1.00805429689373	1.02257389142831	1.00517517152490	1.04822171074928	1.02464344448085	1.04080612053662	0.945559989856376	1.01203991965278	1.03018828385989	1.01457862865791	1.00355875441724	1.01335802009073	1.01981320340192	0.984736201762526	0.984663511624391	1.04685339594919	1.01775909882078	1.00751204143742	1.06177582429164	0.934776107301890	0.968157553807589	0.942548432743970	1.04294160299120	0.992554675245637	0.982829460933067	0.933326215858205	0.937250128481272	1.01105953451663	0.950132326961870	1.01238754936180	1.10000000000000	1.00137425707223	0.900000000000000	0.957297353236093	0.953707448451371	1.02234396940508	0.978943195392707	0.963358078177175	0.957739777611244	0.978426732549911	0.969488310557602	0.946408901995546	1.00399968107289	1.00716201359055	404.636612074460	36.9158893098740	22.1538735114988	-145.524786158144	-221.590352372492	40.6693376850609	27.9099681221475	-139.176835680004	-210.703067719348	-208.969657972210	-55.5554978990142	11.8683591388715	4.62292756450625	5.87907861905282	418.247264293391	0	2.19932255032842e-07	418.247286286616]
% 
% gamma = 1; % PENALIZAÇÃO REATIVO
% 
% n_particles= 150;
% 
% max_iter= 500;
% 
% c1= 2;
% c2 = 2;
% 
% %%%%%%%%%% LIMITES DE TENSÃO
% 
% vmax = 1.1;
% vmin = 0.9;
% delta_v = ones(1,n_gen)*vmax - ones(1,n_gen)*vmin;
% 
% max_v = ones(1,n_gen)*vmax;
% 
% min_v = ones(1,n_gen)*vmin;
% 
% %%%%%%%%% LIMITES DO TRAFO
% 
% trafomin = 0.9;
% trafomax = 1.1;
% step = 0.01;
% 
% delta_t = ones(1,n_tap)*trafomax - ones(1,n_tap)*trafomin;
% 
% max_t = ones(1,n_tap)*trafomax;
% 
% min_t = ones(1,n_tap)*trafomin;
% 
% t = trafomin:step:trafomax;
% 
% enxame = ones(n_particles,190+4);
% 
% shunt_list = [96,99,133,143,145,152,158,169,210,217,219,227,268,283];
% 
% max_s = [4.5,0.59,0.39,0,0,0.59,0.59,0,0,0,0,0.59,0.15,0.15]*100;
% 
% min_s = [0,0,0,-4.5,-4.5,0,0,-2.5,-4.5,-4.5,-1.5,0,0,0]*100;
% 
% min_total = [min_v, min_t, min_s];
% 
% max_total = [max_v, max_t, max_s];
%  
% delta_s = [4.5, 0.59,0.39,4.5,4.5,0.59,0.59,2.5,4.5,4.5,1.5,0.59,0.15,0.15]*100;
% 
% % delta_s = max_s - min_s;
% 
% delta_total = [delta_v, delta_t, delta_s];
% 
% lista_gbest_val = [1e20];
% lista_gbest = [];
% 
% 
% init_gen = sep.gen(:,6);
% init_shunt = sep.bus(shunt_list,6);
% init_trafo = sep.branch(sep.branch(:,9)~=0,9);
% 
% 
% init = [transpose(init_gen), transpose(init_trafo), transpose(init_shunt)];
% 
% %% INICIALIZA ENXAME E VELOCIDADE
% 
% 
% enxame = cria_enxame(sep, n_particles, vmin,vmax,t,shunt_list,enxame);
% velocidade = cria_enxame(sep, n_particles, -vmax,vmax,t,shunt_list,enxame);
% velocidade = velocidade * atenuacao;
% enxame(1,:) = [init,0,0,0,0];
% enxame(:,191:194) = 1e3;
% 
% for iteracao=1:max_iter
% 
%     %% AVALIA ENXAME
% 
%     
%     enxame_avaliado = avalia_fluxo(enxame,sep,n_particles,vmin,vmax,S_base, shunt_list, n_gen,n_tap,n_shunt,alfa,gamma,t,iteracao);
% 
%     [valor,posi] = min(enxame_avaliado(:,length(enxame_avaliado(1,:))));
% 
%     enxame_avaliado(:,190:194)
%     %% DEFINE GBEST
% 
%     melhor = enxame_avaliado(posi,:);
% 
%     melhor_val = valor;
% 
%     if melhor_val < lista_gbest_val(length(lista_gbest_val))
% 
%         lista_gbest_val = [lista_gbest_val,melhor_val];
% 
%         lista_gbest = [lista_gbest; melhor];
% 
%         gbest = melhor;
%     end
%     
%     
%     
%     %% DEFINE MEMÓRIA PBEST
%     
%     if iteracao==1
%         
%         memoria=enxame_avaliado;
%     
%     end
%     
%     if iteracao>1
%         
%         for idx=1:n_particles
%             
%             if memoria(idx,length(memoria(1,:)))>enxame_avaliado(idx,length(enxame_avaliado(1,:)))
%                 
%                 memoria(idx,:) = enxame_avaliado(idx,:);
%                 
%             end
%             
%             
%         end
%         
%         
%     end
%     
%     %% CALCULA PESO DE INERCIA LINEAR
%     
%      w = w_max - (w_max-w_min)*iteracao/max_iter;
% 
% 
%     
%     %% ATUALIZA VELOCIDADES
%     
%     velocidade = velocidade.*w + c1.*rand(n_particles,1).*(memoria-enxame_avaliado) + c2.*rand(n_particles,1).*(gbest-enxame_avaliado);  
%     
%     passo = 0.01;
%         
%     for linha=1:n_particles
%         
%         for coluna=1:length(velocidade(1,:))-4
%             
%             if velocidade(linha,coluna) < -delta_total(coluna)*passo
%                 
%                velocidade(linha,coluna) = -delta_total(coluna)*passo;
%                
%             end
%                
%             if velocidade(linha,coluna) > delta_total(coluna)*passo
%                 
%                velocidade(linha,coluna) = delta_total(coluna)*passo;
%                
%             end
%             
%         end
%         
%     end
%                 
%     enxame_avaliado = enxame_avaliado + velocidade;
%     
%     
% 
%     for linha=1:n_particles
%         
%         for coluna=1:length(enxame_avaliado(1,:))-4
%             
%             if enxame_avaliado(linha,coluna) > max_total(coluna)
%                 
%                enxame_avaliado(linha,coluna) = max_total(coluna);
%                
%             end
%                
%             if enxame_avaliado(linha,coluna) < min_total(coluna)
%                 
%                enxame_avaliado(linha,coluna) = min_total(coluna);
%                
%             end
%             
%         end
%         
%     end
%     
%     
%     enxame = enxame_avaliado;
%     
% 
%     display('---------')
%     display('Iteração:')
%     display(num2str(iteracao))
%     display('Perdas: ')
%     display(num2str(gbest(length(gbest)-3)))
%     display('Penalização de Tensão: ')
%     display(num2str(gbest(length(gbest)-2)))
%     display('Penalização de Reativo: ')
%     display(num2str(gbest(length(gbest)-1)))
%     display('Fitness: ')
%     display(num2str(gbest(length(gbest))))
%     
%     display('Peso de Inércia: ')
%     display(num2str(w))
%     
%     losses = [losses, gbest(:,194-3)];
%     fit = [fit, gbest(:,194)];
%   
% 
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNÇÃO PARA AVALIAR ENXAME
% 
% gbest(194-4-14:190) = int64(gbest(194-4-14:190));
% 
% trafos = gbest(1+n_gen:n_gen+n_tap)
% 
% for i=1:length(trafos)
%     
%     aux = abs(t - trafos(i));
%     
%     [v,p] = min(aux);
%     
%     trafos(i) = t(p);
%     
% end
% 
% 
% 
% 
% 
% gbest(1+n_gen:n_gen+n_tap) = trafos;
% 
% sep = carregar_dados_no_sistema(sep,shunt_list,gbest,n_gen,n_tap,n_shunt)
% 
% sep = runpf(sep);
% 
% figure
% 
%     plot(1:1:n_gen, sep.gen(:,3), 'blue');
% 
% hold on
% plot(sep.gen(:,4), 'red');
% 
% plot(sep.gen(:,5), 'green');
% 
% grid on
% grid minor
% 
% 
% title('Potência Reativa dos Geradores')
% xlabel('Gerador')
% ylabel('Q [MVAR]')
% 
% figure
% 
% plot(sep.bus(:,8), 'blue');
% 
% hold on
% 
% plot(sep.bus(:,12), 'red');
% 
% plot(sep.bus(:,13), 'green');
% 
% title('Magnitude de Tensão nas Barras')
% xlabel('Barra')
% ylabel('|V| [pu]')
% 
% 
% 
% grid on
% grid minor
% 
% figure
% 
% x = 1:1:107;
% 
% yy = ones(length(x));
% 
% scatter(x,trafos,'x','red','DisplayName','Discreto')
% hold on
% scatter(x,init_trafo,'x','blue','DisplayName','Contínuo')
% hold on
% 
% 
% for i=1:length(t)
%     
%     plot(x,yy*t(i),'-.','Colo','black')
% 
% end
% 
% 
% grid minor
% 
% xlim([0,107])
% title('TAP dos Transformadores')
% xlabel('Transformador')
% ylabel('Valor do TAP')
% 
% 
% figure
% 
% plot(losses)
% title('Otimização das Perdas')
% xlabel('Iteração')
% ylabel('Perdas [MW]')
% grid on
% grid minor
% 
% figure
% 
% plot(fit)
% title('Otimização da Função Fitness')
% xlabel('Iteração')
% ylabel('Fitness')
% grid on
% grid minor
% 
% 
% 
% function enxame_aval = avalia_fluxo(enxame_,sep_,n_particles_,vmin_,vmax_,S_base_,shunt_list_,n_gen_,n_tap_,n_shunt_,alfa_,gamma_,t_, iteracao_)
% 
%     for k=1:n_particles_
%         
%         
%             
%        arredondado = enxame_(k,:);
% 
%         
%         tap = arredondado(n_gen_+1:n_gen_+n_tap_);
%         
%         for elemento=1:n_tap_
%             
%             aux = abs(t_-tap(elemento));
%             
%             [val, posi] = min(aux);
%             
%             tap(elemento) = t_(posi);
% 
%         end
%        
%         arredondado(n_gen_+1:n_gen_+n_tap_) = tap;
%         
%         arredondado(n_gen_+1+n_tap_:n_gen_+n_tap_+n_shunt_) = int64(arredondado(n_gen_+1+n_tap_:n_gen_+n_tap_+n_shunt_));
%        
%         sep_ = carregar_dados_no_sistema(sep_, shunt_list_, arredondado, n_gen_, n_tap_, n_shunt_);
%         
%         mpopt = mpoption('pf.alg','FDBX', 'verbose', 0,'out.all',0, 'pf.tol',1e-7, 'pf.fd.max_it',50);
%                    
%         sep_ = runpf(sep_,mpopt);
%         
%         
%         pen_v = penalizacao_tensao(sep_,vmax_,vmin_);
% 
%         perdas = calcula_perdas(sep_);
% 
%         pen_q = penalizacao_reativo(sep_);
% 
%         fitness = perdas + alfa_*pen_v + gamma_*pen_q;
% 
%         enxame_(k,length(enxame_(k,:))-3) = perdas;
%         enxame_(k,length(enxame_(k,:))-2) = pen_v;
%         enxame_(k,length(enxame_(k,:))-1) = pen_q;
%         enxame_(k,length(enxame_(k,:))) = fitness;
%         
%         enxame_aval = enxame_;
%         
%         
%     end
%     
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNÇÃO PARA CALCULAR PENALIZAÇÃO DE
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GERAÇÃO DE REATIVO
% 
% 
% function penq = penalizacao_reativo(sep_)
% 
%     dif_max = sep_.gen(:,4)-sep_.gen(:,3);
% 
%     max = sum(abs(dif_max(dif_max<0)));
% 
%     dif_min = sep_.gen(:,3)-sep_.gen(:,5);
% 
%     min = sum(abs(dif_min(dif_min<0)));
%     
%     penq = max+min;
%     
% end
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNÇÃO PARA CALCULAR PERDAS
% 
% function perdas_ = calcula_perdas(sep_)
% 
% 
%     perdas_ = sum(real(get_losses(sep_)));
% 
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNÇÃO PARA CALCULAR PENALIZAÇÃO DE TENSÃO
% 
% function penv = penalizacao_tensao(sep_,vmax_,vmin_)
% 
% 
%     sup = sum(abs(vmax_ -sep_.bus(sep_.bus(:,8)>vmax_,8)));
% 
%     inf = sum(abs(vmin_ -sep_.bus(sep_.bus(:,8)<vmin_,8)));
%     
%     penv = sup+inf;
% 
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNÇÃO PARA CARREGAR DADOS NO SISTEMA
% 
% function sep_ = carregar_dados_no_sistema(sep_,shunt_list_,vetor_,n_gen_,n_tap_,n_shunt_)
% 
%     sep_.branch((sep_.branch(:,9) ~= 0),9) = vetor_(n_gen_+1:n_gen_+n_tap_);
% 
%     sep_.gen(:,6) = vetor_(1:n_gen_);
% 
%     sep_.bus(shunt_list_,6) = vetor_(n_gen_+1+n_tap_:n_gen_+n_tap_+n_shunt_);
% 
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNÇÃO PARA CRIAR ENXAME
% 
% function enxame_ = cria_enxame(sep_, n_particles_, vmin_,vmax_,t_,shunt_list_,enxame_)
% 
% 
%     for p=1:n_particles_
% 
%         v_gen = vmin_ + (vmax_-vmin_)*rand(1,length(sep_.gen(:,6)));
%         enxame_(p,1:length(v_gen))=v_gen;
%         aux = sep_.branch(:,9);
%         taps = aux(aux ~= 0);
%         tap = [];
% 
%         for i=1:length(taps)
% 
%             tap = [tap, t_(randi(length(t_)))];
% 
%         end
% 
%         enxame_(p,length(v_gen)+1:length(v_gen)+length(tap))=tap;
% 
% 
%         shunts = [];
% 
%         for i=1:length(shunt_list_)
% 
%             if i==1
% 
%                 vetor = [0,2,3.5,4.5];            
%                 vetor = vetor*100;
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
% 
%             end
% 
%             if i==2
% 
%                 vetor = [0,0.25,0.44,0.59];
%                 vetor = vetor*100;
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
% 
%             end
% 
%             if i == 3
% 
%                 vetor = [0,0.19,0.34,0.39];
%                 vetor = vetor*100;
% 
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
% 
%             if i == 4
% 
%                 vetor = [-4.5,0];
%                 vetor = vetor*100;
% 
% 
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 5
% 
%                 vetor = [-4.5,0];
%                 vetor = vetor*100;
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 6
% 
%                 vetor = [0,0.25,0.44,0.59];
%                
%                 vetor = vetor*100;
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 7
% 
%                 vetor = [0,0.25,0.44,0.59];
%                 vetor = vetor*100;
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 8
% 
%                 vetor = [-2.5,0];
%                 vetor = vetor*100;
% 
%                 
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 9
% 
%                 vetor = [-4.5,0];
%                 vetor = vetor*100;
% 
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 10
% 
%                 vetor = [-4.5,0];
%                 vetor = vetor*100;
% 
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 11
% 
%                 vetor = [-1.5,0];
%                 vetor = vetor*100;
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 12
% 
%                 vetor = [0,0.25,0.44,0.59];
%                 vetor = vetor*100;
%                 
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
% 
% 
%             if i == 13
% 
%                 vetor = [0,0.15];
%                 vetor = vetor*100;
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%             if i == 14
% 
%                 vetor = [0,0.15];
%     
%                 vetor = vetor*100;
% 
%                 shunts = [shunts, vetor(randi(length(vetor)))];
%             end
% 
%         end
% 
%         enxame_(p,length(v_gen)+1+length(tap):length(v_gen)+length(tap)+length(shunts))=shunts;
% 
%     end
% 
% end
% 
% 

csvwrite('dadosdebarra.csv',sep.bus)

csvwrite('dadosgerador.csv',sep.gen)

csvwrite('dadosramo.csv',sep.branch)