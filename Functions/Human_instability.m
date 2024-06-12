%% Initialize
 close all;
 clc;
 clear all;

%% Static values
input_table = readcell('Input_Main_Data_RTC.xlsx');
input_table(:,1) = [];
input_table(:,2) = [];
input_data = cell2mat(input_table);

m = input_data(49,1); %m
L = input_data(50,1); %m
Cd_HI = input_data(51,1); %adm
u = input_data(52,1); %adm
B = input_data(53,1); %m
pperson = input_data(54,1); %Kg/m³
pfluid = input_data(55,1); %Kg/m³
h_max_channel = input_data(56,1); %m
L_channel = input_data(57,1); %m
 
 %% Input from MATLAB
 load out_r_final_results.mat
 load time_plot_results.mat;
 load altura_maxima_canal_results.mat;
 out_r = out_r_final';
 h_edges = round([0 : 0.1 : h_max_channel],1,"decimals");
 %zzz is the document's name of the water level series
 time_plot = ([1: 1: length(zzz)]); % min
 A = L_channel.*zzz; %m²
 v = out_r./(A); %m/s

 %% Equation of human instability (Rotava, 2013)
 Fperson = m*g;
 Vc = (m.*zzz)/(L*pperson); 
 Fbuoy = pfluid.*Vc*g;
 Ffriction = u*(Fperson-Fbuoy);
 Fflow = 0.5*pfluid*Cd*B*(v.^2).*zzz;
 Behavior = Fflow-Ffriction;

%% Graphic histogram_colormap
% Need change the color of colorbar for each aplication or automatizate te code
imagesc(time_plot,h_edges,Behavior);
set(gca,'YDir','normal');
ylabel({'Nível de água no canal (m)'});
hold on
plot(zzz, Color='white', LineWidth=1,LineStyle='--');
% Create xlabel
xlabel({'Escala de tempo na simulação (s)'});
colorbar('Ticks',[(round(min(Behavior),0)/2),0-(round(min(Behavior),0)*0.1),round(max(Behavior),0)/2],...
         'TickLabels',{'Seguro','Alerta','Arraste'},'Color','black','Limits',[round(min(Behavior),0),round(max(Behavior),0)]);
colormap('copper')
ylim([0,round(max(zzz),0)])
grid;