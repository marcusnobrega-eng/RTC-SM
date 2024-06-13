%% Plotting Results
% Developer - Marcus Nobrega
% 8/1/2021
% Objective: Plot results from the simulation

%% Flows
close all
figure(1)
time_plot = [0:time_step:(n_steps-1)*time_step]/60/60; % min
dim = min(length(out_r_final),length(time_plot));
subplot(4,1,1)
set(gcf,'units','inches','position',[5,0,7,12])
plot(time_plot(1:dim),Qout_w(1:dim),'green','LineWidth',1)
hold on
plot(time_plot(1:dim),out_r_final(1:dim),'red','LineWidth',1)
xlabel('Elapsed time (hr)','Interpreter','Latex')
ylabel('Flow ($\mathrm{m^3/s}$)','Interpreter','Latex')
yyaxis right
plot(time_plot(1:dim),i_reservoir(1:dim),'blue')
set(gca,'FontSize',12)
set(gca,'ydir','reverse')
set(gca,'ycolor','black')
ylim([0,400])
ylabel('Rainfall (mm/h)','Interpreter','Latex')
legend('Watershed','Reservoir','Rainfall','Interpreter','Latex')
grid on
subplot(4,1,2)
% Water Levels
plot(time_plot(1:dim),h_r_final(1:dim),'blue','LineWidth',1)
xlabel('Elapsed Time (h)','FontSize',12,'Interpreter','Latex'); ylabel('Water Surface Depth (m)','FontSize',12,'Interpreter','Latex');
set(gca,'FontSize',12)
hold on
plot(time_plot(1:dim),Reservoir_Parameters.hs*ones(1,length(1:dim)),'black','LineWidth',1.5,'LineStyle','--')
grid on
hold on
plot(time_plot(1:dim),MPC_Control_Parameters.max_res_level*ones(1,length(1:dim)),'red','LineWidth',1.5,'LineStyle','-.')
legend('show')
legend('Water Depth','Spillway Height','$h^r_{max}$','interpreter','latex')
subplot(4,1,3)
% Controls
% z = length(u);
% u(z+1) = u(z-1);
%%% ---- NO GATE CONTROL ---- %%%
if flags.flag_gatecontrol ~=1
    plot(time_plot(1:dim),u(1:dim),'red','LineWidth',1)
else
    plot(time_plot(1:dim),u_v(1:dim),'red','LineWidth',1,'LineStyle','--')
    hold on
    plot(time_plot(1:dim),u_s(1:dim),'blue','LineWidth',1,'LineStyle',':');
    legend('Valve','Gate')
end
set(gca,'FontSize',12)
set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1])
xlabel('Elapssed Time (hr)','Interpreter','Latex')
ylabel('Control Signal','Interpreter','Latex')
grid on

subplot(4,1,4)
plot([0:1:(n_horizons-1)]*MPC_Control_Parameters.Control_Horizon/60,OF_value,'green','LineWidth',1,'Marker','*','Color','black')
xlabel('Time (hr)','Interpreter','latex')
ylabel('Cost Function','Interpreter','latex')
grid on
exportgraphics(gcf,'Outputs\MPC.pdf','ContentType','vector')

% % Exporting Chart
% exportgraphics(gcf,'RTC_Charts.pdf','ContentType','vector')
% if flag_save_data == 1
%     % Exporting txt files
%     Reservoir_Outflow = table((time_plot(1:dim)'),(out_r_final(1:dim)),'VariableNames',{'Time (hr)','Reservoir Outflow (m3/s)'});
%     writetable(Reservoir_Outflow)
%     Watershed_Outflow = table((time_plot(1:dim)'),(Qout_w(1:dim)),'VariableNames',{'Time (hr)','Watershed Outflow (m3/s)'});
%     writetable(Watershed_Outflow)
%     Channel_Level = table((time_plot(1:dim)'),(max(h_c_final(:,1:dim)))','VariableNames',{'Time (hr)','Channel Surface Depth (m)'});
%     writetable(Channel_Level)
%     Reservoir_Level = table((time_plot(1:dim)'),(h_r_final(1:dim)),'VariableNames',{'Time (hr)','Channel Surface Depth (m)'});
%     writetable(Reservoir_Level)
%     Channel_Outflow = table((time_plot(1:dim)'),(out_c_final(1:dim)),'VariableNames',{'Time (hr)','Channel Outflow (m3/s)'});
%     writetable(Channel_Outflow)
% end
% close all
%% Hydrographs
figure(2)
time_plot = [0:time_step:(n_steps-1)*time_step]/60/60; % min
set(gcf,'units','inches','position',[5,0,5,3])
plot(time_plot(1:dim),Qout_w(1:dim),'green','LineWidth',1)
hold on
plot(time_plot(1:dim),out_r_final(1:dim),'red','LineWidth',1)
xlabel('Elapsed time (hr)','Interpreter','Latex')
ylabel('Flow ($m^3/s$)','Interpreter','Latex')
yyaxis right
plot(time_plot(1:dim),i_reservoir(1:dim),'blue')
set(gca,'FontSize',12)
set(gca,'ydir','reverse')
set(gca,'ycolor','black')
ylim([0,400])
ylabel('Rainfall (mm/h)','Interpreter','Latex')
legend('Watershed','Reservoir','Rainfall','Interpreter','Latex')
exportgraphics(gcf,'Outputs\Hydrographs_System_Detail.pdf','ContentType','vector')

%% Objective Function Values
figure(3)
set(gcf,'units','inches','position',[5,0,5,3])
plot([1:1:n_horizons],OF_value,'green','LineWidth',1)
xlabel('Control Horizon','Interpreter','latex')
ylabel('Cost Function','Interpreter','latex')
grid on
exportgraphics(gcf,'Outputs\Objective_Function.pdf','ContentType','vector')


%% Stage-Area-Volume
% Reservoir functions
Area_Value = zeros(dim,1);
Volume_Value = zeros(dim,1);
for zzz = 1:dim
    [~,Area,~,Volume] = reservoir_area(h_r_final(zzz),stage_area,1); % Call Reservoir Functions
    Area_Value(zzz,1) = Area;
    Volume_Value(zzz,1) = Volume;
end
% time_plot = [0:time_step:(n_steps-1)*time_step]/60/60; % min
% set(gcf,'units','inches','position',[5,3,5,3])
% plot(time_plot(1:dim),Area_Value,'green','LineWidth',1)
% 
% xlabel('Elapsed time (hr)','Interpreter','Latex')
% ylabel('Area ($m^2$)','Interpreter','Latex')
% yyaxis right
% hold on
% plot(time_plot(1:dim),Volume_Value,'red','LineWidth',1)
% ylabel('Volume ($\mathrm{m^3}$)','Interpreter','Latex')
% legend('Area','Volume','Interpreter','Latex')
% exportgraphics(gcf,'Area_Volume.pdf','ContentType','vector')
ylabels = {'Area ($\mathrm{m^2}$)','Volume ($\mathrm{m^3}$)','Stage ($\mathrm{m}$)'};
[ax,hlines] = plotyyy(time_plot(1:dim),Area_Value,time_plot(1:dim),Volume_Value,time_plot(1:dim),h_r_final(1:dim,1),ylabels);
ax(1).XAxis.TickLabelInterpreter = 'latex';
ax(2).XAxis.TickLabelInterpreter = 'latex';
ax(3).XAxis.TickLabelInterpreter = 'latex';
ax(1).YAxis.TickLabelInterpreter = 'latex';
ax(2).YAxis.TickLabelInterpreter = 'latex';
ax(3).YAxis.TickLabelInterpreter = 'latex';
hlines(1).LineWidth = 2;
hlines(1).LineStyle = '--';
hlines(2).LineWidth = 2;
hlines(2).LineStyle = '-.';
hlines(3).LineWidth = 2;
hlines(3).LineStyle = '-';
exportgraphics(gcf,'Outputs\Stage_Area_Volume_Time.pdf','ContentType','vector')


%% Functions
function [ax,hlines] = plotyyy(x1,y1,x2,y2,x3,y3,ylabels)
%PLOTYYY - Extends plotyy to include a third y-axis
%
%Syntax:  [ax,hlines] = plotyyy(x1,y1,x2,y2,x3,y3,ylabels)
%
%Inputs: x1,y1 are the xdata and ydata for the first axes' line
%        x2,y2 are the xdata and ydata for the second axes' line
%        x3,y3 are the xdata and ydata for the third axes' line
%        ylabels is a 3x1 cell array containing the ylabel strings
%
%Outputs: ax -     3x1 double array containing the axes' handles
%         hlines - 3x1 double array containing the lines' handles
%
%Example:
%x=0:10; 
%y1=x;  y2=x.^2;   y3=x.^3;
%ylabels{1}='First y-label';
%ylabels{2}='Second y-label';
%ylabels{3}='Third y-label';
%[ax,hlines] = plotyyy(x,y1,x,y2,x,y3,ylabels);
%legend(hlines, 'y = x','y = x^2','y = x^3',2)
%
%m-files required: none

%Author: Denis Gilbert, Ph.D., physical oceanography
%Maurice Lamontagne Institute
%Dept. of Fisheries and Oceans Canada
%email: gilbertd@dfo-mpo.gc.ca  
%Web: http://www.qc.dfo-mpo.gc.ca/iml/
%April 2000; Last revision: 14-Nov-2001

if nargin==6
   %Use empty strings for the ylabels
   ylabels{1}=' '; ylabels{2}=' '; ylabels{3}=' ';
elseif nargin > 7
   error('Too many input arguments')
elseif nargin < 6
   error('Not enough input arguments')
end

figure('units','normalized',...
       'DefaultAxesXMinorTick','on','DefaultAxesYminorTick','on');

set(gca,'FontSize',14)
set(gcf,'units','inches','position',[5,0,5,4])
%Plot the first two lines with plotyy
[ax,hlines(1),hlines(2)] = plotyy(x1,y1,x2,y2);
xlabel('Elapsed Time (h)','Interpreter','latex')
cfig = get(gcf,'color');
pos = [0.1  0.1  0.7  0.8];
offset = pos(3)/5.5;

%Reduce width of the two axes generated by plotyy 
pos(3) = pos(3) - offset/2;
set(ax,'position',pos);  

%Determine the position of the third axes
pos3=[pos(1) pos(2) pos(3)+offset pos(4)];

%Determine the proper x-limits for the third axes
limx1=get(ax(1),'xlim');
limx3=[limx1(1)   limx1(1) + 1.2*(limx1(2)-limx1(1))];
%Bug fix 14 Nov-2001: the 1.2 scale factor in the line above
%was contributed by Mariano Garcia (BorgWarner Morse TEC Inc)

ax(3)=axes('Position',pos3,'box','off',...
   'Color','none','XColor','k','YColor','r',...   
   'xtick',[],'xlim',limx3,'yaxislocation','right');

hlines(3) = line(x3,y3,'Color','r','Parent',ax(3));
limy3=get(ax(3),'YLim');

%Hide unwanted portion of the x-axis line that lies
%between the end of the second and third axes
line([limx1(2) limx3(2)],[limy3(1) limy3(1)],...
   'Color',cfig,'Parent',ax(3),'Clipping','off');
axes(ax(2))

%Label all three y-axes
set(get(ax(1),'ylabel'),'string',ylabels{1},'interpreter','latex')
set(get(ax(2),'ylabel'),'string',ylabels{2},'interpreter','latex')
set(get(ax(3),'ylabel'),'string',ylabels{3},'interpreter','latex')
end

