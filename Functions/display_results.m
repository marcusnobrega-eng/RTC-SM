%% Plotting Results
% Developer - Marcus Nobrega
% 8/1/2021
% Objective: Plot results from the simulation

% Flows
time_plot = [0:time_step:(n_steps-1)*time_step]/60/60; % min
dim = min(length(out_r_final),length(time_plot));
subplot(3,1,1)
set(gcf,'units','inches','position',[5,0,7,12])
plot(time_plot(1:dim),Qout_w(1:dim),'green','LineWidth',1)
hold on
plot(time_plot(1:dim),out_r_final(1:dim),'red','LineWidth',1)
hold on
plot(time_plot(1:dim),out_c_final(1:dim),'blue','LineWidth',1)
xlabel('Elapsed time (hr)','Interpreter','Latex')
ylabel('Flow ($m^3/s$)','Interpreter','Latex')
yyaxis right
plot(time_plot(1:dim),i_reservoir(1:dim),'blue')
set(gca,'FontSize',12)
set(gca,'ydir','reverse')
set(gca,'ycolor','black')
ylim([0,400])
ylabel('Rainfall (mm/h)','Interpreter','Latex')
legend('Watershed','Reservoir','Channel','Rainfall','Interpreter','Latex')
subplot(3,1,2)
% Water Levels
plot(time_plot(1:dim),h_r_final(1:dim),'red','LineWidth',1)
hold on
plot(time_plot(1:dim),max(h_c_final(:,1:dim)),'blue','LineWidth',1)
hold on
xlabel('Elapsed Time (hr)','FontSize',12,'Interpreter','Latex'); ylabel('Water Surface Depth (m)','FontSize',12,'Interpreter','Latex');
legend('show')
legend('Reservoir','Channel');
set(gca,'FontSize',12)
subplot(3,1,3)
% Controls
% z = length(u);
% u(z+1) = u(z-1);
plot(time_plot(1:dim),u(1:dim),'red','LineWidth',1)
set(gca,'FontSize',12)
xlabel('Elapssed Time (hr)','Interpreter','Latex')
ylabel('Control Signal','Interpreter','Latex')

% Exporting Chart
exportgraphics(gcf,'RTC_Charts.pdf','ContentType','vector')
if flag_save_data == 1
    % Exporting txt files
    Reservoir_Outflow = table((time_plot(1:dim)'),(out_r_final(1:dim)),'VariableNames',{'Time (hr)','Reservoir Outflow (m3/s)'});
    writetable(Reservoir_Outflow)
    Watershed_Outflow = table((time_plot(1:dim)'),(Qout_w(1:dim)),'VariableNames',{'Time (hr)','Watershed Outflow (m3/s)'});
    writetable(Watershed_Outflow)
    Channel_Level = table((time_plot(1:dim)'),(max(h_c_final(:,1:dim)))','VariableNames',{'Time (hr)','Channel Surface Depth (m)'});
    writetable(Channel_Level)
    Reservoir_Level = table((time_plot(1:dim)'),(h_r_final(1:dim)),'VariableNames',{'Time (hr)','Channel Surface Depth (m)'});
    writetable(Reservoir_Level)
    Channel_Outflow = table((time_plot(1:dim)'),(out_c_final(1:dim)),'VariableNames',{'Time (hr)','Channel Outflow (m3/s)'});
    writetable(Channel_Outflow)
end
close all
% Hydrographs
time_plot = [0:time_step:(n_steps-1)*time_step]/60/60; % min
set(gcf,'units','inches','position',[5,0,5,3])
plot(time_plot(1:dim),Qout_w(1:dim),'green','LineWidth',1)
hold on
plot(time_plot(1:dim),out_r_final(1:dim),'red','LineWidth',1)
hold on
plot(time_plot(1:dim),out_c_final(1:dim),'blue','LineWidth',1)
hold on
plot(time_plot(1:dim),out_c_final(1:dim),'black','LineWidth',1)
xlabel('Elapsed time (hr)','Interpreter','Latex')
ylabel('Flow ($m^3/s$)','Interpreter','Latex')
yyaxis right
plot(time_plot(1:dim),i_reservoir(1:dim),'blue')
set(gca,'FontSize',12)
set(gca,'ydir','reverse')
set(gca,'ycolor','black')
ylim([0,400])
ylabel('Rainfall (mm/h)','Interpreter','Latex')
legend('Watershed','Reservoir','Channel','Rainfall','Interpreter','Latex')
exportgraphics(gcf,'Hydrographs.pdf','ContentType','vector')
close all