%% Watershed Post-Processing
% 03/04/2023
% Developer: Marcus Nobrega
% Goal - Post Processing Results of Watershed
% New Adds: New Colormaps

%% ColorMaps

% Spectrum
RGB=[0.1127         0    0.3515
     0.2350         0    0.6663
     0.3536         0    1.0000
     0.4255         0    1.0000
     0.4384         0    1.0000
     0.3888         0    1.0000
     0.2074         0    1.0000
          0         0    1.0000
          0    0.4124    1.0000
          0    0.6210    1.0000
          0    0.7573    0.8921
          0    0.8591    0.6681
          0    0.9642    0.4526
          0    1.0000    0.1603
          0    1.0000         0
          0    1.0000         0
          0    1.0000         0
          0    1.0000         0
     0.4673    1.0000         0
     0.8341    1.0000         0
     1.0000    0.9913         0
     1.0000    0.8680         0
     1.0000    0.7239         0
     1.0000    0.5506         0
     1.0000    0.3346         0
     1.0000         0         0
     1.0000         0         0
     1.0000         0         0
     1.0000         0         0
     0.9033         0         0
     0.7412         0         0
     0.5902         0         0];
Spectrum=interp1(linspace(1, 256, 32),RGB,[1:1:256]);

%% Coordinates
x_grid = xulcorner + resolution*[1:1:size(DEM,2)]; y_grid = yulcorner - resolution*[1:1:size(DEM,1)];
%% Plotting GIFs - Depths
h = figure;
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
filename = 'Depths_Dynamics.gif';
a_grid = resolution;
b_grid = resolution;
date_records = time_begin; % Saving data
dmax = 0; infmax = 0;
for t = 1:size(d,2)
    % Time of Records in Date
    factor = record_time_maps/record_time_hydrographs;
    date_records = date_records + (factor)*record_time_hydrographs/24/60;
    % Draw plot
%     t_title = round(time_store(t)*time_step/60);
    t_title = date_records;
    z = reshape(d(:,t)/1000,[],size(DEM,2));
    idx = isnan(DEM);
    z(idx) = nan;
    dmax = max(dmax,z);
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    % UTM Coordinates
%     x_grid = [xbegin:1:xend]; y_grid = [ybegin:1:yend];
    h_min = 0;
    F = z;
    zmax = max(max(d(~isnan(d))))/1000;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 0.1;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    axis tight; grid on; box on;
    title(datestr(t_title),'Interpreter','Latex','FontSize',12)
    view(0,90)
    caxis([h_min zmax]);
    colormap(Spectrum)
    hold on
    k = colorbar ;
    ylabel(k,'Depths (m)','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('Elevation (m)','Interpreter','Latex','FontSize',12)
    drawnow
    % Capture the plot as an image
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if t == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    hold off
end
clf

%% Plotting Isoeital Maps
if flags.flag_spatial_rainfall == 1 && flags.flag_rainfall == 1
    ticksize = [0.01 0.005];
    set(gcf,'units','inches','position',[3,3,4,3])   
    % Sum all rainfall
    dt_rainfall = time_store_rainfall_time(3) - time_store_rainfall_time(2); % min
    rainfall_vol_spatial = sum(spatial_rainfall_maps,3)*dt_rainfall/60; % mm
    z = rainfall_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(Spectrum)
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    ylabel(k,'Rainfall Volume (mm)','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('Rainfall Volume (mm)','Interpreter','Latex','FontSize',12) 
end
exportgraphics(gcf,'Outputs\Isoietal_Map.TIF','ContentType','image','Colorspace','rgb','Resolution',300)

%% Plotting Isoeital Maps of ETP
if flags.flag_spatial_ETP == 1 && flags.flag_ETP == 1
    ticksize = [0.01 0.005];
    set(gcf,'units','inches','position',[3,3,4,3])   
    % Sum all rainfall
    dt_ETP = time_store_ETP_time(3) - time_store_ETP_time(2); % min
    ETR_vol_spatial = sum(ETR_save,3)*dt_ETP/60/24; % mm
    z = ETR_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(Spectrum)
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    ylabel(k,'ETR Volume (mm)','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('ETR Volume (mm)','Interpreter','Latex','FontSize',12) 
end
exportgraphics(gcf,'Outputs\Isoietal_Map_ETR.TIF','ContentType','image','Colorspace','rgb','Resolution',300)

%% Plotting Isoeital Maps of ETP
if flags.flag_spatial_ETP == 1 && flags.flag_ETP == 1
    ticksize = [0.01 0.005];
    set(gcf,'units','inches','position',[3,3,4,3])   
    % Sum all rainfall
    dt_ETP = time_store_ETP_time(3) - time_store_ETP_time(2); % min
    ETP_vol_spatial = sum(ETP_save,3)*dt_ETP/60/24; % mm
    z = ETP_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(Spectrum)
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    ylabel(k,'ETP Volume (mm)','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('ETP Volume (mm)','Interpreter','Latex','FontSize',12) 
end
exportgraphics(gcf,'Outputs\Isoietal_Map_ETP.TIF','ContentType','image','Colorspace','rgb','Resolution',300)

%% Plotting All Three Maps Together
if flags.flag_spatial_rainfall == 1 && flags.flag_rainfall == 1 && flags.flag_spatial_ETP == 1 && flags.flag_ETP == 1
    close all
    set(gcf,'units','inches','position',[3,0,9,8])       
    ax1 = subplot(2,3,1);
    ticksize = [0.01 0.005];
    % Sum all rainfall
    dt_rainfall = time_store_rainfall_time(3) - time_store_rainfall_time(2); % min
    rainfall_vol_spatial = sum(spatial_rainfall_maps,3)*dt_rainfall/60; % mm
    z = rainfall_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(ax1,pmkmp(256,'Edge'))
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    k.Location = 'northoutside';
    ylabel(k,'$\int{i(t)\mathrm{dt}}$ (mm)','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('$\int{i(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12) 
    set(gca,'fontsize',12);
    ax2 = subplot(2,3,2);
    ticksize = [0.01 0.005];  
    % Sum all rainfall
    dt_ETP = time_store_ETP_time(3) - time_store_ETP_time(2); % min
    ETR_vol_spatial = sum(ETR_save,3)*dt_ETP/60/24; % mm
    z = ETR_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(ax2,Spectrum)
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    k.Location = 'northoutside';
    ylabel(k,'$\int{e_{\mathrm{TR}}(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('$\int{e_{\mathrm{TR}}(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12)
    set(gca,'fontsize',12);
    ax3 = subplot(2,3,3);
    ticksize = [0.01 0.005];
    % Sum all rainfall
    dt_ETP = time_store_ETP_time(3) - time_store_ETP_time(2); % min
    ETP_vol_spatial = sum(ETP_save,3)*dt_ETP/60/24; % mm
    z = ETP_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(ax3,pmkmp(256,'Swtth'))
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    k.Location = 'northoutside';
    set(gca,'fontsize',12);
    ylabel(k,'$\int{e_{\mathrm{TP}}(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('$\int{e_{\mathrm{TP}}(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12)     

    % Plotting Hyeto and Hydrograph
    if flags.flag_spatial_rainfall == 1 && flags.flag_rainfall == 1
        n_cells = sum(sum(DEM > 0));
        avg_rainfall = squeeze(sum(sum((spatial_rainfall_maps)))/n_cells);
    end
    subplot(2,3,4:6)
    % Fluxes
    f_rate = zeros(1,size(I,2));
    for zzz = 1:(size(I,2))
        if zzz == 1
            f_rate(1,zzz) = 0;
        else
            Vol_begin = nansum(I(:,zzz-1))/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            Vol_end = nansum(I(:,zzz))/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            if Vol_end ~= Vol_begin
                ttt = 1;
            end
            f_rate(1,zzz) = (Vol_end - Vol_begin)/(record_time_maps/60); % mm/hr; % Soil flux
        end
    end
    flags.flag_date = 3; % 1 min, 2 hour, 3 day, 4 month
    date_string = {'Elased time (min)','Elapsed time (h)','Elapsed time (days)','Elapsed time (months)'};
    if flags.flag_date == 1
        time_scale = 1;
    elseif flags.flag_date == 2
        time_scale = 1/60;
    elseif flags.flag_date == 3
        time_scale = 1/60/24;
    else
        time_scale = 1/60/24/30;
    end     
    time_real = time_begin + time_records_hydrographs/60/24;
    c_hydrograph = [119,136,153]/255;
    plot(time_real,Qout_w(1:length(time_real)),'linewidth',2,'color',c_hydrograph);
    xlabel('Date','interpreter','latex');
    ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
    set(gca,'fontsize',12);
    grid on
    hold on
    yyaxis right
    set(gca,'ycolor','black');
    set(gca,'ydir','reverse')
    set(gca,'FontName','Garamond')    
    c_rainfall = [0,0,128]/255;
    time_avg_rainfall = time_begin + (1:dt_rainfall:tfinal)/60/24;
    plot(time_avg_rainfall,avg_rainfall(1:length(time_avg_rainfall)),'linewidth',2,'color',c_rainfall,'LineStyle','-');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    hold on
    c_fg = [244,164,96]/255;
    plot(time_records*time_scale,f_rate(1:length(time_records)),'linewidth',2,'color',c_fg,'LineStyle','-.');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    legend('$Q_{out}$','$\bar{i}$','$\bar{f_g}$','interpreter','latex')
    ylim([ceil(min(f_rate)/5)*6 300])
    grid on       
end
exportgraphics(gcf,'Outputs\Rain_ETP_ETR.TIF','ContentType','image','Colorspace','rgb','Resolution',300)

%% Plotting All Three Maps Together (PAPER)
if flags.flag_spatial_rainfall == 1 && flags.flag_rainfall == 1 && flags.flag_spatial_ETP == 1 && flags.flag_ETP == 1
    close all
    set(gcf,'units','inches','position',[3,0,9,8])       
    ax1 = subplot(2,3,1);
    ticksize = [0.01 0.005];
    % Sum all rainfall
    dt_rainfall = time_store_rainfall_time(3) - time_store_rainfall_time(2); % min
    rainfall_vol_spatial = sum(spatial_rainfall_maps,3)*dt_rainfall/60; % mm
    z = rainfall_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(ax1,pmkmp(256,'Edge'))
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    k.Location = 'northoutside';
    ylabel(k,'$\int{i(t)\mathrm{dt}}$ (mm)','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('$\int{i(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12) 
    set(gca,'fontsize',12);
    ax2 = subplot(2,3,2);
    ticksize = [0.01 0.005];  
    % Sum all rainfall
    dt_ETP = time_store_ETP_time(3) - time_store_ETP_time(2); % min
    ETR_vol_spatial = sum(ETR_save,3)*dt_ETP/60/24; % mm
    z = ETR_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(ax2,Spectrum)
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    k.Location = 'northoutside';
    ylabel(k,'$\int{e_{\mathrm{TR}}(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('$\int{e_{\mathrm{TR}}(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12)
    set(gca,'fontsize',12);
    ax3 = subplot(2,3,3);
    ticksize = [0.01 0.005];
    % Sum all rainfall
    dt_ETP = time_store_ETP_time(3) - time_store_ETP_time(2); % min
    ETP_vol_spatial = sum(ETP_save,3)*dt_ETP/60/24; % mm
    z = ETP_vol_spatial;
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(z)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
%     plot3(x_coordinate, y_coordinate,zmax*ones(n_raingauges), 'r.', 'MarkerSize', 30)
    hold on
    h_min = min(min(z));
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    ax = gca;
    ax.FontName = 'Garamond';
    axis tight; grid on; box on;
    view(0,90)
    caxis([h_min zmax]);
    colormap(ax3,pmkmp(256,'Swtth'))
    hold on
    k = colorbar ;
    k.FontName = 'Garamond';
    k.TickDirection = 'out';
    k.TickLength = ticksize;
    k.Location = 'northoutside';
    set(gca,'fontsize',12);
    ylabel(k,'$\int{e_{\mathrm{TP}}(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('$\int{e_{\mathrm{TP}}(t)\mathrm{dt}}$','Interpreter','Latex','FontSize',12)     

    % Plotting Hyeto and Hydrograph
    if flags.flag_spatial_rainfall == 1 && flags.flag_rainfall == 1
        n_cells = sum(sum(DEM > 0));
        avg_rainfall = squeeze(sum(sum((spatial_rainfall_maps)))/n_cells);
    end
    subplot(2,3,4:6)
    % Fluxes
    f_rate = zeros(1,size(I,2));
    for zzz = 1:(size(I,2))
        if zzz == 1
            f_rate(1,zzz) = 0;
        else
            Vol_begin = nansum(I(:,zzz-1))/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            Vol_end = nansum(I(:,zzz))/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            if Vol_end ~= Vol_begin
                ttt = 1;
            end
            f_rate(1,zzz) = (Vol_end - Vol_begin)/(record_time_maps/60); % mm/hr; % Soil flux
        end
    end   

flags.flag_date = 3; % 1 min, 2 hour, 3 day, 4 month
if flags.flag_date == 1
    time_scale = 1;
elseif flags.flag_date == 2
    time_scale = 1/60;
elseif flags.flag_date == 3
    time_scale = 1/60/24;
else
    time_scale = 1/60/24/30;
end     
    
    c_hydrograph = [119,136,153]/255;
    plot(time_real,Qout_w(1:length(time_real)),'linewidth',2,'color',c_hydrograph);
    xlabel('Date','interpreter','latex');
    ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
    set(gca,'fontsize',12);
    grid on
    hold on
    yyaxis right
    set(gca,'ycolor','black');
    set(gca,'ydir','reverse')
    set(gca,'FontName','Garamond')    
    c_rainfall = [0,0,128]/255;
    time_avg_rainfall = time_begin + (1:dt_rainfall:tfinal)/60/24;
    plot(time_avg_rainfall,avg_rainfall(1:length(time_avg_rainfall)),'linewidth',2,'color',c_rainfall,'LineStyle','-');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    hold on
    c_fg = [244,164,96]/255;
    plot(time_records*time_scale,f_rate(1:length(time_records)),'linewidth',2,'color',c_fg,'LineStyle','-.');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    legend('$Q_{out}$','$\bar{i}$','$\bar{f_g}$','interpreter','latex')
    ylim([ceil(min(f_rate)/5)*6 300])
    grid on     


    % Insert Graphs - Hydrograph Detail
    axes('Position',[.7 .15 .2 .2])
    plot(time_real,Qout_w(1:length(time_real)),'linewidth',2,'color',c_hydrograph);
    xlabel('Date','interpreter','latex');
    ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
    set(gca,'fontsize',12);
    grid on
    hold on
    yyaxis right
    set(gca,'ycolor','black');
    set(gca,'ydir','reverse')
    set(gca,'FontName','Garamond')    
    c_rainfall = [0,0,128]/255;
    plot(time_avg_rainfall,avg_rainfall(1:length(time_avg_rainfall)),'linewidth',2,'color',c_rainfall,'LineStyle','-');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    hold on
    c_fg = [244,164,96]/255;
    plot(time_records*time_scale,f_rate(1:length(time_records)),'linewidth',2,'color',c_fg,'LineStyle','-.');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    legend('$Q_{out}$','$\bar{i}$','$\bar{f_g}$','interpreter','latex')
    ylim([ceil(min(f_rate)/5)*6 300])
    grid on        

    % Insert Chart - Percentile
    axes('Position',[.2 .15 .2 .2])
    x_prct = [98:0.01:100];
    x_prct_return_period = 100 - x_prct;
    prct_values_flow = prctile(Qout_w,x_prct);
    prct_values_rain = prctile(avg_rainfall,x_prct);
    plot(x_prct_return_period,prct_values_flow,'color',c_hydrograph,'linewidth',2,'linestyle','--')
    hold on
    plot(x_prct_return_period,prct_values_rain,'color',c_rainfall,'linewidth',2,'linestyle','-.')
    set(gca,'FontName','Garamond')
    set(gca,'Fontsize',12)
    xlabel('Prob($x \leq x^*$) (\%)','interpreter','latex');
    ylabel('x','interpreter','latex');
end
exportgraphics(gcf,'Outputs\Rain_ETP_ETR.TIF','ContentType','image','Colorspace','rgb','Resolution',300)

%% Plotting Spatial Rainfall
if flags.flag_spatial_rainfall == 1
h = figure;
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
filename = 'Spatial_Rainfall.gif';
a_grid = resolution;
b_grid = resolution;
date_records = time_begin; % Saving data
time_step_rainfall = time_store_rainfall_time(3) - time_store_rainfall_time(2);
for t = 1:size(spatial_rainfall_maps,3)
    % Time of Records in Date
    factor = 1;
    date_records = date_records + (factor)*time_step_rainfall/24/60;
    % Draw plot
%     t_title = round(time_store(t)*time_step/60);
    t_title = date_records;
    z = spatial_rainfall_maps(:,:,t);
    idx = isnan(DEM);
    z(idx) = nan;
    zmax = max(max(max(spatial_rainfall_maps)));
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % Rain Gauges
    rainfall = Rainfall_Properties.rainfall_raingauges(t,1:Rainfall_Properties.n_raingauges)';
    plot3(x_coordinate, y_coordinate,zmax*ones(size(rainfall)), 'r.', 'MarkerSize', 30)
    hold on
    % UTM Coordinates
%     x_grid = [xbegin:1:xend]; y_grid = [ybegin:1:yend];
    h_min = 0;
    F = z;
    if isempty(zmax) || isinf(zmax) || zmax == 0
        zmax = 10;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    axis tight; grid on; box on;
    title(datestr(t_title),'Interpreter','Latex','FontSize',12)
    view(0,90)
    caxis([h_min zmax]);
    colormap(Spectrum)
    hold on
    k = colorbar ;
    ylabel(k,'Rainfall Intensity (mm/h)','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('Rainfall Intensity (mm/h)','Interpreter','Latex','FontSize',12)
    drawnow
    % Capture the plot as an image
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if t == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    hold off
end
clf
end

%% Plotting Infiltration
close all
h = figure;
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
filename = 'Infiltration_Dynamics.gif';
a_grid = resolution;
b_grid = resolution;
date_records = time_begin; % Saving data
for t = 1:size(d,2)
    % Time of Records in Date
    factor = record_time_maps/record_time_hydrographs;
    date_records = date_records + (factor)*record_time_hydrographs/24/60; 
    % Draw plot
%     t_title = round(time_store(t)*time_step/60);
    t_title = date_records;
    z = reshape(I(:,t),[],size(DEM,2));
    infmax = max(infmax,z);
    idx = isnan(DEM);
    z(idx) = nan;
    xmax = length(z(1,:));
    xend = xmax;
    ymax = length(z(:,1));
    yend = ymax;
    xbegin = 1;
    ybegin = 1;
    % UTM Coordinates
%     x_grid = [xbegin:1:xend]; y_grid = [ybegin:1:yend];
    h_min = 0;
    F = z;
    zmax = max(max(I(~isnan(I))));
    if isempty(zmax) || isinf(zmax) || isnan(zmax)
        zmax = 0.1;
    end
    map = surf(x_grid,y_grid,F);
    set(map,'LineStyle','none')
    axis tight; grid on; box on;
    title(datestr(t_title),'Interpreter','Latex','FontSize',12)
    view(0,90)
    colorbar
    caxis([h_min zmax]);
    colormap(Spectrum)
    hold on
    k = colorbar ;
    ylabel(k,'Infiltration (mm)','Interpreter','Latex','FontSize',12)
    xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
    ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
    zlabel ('Infiltration (mm)','Interpreter','Latex','FontSize',12)
    drawnow
    % Capture the plot as an image
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if t == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    hold off
end
clf


%% Exporting Max_Depth Raster
Raster_Max_Depth = DEM_raster;
Raster_Max_Depth.Z = reshape(max_depth/1000,[],size(DEM,2)); % meters
GRIDobj2geotiff(Raster_Max_Depth,'Maximum Depths')


%% Max Depth and Inf max
ticksize = [0.04 0.03];
set(gcf,'units','inches','position',[0,1,6.5,6])
if flags.flag_ETP == 1 && flags.flag_rainfall == 1
    ax1 = subplot(2,2,1);
elseif flags.flag_ETP == 0 && flags.flag_rainfall == 1
    ax1 = subplot(1,2,1)
end
zzz = dmax;
zzz(zzz==0) = nan;
% zzz(zzz<0.001) = nan;
surf(x_grid,y_grid,zzz);
zmin = min(min(zzz)); zmax = max(max(zzz));
xmax = length(zzz(1,:));
xend = xmax;    
ymax = length(zzz(:,1));
yend = ymax;
xbegin = 1;
ybegin = 1;
a_grid = resolution;
% UTM Coordinates
% x_grid = [xbegin:1:xend]; y_grid = [ybegin:1:yend];
if isempty(zmax) || isinf(zmax) || isnan(zmax)
        zmax = 0.1;
end
axis tight; grid on; box on;
set(gca, 'YDir','reverse')
xlabel('x (m)','interpreter','latex')
xlabel('y (m)','interpreter','latex')
set(gca,'ticklength',ticksize);
box on
view(0,90);
color = colormap(ax1,pmkmp(25,'Edge'));
h = colorbar('northoutside');
h.Label.String = 'Max. Depth (m)';
h.Label.Interpreter = 'Latex';
hold on
set(gca,'fontsize',12); shading interp;
hold off

if flags.flag_ETP == 1 && flags.flag_rainfall == 1
    ax2 = subplot(2,2,2);
elseif flags.flag_ETP == 0 && flags.flag_rainfall == 1
    ax2 = subplot(1,2,2);
end

zzz = infmax; % mm
zzz(zzz<=5) = nan;
zzz = zzz/1000; % m
surf(x_grid,y_grid,zzz);
set(gca,'ticklength',ticksize);
zmin = min(min(zzz)); zmax = max(max(zzz));
xmax = length(zzz(1,:));
xend = xmax;
ymax = length(zzz(:,1));
yend = ymax;
xbegin = 1;
ybegin = 1;
a_grid = resolution;
% UTM Coordinates
% x_grid = [xbegin:1:xend]; y_grid = [ybegin:1:yend];
axis tight; grid on; box on;
set(gca, 'YDir','reverse')
xlabel('x (m)','interpreter','latex')
xlabel('y (m)','interpreter','latex')
view(0,90);
color = colormap(ax2,pmkmp(25,'Swtth'));
h = colorbar('northoutside');
h.Label.String = 'Max. Soil Depth (m)';
h.Label.Interpreter = 'Latex';
hold on
set(gca,'fontsize',12); shading interp;
box on
hold off

if flags.flag_ETP == 1 && flags.flag_rainfall == 1 && flags.flag_spatial_ETP ~= 1
    ax3 = subplot(2,2,3);
    % ETP + fG
    factor = record_time_ETP/record_time_hydrographs;
    for zzz = 1:(length(ETP_saving))
        ETP_daily(zzz,1) = ETP_saving(zzz,1);
        f_g_daily(zzz,1) = nansum(k_out*24.*idx_cells)/(sum(idx_cells)); % mm/day
    end

    data_ETP = readtable('Input_Data.xlsx','Sheet','Concentrated_ETP_Data'); % Input ETP data to interpolate from IDW method.
    n_obs_ETP_conc = sum((table2array(data_ETP(:,3))>=0)); % Number of observations   
    time_ETP = datetime(datestr(table2array(data_ETP(3:n_obs_ETP_conc,2))));
    plot(time_ETP(1:length(ETP_daily)),ETP_daily(1:length(ETP_daily)),'linewidth',1,'color','black','marker','*');
    hold on
    plot(time_ETP(1:length(f_g_daily)),f_g_daily(1:length(f_g_daily)),'linewidth',1,'color','red');
    xlabel('Date','interpreter','latex');
    ylabel('$E_{\mathrm{to}}$ and $f_g$ ($\mathrm{mm/day}$)','interpreter','latex');
    set(gca,'fontsize',12);
    set(gca,'ticklength',ticksize);
    legend('$E_{to}$','$f_g$','interpreter','latex')
    grid on
    hold on    
end



if flags.flag_ETP == 1 && flags.flag_rainfall == 1 && flags.flag_spatial_ETP ~= 1
    ax4 = subplot(2,2,4);
    tempmed = table2array(data_ETP(3:n_obs_ETP_conc,3));
    tempmax = table2array(data_ETP(3:n_obs_ETP_conc,4));
    tempmin = table2array(data_ETP(3:n_obs_ETP_conc,5));
    windspeed = table2array(data_ETP(3:n_obs_ETP_conc,6));
    relative_h = table2array(data_ETP(3:n_obs_ETP_conc,7))/100;
    G = table2array(data_ETP(3:n_obs_ETP_conc,8));
    
    % plot 
    plot(time_ETP,tempmed,'linewidth',.5,'color','green','marker','*','linestyle','-');
    hold on
    plot(time_ETP,tempmax,'linewidth',.5,'color','green','marker','+','linestyle','-');
    hold on
    plot(time_ETP,tempmin,'linewidth',.5,'color','green','marker','x','linestyle','-');
    hold on
    plot(time_ETP,windspeed,'linewidth',.5,'color','blue','marker','.','linestyle','-');
    hold on
    ylabel('$t$ ($^{\circ}$  ($C$) or $u_2$ ($\mathrm{m/s}$)','interpreter','latex')
    yyaxis right
    set(gca,'ycolor','black')
    ylabel('$u_r (-)$ or $g_r$ ($\mathrm{MJ m^{-2} day^{-1}}$)','interpreter','latex')
    plot(time_ETP,relative_h,'linewidth',.5,'color','blue','marker','*','linestyle','--');
    hold on
    plot(time_ETP,G,'linewidth',.5,'color','red','marker','*');
    set(gca,'fontsize',12);
    set(gca,'ticklength',ticksize);
    legend('$t_{\mathrm{med}}$','$t_{\mathrm{max}}$','$t_{\mathrm{min}}$','$u_{\mathrm{2}}$','$u_{\mathrm{r}}$','$g_r$','interpreter','latex')
    grid on    
end

exportgraphics(gcf,'Outputs\Summary.TIF','ContentType','image','Colorspace','rgb','Resolution',300)

%% Hydrographs - Elapsed Time
% Date Scale
close all
flags.flag_date = 3; % 1 min, 2 hour, 3 day, 4 month
date_string = {'Elased time (min)','Elapsed time (h)','Elapsed time (days)','Elapsed time (months)'};
if flags.flag_date == 1
    time_scale = 1;
elseif flags.flag_date == 2
    time_scale = 1/60;
elseif flags.flag_date == 3
    time_scale = 1/60/24;
else
    time_scale = 1/60/24/30;
end
%     t_title = time_real(t);
set(gcf,'units','inches','position',[0,1,6.5,4])
plot(time_records_hydrographs*time_scale,Qout_w(1:length(time_records_hydrographs)),'linewidth',2,'color','black');
xlabel(date_string(flags.flag_date),'interpreter','latex');
ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
set(gca,'fontsize',14);
grid on
hold on
yyaxis right
set(gca,'ycolor','black');
plot(time_records_hydrographs*time_scale,Depth_out(1:length(time_records_hydrographs)),'linewidth',2,'color','black','LineStyle','--');
ylabel('Depth ($\mathrm{m}$)','interpreter','latex');
grid on
exportgraphics(gcf,'Outputs\Hydrograph Watershed.pdf','ContentType','vector')

%% Rainfall and Flow Discharge
close all
set(gcf,'units','inches','position',[0,1,6.5,4])
plot(time_records_hydrographs*time_scale,Qout_w(1:length(time_records_hydrographs)),'linewidth',2,'color','black');
xlabel(date_string(flags.flag_date),'interpreter','latex');
ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
set(gca,'fontsize',14);
grid on
hold on
yyaxis right
set(gca,'ycolor','black');
set(gca,'ydir','reverse')
plot(time_save/24/60,rain_outlet(1:length(time_save)),'linewidth',2,'color','blue','LineStyle','-');
ylabel('Rainfall Intensity ($\mathrm{mm/h}$)','interpreter','latex');
ylim([0 300])
grid on
exportgraphics(gcf,'Outputs\Hydrograph_Rainfall.pdf','ContentType','vector')

%% Hydrographs - Real-Time
% Date Scale
time_real = time_begin + time_records_hydrographs/60/24;
close all
set(gcf,'units','inches','position',[0,1,6.5,4])
plot(time_real,Qout_w(1:length(time_real)),'linewidth',2,'color','black');
xlabel('Date','interpreter','latex');
ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
set(gca,'fontsize',14);
grid on
hold on
yyaxis right
set(gca,'ycolor','black');
plot(time_real,Depth_out(1:length(time_real)),'linewidth',2,'color','black','LineStyle','--');
ylabel('Depth ($\mathrm{m}$)','interpreter','latex');
grid on
legend('Flow','Depth','interpreter','latex')
exportgraphics(gcf,'Outputs\Hydrograph Watershed.pdf','ContentType','vector')

%% Rainfall and Flow Discharge - Real Time
close all
set(gcf,'units','inches','position',[0,1,6.5,4])
time_real = time_begin + time_records_hydrographs/60/24;
time_real_rainfall = time_begin + time_save/60/24;
plot(time_real,Qout_w(1:length(time_real)),'linewidth',2,'color','black');
xlabel('Date','interpreter','latex');
ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
set(gca,'fontsize',14);
grid on
hold on
yyaxis right
set(gca,'ycolor','black');
set(gca,'ydir','reverse')
plot(time_real_rainfall,rain_outlet(1:length(time_real_rainfall)),'linewidth',2,'color','blue','LineStyle','-');
ylabel('Rainfall Intensity ($\mathrm{mm/h}$)','interpreter','latex');
ylim([0 300])
grid on
exportgraphics(gcf,'Outputs\Hydrograph_Rainfall.pdf','ContentType','vector')

%% Climatologic Data Summary

if flags.flag_spatial_ETP ~=1 && flags.flag_ETP == 1
    % clear all;
    % load workspace_3months_1dayleft.mat
    close all
    % Date ETP
    factor = record_time_ETP/record_time_hydrographs;
    if flags.flag_ETP == 0
    for zzz = 1:n_obs_model
        ETP_daily(zzz,1) = ETP_saving(zzz,1);
        f_g_daily(zzz,1) = nansum(k_out*24.*idx_cells)/(sum(idx_cells)); % mm/day
    end
    set(gcf,'units','inches','position',[0,1,6.5,4])
    plot(time_ETP,ETP_daily(1:length(time_ETP)),'linewidth',2,'color','black','marker','*');
    hold on
    plot(time_ETP,f_g_daily(1:length(time_ETP)),'linewidth',2,'color','red');
    xlabel('Date','interpreter','latex');
    ylabel('$E_{\mathrm{to}}$ and $f_g$ ($\mathrm{mm/day}$)','interpreter','latex');
    set(gca,'fontsize',14);
    grid on
    hold on
    
    yyaxis right
    set(gca,'ycolor','black');
    set(gca,'ydir','reverse')
    plot(time_real_rainfall,rain_outlet,'linewidth',2,'color','blue','LineStyle','-');
    ylabel('Rainfall Intensity ($\mathrm{mm/h}$)','interpreter','latex');
    ylim([0 300])
    grid on
    legend('ETP','$f_g$','Rainfall','interpreter','latex')
    
    exportgraphics(gcf,'Outputs\Climatologic_Summary.pdf','ContentType','vector')
    % 
    end
    %% Rainfall + Infiltration at the Outlet - Real Time
    % We want to calculate an average infiltration rate for the watershed
    %%%% ---------- Average Infiltration Rate for the Maps time-step ----------
    f_rate = zeros(1,size(I,2));
    for zzz = 1:(size(I,2))
        if zzz == 1
            f_rate(1,zzz) = 0;
        else
            Vol_begin = nansum(I(:,zzz-1))/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            Vol_end = nansum(I(:,zzz))/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            if Vol_end ~= Vol_begin
                ttt = 1;
            end
            f_rate(1,zzz) = (Vol_end - Vol_begin)/(record_time_maps/60); % mm/hr; % Soil flux
        end
    end
    set(gcf,'units','inches','position',[0,1,6.5,4])
    plot(time_records_hydrographs*time_scale,Qout_w(1:length(time_records_hydrographs)),'linewidth',2,'color','black');
    xlabel(date_string(flags.flag_date),'interpreter','latex');
    ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
    set(gca,'fontsize',14);
    grid on
    hold on
    yyaxis right
    set(gca,'ycolor','black');
    set(gca,'ydir','reverse')
    plot(time_save/60/24,rain_outlet(1:length(time_save)),'linewidth',2,'color','blue','LineStyle','-');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    hold on
    plot(time_records*time_scale,f_rate(1:length(time_records)),'linewidth',2,'color','red','LineStyle','-.');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    legend('Outlet Flow','Rainfall','Mean Soil Flux')
    ylim([ceil(min(f_rate)/5)*6 300])
    grid on
    exportgraphics(gcf,'Outputs\Hydrograph_Effective_Rainfall.pdf','ContentType','vector')
    
    %%
    %%%% ---------- Average Infiltration at Hydrograph Times ---------- %%%%
    close all
    set(gcf,'units','inches','position',[0,1,6.5,4])
    plot(time_real,Qout_w(1:length(time_real)),'linewidth',2,'color','black');
    xlabel('Date','interpreter','latex');
    ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
    set(gca,'fontsize',14);
    grid on
    hold on
    yyaxis right
    set(gca,'ycolor','black');
    set(gca,'ydir','reverse')
    plot(time_real_rainfall,rain_outlet(1:length(time_real_rainfall)),'linewidth',2,'color','blue','LineStyle','-');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    hold on
    plot(time_records*time_scale,f_rate(1:length(time_records)),'linewidth',2,'color','red','LineStyle','-.');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    legend('Outlet Flow','Rainfall','Mean Soil Flux','interpreter','latex')
    ylim([ceil(min(f_rate)/5)*6 300])
    grid on
    exportgraphics(gcf,'Outputs\Hydrograph_Effective_Rainfall_Detail.pdf','ContentType','vector')
    % 
    %%
    %% Rainfall + Infiltration at the Outlet 
    % We want to calculate an average infiltration rate for the watershed
    %%%% ---------- Average Infiltration Rate for the Maps time-step ----------
    f_rate = zeros(1,size(I,2));
    for zzz = 1:(size(I,2))
        if zzz == 1
            f_rate(1,zzz) = 0;
        else
            Vol_begin = nansum(I(:,zzz-1))/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            Vol_end = nansum(I(:,zzz))/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            if Vol_end ~= Vol_begin
                ttt = 1;
            end
            f_rate(1,zzz) = (Vol_end - Vol_begin)/(record_time_maps/60); % mm/hr; % Soil flux
        end
    end
    set(gcf,'units','inches','position',[0,1,6.5,4])
    plot(time_records_hydrographs*time_scale,Qout_w(1:length(time_records_hydrographs)),'linewidth',2,'color','black');
    xlabel(date_string(flags.flag_date),'interpreter','latex');
    ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
    set(gca,'fontsize',14);
    grid on
    hold on
    yyaxis right
    set(gca,'ycolor','black');
    set(gca,'ydir','reverse')
    plot([0:time_step:tfinal*60]/60/60/24,rain_outlet(1:length(time_real_rainfall)),'linewidth',2,'color','blue','LineStyle','-');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    hold on
    plot(time_records*time_scale,f_rate(1:length(time_records)),'linewidth',2,'color','red','LineStyle','-.');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    legend('Outlet Flow','Rainfall','Mean Soil Flux')
    ylim([ceil(min(f_rate)/5)*6 300])
    grid on
    exportgraphics(gcf,'Outputs\Hydrograph_Effective_Rainfall.pdf','ContentType','vector')
    
    %%
    % %%%% ---------- Average Infiltration at Hydrograph Times ---------- %%%%
    close all
    set(gcf,'units','inches','position',[0,1,6.5,4])
    plot(time_records_hydrographs*time_scale,Qout_w(1:length(time_records_hydrographs)),'linewidth',2,'color','black');
    xlabel(date_string(flags.flag_date),'interpreter','latex');
    ylabel('Flow discharge ($\mathrm{m^3/s}$)','interpreter','latex');
    set(gca,'fontsize',14);
    grid on
    hold on
    yyaxis right
    set(gca,'ycolor','black');
    set(gca,'ydir','reverse')
    plot([0:time_step:tfinal*60]/60/60/24,rain_outlet,'linewidth',2,'color','blue','LineStyle','-');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    hold on
    plot(time_records*time_scale,f_rate(1:length(time_records)),'linewidth',2,'color','red','LineStyle','-.');
    ylabel('Rate ($\mathrm{mm/h}$)','interpreter','latex');
    legend('Outlet Flow','Rainfall','Mean Soil Flux')
    ylim([ceil(min(f_rate)/5)*6 300])
    grid on
    exportgraphics(gcf,'Outputs\Hydrograph_Effective_Rainfall.pdf','ContentType','vector')
    % 
    % %% Soil Content at Specifc Cell - Real Time
    % close all
    % 
    % % Time of Records in Date
    % for zzz = 1:(size(I,2))
    %     factor = record_time_maps/record_time_hydrographs;
    %     if zzz == 1
    %        date_records_maps(zzz,1) =  time_real(1); % Saving data
    %     else
    %        date_records_maps(zzz,1) = date_records_maps(zzz-1,1) + (factor)*record_time_hydrographs/24/60;  % Saving data
    %     end
    % end
    % cell_coord = pos; % If you want at the outlet, assume it as "pos"
    % set(gca,'ycolor','black');
    % plot(date_records_maps,I(pos,1:length(date_records_maps)),'linewidth',2,'color','black','LineStyle','-.');
    % ylabel('Depth ($\mathrm{mm}$)','interpreter','latex');
    % ylim([ceil(min(I(pos,:))/5)*4 ceil(max(I(pos,:)/5*8))]);
    % grid on
    % hold on
    % yyaxis right
    % set(gca,'ydir','reverse','ycolor','black')
    % % Rainfall Volume
    % rain_cumulative_pos = cumsum(i(pos,:))*record_time_hydrographs/60;
    % % ETP
    % etp_cumulative_pos = cumsum(ETP_saving/24)*record_time_hydrographs/60;
    % plot(time_real,rain_cumulative_pos(1,1:length(time_real)),'linewidth',2,'color','blue','linestyle','--');
    % hold on
    % plot(time_real,etp_cumulative_pos(1:length(time_real),1),'linewidth',2,'color','green','linestyle','-.');
    % ylim([ceil(min(rain_cumulative_pos)/5)*4 ceil(max(rain_cumulative_pos/5*12))]);
    % ylabel('Cumulative Rainfall and ETP ($\mathrm{mm}$)','interpreter','latex')
    % xlabel('Date','interpreter','latex');
    % lgd = legend('Infiltrated Depth','Cumulative Rainfall','Cumulative ETP','location','northoutside');
    % lgd.NumColumns = 3;
    % exportgraphics(gcf,'Infiltrated_Depth.pdf','ContentType','vector')
    
    %% Soil Content at Specifc Cell
    % close all
    % cell_coord = pos; % If you want at the outlet, assume it as "pos"
    % set(gca,'ycolor','black');
    % plot(time_records*time_scale,I(pos,1:length(time_store)),'linewidth',2,'color','black','LineStyle','-.');
    % ylabel('Depth ($\mathrm{mm}$)','interpreter','latex');
    % ylim([ceil(min(I(pos,:))/5)*4 ceil(max(I(pos,:)/5*8))]);
    % grid on
    % hold on
    % yyaxis right
    % set(gca,'ydir','reverse','ycolor','black')
    % % Rainfall Volume
    % rain_cumulative_pos = cumsum(i(pos,:))*record_time_hydrographs/60;
    % % ETP
    % etp_cumulative_pos = cumsum(ETP_saving/24)*record_time_hydrographs/60;
    % plot(time_records_hydrographs*time_scale,rain_cumulative_pos,'linewidth',2,'color','blue','linestyle','--');
    % hold on
    % plot(time_records_hydrographs*time_scale,etp_cumulative_pos,'linewidth',2,'color','green','linestyle','-.');
    % ylim([ceil(min(rain_cumulative_pos)/5)*4 ceil(max(rain_cumulative_pos/5*12))]);
    % ylabel('Cumulative Rainfall and ETP ($\mathrm{mm}$)','interpreter','latex')
    % xlabel(date_string(flags.flag_date),'interpreter','latex');
    % lgd = legend('Infiltrated Depth','Cumulative Rainfall','Cumulative ETP','location','northoutside');
    % lgd.NumColumns = 3;
    % exportgraphics(gcf,'Infiltrated_Depth.pdf','ContentType','vector')
end