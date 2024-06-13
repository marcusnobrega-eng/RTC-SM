%% Watershed Physic Infomration
% Marcus Nóbrega & Matheus Schroeder 
% Goal: Estiamate watershed properties
% Last Modified: 09/2022

try 
    mkdir Outputs\
end

try
    mkdir Outputs\Input_Data
end

%% Data in raster format (GIS Data)

% Rasters
LULC_raster = GRIDobj(LULC_path); % Land Use and Land Cover Classification
DEM_raster = GRIDobj(DEM_path); % Digital Elevation Model
SOIL_raster = GRIDobj(SOIL_path); % Soil Map

%% Treating Values with Issues

% Extent Problem
if sum(size(DEM_raster.Z)) >= sum(size(LULC_raster.Z)) && sum(size(DEM_raster.Z)) > sum(size(SOIL_raster.Z)) % DEM is larger
    raster_resample = DEM_raster;
    % Resample other two rasters
    LULC_raster = resample(LULC_raster,raster_resample);
    SOIL_raster = resample(SOIL_raster,raster_resample);
end

if sum(size(SOIL_raster.Z)) >= sum(size(DEM_raster.Z)) && sum(size(SOIL_raster.Z)) > sum(size(LULC_raster.Z))  % SOIL is larger
    raster_resample = DEM_raster;
    % Resample other two rasters
    LULC_raster = resample(LULC_raster,raster_resample);
    DEM_raster = resample(DEM_raster,raster_resample);
end

if sum(size(LULC_raster.Z)) >= sum(size(DEM_raster.Z)) && sum(size(LULC_raster.Z)) > sum(size(SOIL_raster.Z))  % SOIL is larger
    raster_resample = DEM_raster;
    % Resample other two rasters
    SOIL_raster = resample(SOIL_raster,raster_resample);
    DEM_raster = resample(DEM_raster,raster_resample);
end

% Raster Extent
xulcorner = DEM_raster.refmat(3,1); % Up Left Corner
yulcorner = DEM_raster.refmat(3,2);
Resolution = DEM_raster.refmat(2,1);
DEM = DEM_raster.Z;
idx_nan_dem = DEM <= 0;
DEM(idx_nan_dem) = nan;

% Fill
if flags.flag_fill  == 1
% Fillsinks
    DEM_filled = fillsinks(DEM_raster,0.001);
    DIFFDEM = DEM_filled - DEM_raster;
    DIFFDEM.Z(DIFFDEM.Z==0) = nan;
    DEM_raster = DEM_filled;
    DEM = DEM_raster.Z;
    if sum(sum(abs(DIFFDEM.Z(~isnan(DIFFDEM.Z))))) == 0
          warning('Your DEM seems to already be filled.')
    else
        imageschs(DEM_raster,DIFFDEM.Z);  
    end
end

% Final Resample
if flags.flag_resample == 1
    LULC_raster = resample(LULC_raster,GIS_Parameters.resolution_resampled,'nearest');
    SOIL_raster = resample(SOIL_raster,GIS_Parameters.resolution_resampled,'nearest');
    DEM_raster = resample(DEM_raster,GIS_Parameters.resolution_resampled,'bilinear');
    DEM = DEM_raster.Z;
end


%% Model Type

% Filling Sinks if flag_diffusive ~= 0
if flags.flag_diffusive == 0
    DEM_filled = fillsinks(DEM_raster); % Filling sinks to be able to run kinematic_wave
    DIFFDEM = DEM_filled - DEM;
    DEM_raster = DEM_filled; % New Filled DEM
    DEM = DEM_raster.Z;
    % imageschs(DEM_raster,DIFFDEM.Z);
end

%% DEM Smoothening
flags.flag_smoothening = input_smoothening_data(1,1);
if flags.flag_smoothening == 1
    [DEM_raster,~,S] = DEM_smoothening(DEM_raster,GIS_Parameters.min_area,flags.flag_trunk,GIS_Parameters.tau,GIS_Parameters.K_value);
end

%% DEM Imposmin

if flags.flag_imposemin == 1
    % Impose Mininum Slope
        FD = FLOWobj(DEM_raster);
        area_km2_impose = 1*(Resolution^2)/1000/1000; % km2
        area_cells_impose = area_km2_impose./((DEM_raster.cellsize/1000)^2); % pixels        
        S = STREAMobj(FD,'minarea',area_cells_impose); % Flow accumulation
        DEM_new = imposemin(S,DEM_raster,GIS_Parameters.sl);
        DEM_DIFF = DEM_new - DEM_raster;
        DEM_raster = DEM_new;
%     imagesc(DEM_DIFF); colorbar; % if you want to plot
    clear area_cells_impose DEM_DIFF area_km2_impose
end

% Fill
if flags.flag_fill  == 1
% Fillsinks
    DEM_filled = fillsinks(DEM_raster);
    DIFFDEM = DEM_filled - DEM_raster;
    DIFFDEM.Z(DIFFDEM.Z==0) = nan;
    DEM_raster = DEM_filled;
    DEM_raster.Z(idx_nan_dem) = nan;
    DEM = DEM_raster.Z;
    if sum(sum(abs(DIFFDEM.Z(~isnan(DIFFDEM.Z))))) == 0
          warning('Your DEM seems to already be filled.')
    else
        imageschs(DEM_raster,DIFFDEM.Z);  
    end
end


%% Raster Values
% Initial Values 
LULC = double(LULC_raster.Z);
DEM = double(DEM_raster.Z);
SOIL = double(SOIL_raster.Z);


% Manual DEM Corrections
% warning('Please be careful here. We are doing manual corrections at the DEM. If you do not need, pelase delete.')
% DEM(1,55) = 0.025 + min(DEM(1,54),DEM(1,56));
% DEM(1,57) = 0.025 + min(DEM(1,56),DEM(1,58));
% DEM(1,59) = 0.025 + min(DEM(1,58),DEM(1,60));
% DEM(1,63) = 0.025 + min(DEM(1,62),DEM(1,64));
% DEM(1,68) = 0.025 + min(DEM(1,67),DEM(1,69));

% New Raster
DEM_Raster.Z = DEM;

neg_DEM = DEM <= 0;
neg_LULC = LULC < 0;
neg_SOIL = SOIL < 0; 
inf_nan_MAPS = isinf(DEM) + isnan(DEM) + neg_DEM + isnan(LULC) + isnan(SOIL) + neg_LULC + neg_SOIL + isinf(LULC) + isinf(SOIL); % Logical array
idx = inf_nan_MAPS > 0;

% Treated Values
LULC(idx) = nan;
DEM(idx) = nan;
SOIL(idx) = nan;

% Rebuilding Rasters to Lowest Extent
LULC_raster.Z = LULC;% Land Use and Land Cover Classification
DEM_raster.Z = DEM; % Digital Elevation Model
SOIL_raster.Z = SOIL; % Soil Map

% Raster Extent
xulcorner = DEM_raster.refmat(3,1); % Up Left Corner
yulcorner = DEM_raster.refmat(3,2);
Resolution = DEM_raster.refmat(2,1);
DEM = DEM_raster.Z;
idx_nan_dem = DEM <= 0;
DEM(idx_nan_dem) = nan;

 %% Set up the Import Options and import the data
Input_LULC = readtable(Input_Data_Label,'Sheet','Input_LULC_Data');
InputLULCDataRTC = table2array(Input_LULC(1:end,1:3));
LULC_Properties.LULC_Type = Input_LULC(1:end,4);
LULC_Properties.LULC_Type = string(table2array(LULC_Properties.LULC_Type));
LULC_Properties.number_of_LULC = size(LULC_Properties.LULC_Type,1);
LULC_Properties.LULC_Index = table2array(Input_LULC(1:end,3));

Input_Soil = readtable(Input_Data_Label,'Sheet','Input_SOIL_Data');
Soil_Properties.InputSoilDataRTC = table2array(Input_Soil(:,1:7));
Soil_Properties.SOIL_Type = Input_Soil(:,8);
Soil_Properties.SOIL_Type = string(table2array(Soil_Properties.SOIL_Type));
Soil_Properties.number_of_SOIL = size(Soil_Properties.SOIL_Type,1);
Soil_Properties.SOIL_Index = table2array(Input_Soil(1:end,7));

LULC_Properties.Impervious_LULC = table2array(Input_LULC(:,5)); % Assigned value for impervious areas
LULC_Properties.n_impervious = sum(~isnan(LULC_Properties.Impervious_LULC)); % Number of impervious areas
LULC_Properties.Impervious_LULC = LULC_Properties.Impervious_LULC(1:LULC_Properties.n_impervious,1);
LULC_Properties.Impervious_Rate = table2array(Input_LULC(:,6)); % Assigned value for impervious directly connected areas
LULC_Properties.Impervious_Rate = LULC_Properties.Impervious_Rate(1:LULC_Properties.n_impervious,1);

%% LULC Parameters
LULC_roughness = InputLULCDataRTC(:,1);
LULC_h_0 = InputLULCDataRTC(:,2);

%% Assigning Values for LULC value
LULC_Properties.roughness = zeros(size(LULC,1),size(LULC,2));
LULC_Properties.h_0 = zeros(size(LULC,1),size(LULC,2));
for i = 1:LULC_Properties.number_of_LULC
    % n
    k = LULC_Properties.LULC_Index(i);
        LULC_Properties.roughness(LULC == k) = LULC_roughness(i,1);
    % h_0
        LULC_Properties.h_0(LULC == k) = LULC_h_0(i,1);
end

clear LULC_roughness LULC_h_0

%% Soil Parameters
Soil_map_theta_sat = Soil_Properties.InputSoilDataRTC(:,1);
Soil_map_theta_i = Soil_Properties.InputSoilDataRTC(:,2);
Soil_map_psi = Soil_Properties.InputSoilDataRTC(:,3);
Soil_map_ksat = Soil_Properties.InputSoilDataRTC(:,4);
Soil_map_F_0 = Soil_Properties.InputSoilDataRTC(:,5);
Soil_map_h_ef_w_0 = Soil_Properties.InputSoilDataRTC(:,6);

%% Assigning Values for Each Soil Type

Soil_Properties.theta_sat = zeros(size(SOIL,1),size(SOIL,2));
Soil_Properties.theta_i = zeros(size(SOIL,1),size(SOIL,2));
Soil_Properties.ksat = zeros(size(SOIL,1),size(SOIL,2));
Soil_Properties.psi = zeros(size(SOIL,1),size(SOIL,2));
Soil_Properties.F_0 = zeros(size(SOIL,1),size(SOIL,2));
LULC_Properties.h_ef_w_0 = zeros(size(SOIL,1),size(SOIL,2));
h_ef_w_0 = zeros(size(SOIL,1),size(SOIL,2));
for i = 1:Soil_Properties.number_of_SOIL
    s = Soil_Properties.SOIL_Index(i);
    % Theta_sat
        Soil_Properties.theta_sat(SOIL == s) = Soil_map_theta_sat(i,1);
    % Theta_i
        Soil_Properties.theta_i(SOIL == s) = Soil_map_theta_i(i,1);
    % ksat
        Soil_Properties.ksat(SOIL == s) = Soil_map_ksat(i,1);
    % psi
        Soil_Properties.psi(SOIL == s) = Soil_map_psi(i,1);
    % F_0
        Soil_Properties.F_0(SOIL == s) = Soil_map_F_0(i,1);
    % h_ef_w_0
        h_ef_w_0(SOIL == s) = Soil_map_h_ef_w_0(i,1);
end

 

clear Soil_map_theta_sat Soil_map_theta_i Soil_map_psi Soil_map_ksat Soil_map_F_0 Soil_map_h_ef_w_0

%% Impervious Areas and Impervious Directly Connected Areas

for i = 1:length(LULC_Properties.Impervious_LULC)
    index = LULC_Properties.Impervious_LULC(i);
    idx_impervious = LULC == index;
    Soil_Properties.ksat(idx_impervious) = (1 - LULC_Properties.Impervious_Rate(i)/100)*Soil_Properties.ksat(idx_impervious); 
    % The idea behind it is that if we decrease ksat proportionally to the
    % impervious rate, we would be simulating impervious directly connected
    % areas
end

%%%% ----- Manual Input Data
slope_outlet = input_watershed_data(1,1);

% Automatic Input Data
Delta_x = DEM_raster.cellsize;
Delta_y = DEM_raster.cellsize;

%% Main Data of soil at watershed
% Changing names to match with the main file
LULC_Properties.n = LULC_Properties.roughness; Soil_Properties.dtheta = Soil_Properties.theta_sat - Soil_Properties.theta_i;

%% Mask in matrices to avoid numerical issues
mask = isnan(DEM) + isinf(DEM) > 0;
Soil_Properties.F_0(mask) = nan; 
LULC_Properties.h_0(mask) = nan; 
LULC_Properties.h_0(mask) = nan; 
Soil_Properties.theta_sat(mask) = nan; 
Soil_Properties.theta_i(mask) = nan; 
Soil_Properties.psi(mask) = nan; 
Soil_Properties.ksat(mask) = nan; 
Soil_Properties.dtheta(mask) = nan; 
h_ef_w_0(mask) = nan; 

%% 1.2 - Flow Direction, Slope and Direction Matrix %%%
 % coordinate system from left up corner (x -->) y (up-down)
[row, col] = find(DEM == min(min(DEM)));
coord_outlet = [row,col];
dim = size(DEM); rows = dim(1); cols = dim(2);

%%%% Flow Direction %%%%
[f_dir,idx_fdir] = FlowDirection(DEM,Delta_x,Delta_y,coord_outlet); % Flow direction matrix

%%%% Slope Calculation %%%%
[slope] = max_slope8D(DEM,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet);
idx_nan_fdir = isnan(slope);

%%%% Direction Matrix %%%%
z = size(f_dir,1)*size(f_dir,2);
index_i = 1:1:z;
index_j = 1:1:z;
dir = ones(z,1);
Direction_Matrix_Zeros = -1*sparse(index_i,index_j,dir); % Spase Matrix
[Direction_Matrix] = sparse(Find_D_Matrix(f_dir,coord_outlet,Direction_Matrix_Zeros));

f_dir(idx) = nan;
slope(idx) = nan;
Direction_Matrix(idx) = 0;
idx_nan = idx;

%% Clearing some variables
clear cols dim dir index index_i index_j input_channel_data input_data input_GIS_processing input_human_instability_data Input_LULC input_mpc_data input_plots_and_recording input_reservoir_data input_simulation_time input_smoothening_data Input_Soil input_table input_watershed_data InputLULCDataRTC
%% Plotting Input Rasters
x_grid = xulcorner + Resolution*[1:1:size(DEM,2)]; y_grid = yulcorner - Resolution*[1:1:size(DEM,1)];
filename = 'Input_Maps';
set(gcf,'units','inches','position',[2,0,8,6])
% -------- Manning  -------- %
t_title = 'Manning';
ax1 = subplot(3,2,1);
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
z = LULC_Properties.roughness; z(idx_nan) = nan;
idx = z < 0;
z(idx) = nan;
idx = isinf(z);
z(idx) = nan;
xmax = size(z,2);
xend = xmax;
ymax = size(z,1);
yend = ymax;
h_min = min(min(z));
F = z;
zmax = max(max(z(~isnan(z))));
if isempty(zmax) || isinf(zmax) || zmax == 0
    zmax = 0.1;
end
map = surf(x_grid,y_grid,F);
set(map,'LineStyle','none'); axis tight; grid on; box on; % this ensures that getframe() returns a consistent size; axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
title((t_title),'Interpreter','Latex','FontSize',12)
view(0,90)
if h_min == zmax
    zmax = 2*h_min;
end
caxis([h_min zmax]);
colormap(jet)
hold on
k = colorbar ;
ylabel(k,'$n$ ($\mathrm{s \cdot m^{-1/3}}$)','Interpreter','Latex','FontSize',12)
xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
zlabel ('$n$ ($\mathrm{s \cdot m^{-1/3}}$)','Interpreter','Latex','FontSize',12)

% ---------- h_0 --------------- %

ax2 = subplot(3,2,2);
t_title = 'Initial Abstraction';
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
z = LULC_Properties.h_0; z(idx_nan) = nan;
idx = z < 0;
z(idx) = nan;
idx = isinf(z);
z(idx) = nan;
xmax = size(z,2);
xend = xmax;
ymax = size(z,1);
yend = ymax;
h_min = min(min(z));
F = z;
zmax = max(max(z(~isnan(z))));
if isempty(zmax) || isinf(zmax) || zmax == 0
    zmax = 0.1;
end
map = surf(x_grid,y_grid,F);
set(map,'LineStyle','none'); axis tight; grid on; box on; % this ensures that getframe() returns a consistent size; axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
title((t_title),'Interpreter','Latex','FontSize',12)
view(0,90)
if h_min == zmax
    zmax = 2*h_min;
end
caxis([h_min zmax]);
colormap(jet)
hold on
k = colorbar ;
ylabel(k,'$h_0$ ($\mathrm{mm})$','Interpreter','Latex','FontSize',12)
xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
zlabel ('$h_0$ ($\mathrm{mm}$)','Interpreter','Latex','FontSize',12)

% ----------  k_sat ------------- %
ax3 = subplot(3,2,3);
t_title = 'Sat. Hyd. Conductivity';
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
z = Soil_Properties.ksat; z(idx_nan) = nan;
idx = z < 0;
z(idx) = nan;
idx = isinf(z);
z(idx) = nan;
xmax = size(z,2);
xend = xmax;
ymax = size(z,1);
yend = ymax;
h_min = min(min(z));
F = z;
zmax = max(max(z(~isnan(z))));
if isempty(zmax) || isinf(zmax) || zmax == 0
    zmax = 0.1;
end
map = surf(x_grid,y_grid,F);
set(map,'LineStyle','none'); axis tight; grid on; box on; % this ensures that getframe() returns a consistent size; axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
title((t_title),'Interpreter','Latex','FontSize',12)
view(0,90)
if h_min == zmax
    zmax = 2*h_min;
end
caxis([h_min zmax]);
colormap(jet)
hold on
k = colorbar ;
ylabel(k,'$k_{sat}$ ($\mathrm{mm/h})$','Interpreter','Latex','FontSize',12)
xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
zlabel ('$k_{sat}$ ($\mathrm{mm/h}$)','Interpreter','Latex','FontSize',12)

% ----------  dtheta ------------- %
ax4 = subplot(3,2,4);
t_title = 'Moisture Deficit';
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
z = Soil_Properties.dtheta; z(idx_nan) = nan;
idx = z < 0;
z(idx) = nan;
idx = isinf(z);
z(idx) = nan;
xmax = size(z,2);
xend = xmax;
ymax = size(z,1);
yend = ymax;
h_min = min(min(z));
F = z;
zmax = max(max(z(~isnan(z))));
if isempty(zmax) || isinf(zmax) || zmax == 0
    zmax = 0.1;
end
map = surf(x_grid,y_grid,F);
set(map,'LineStyle','none'); axis tight; grid on; box on; % this ensures that getframe() returns a consistent size; axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
title((t_title),'Interpreter','Latex','FontSize',12)
view(0,90)
if h_min == zmax
    zmax = 2*h_min;
end
caxis([h_min zmax]);
colormap(jet)
hold on
k = colorbar ;
ylabel(k,'$\Delta \theta$ ($\mathrm{cm^3.cm^{-3}})$','Interpreter','Latex','FontSize',12)
xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
zlabel ('$\Delta \theta$ ($\mathrm{cm^3.cm^{-3}}$)','Interpreter','Latex','FontSize',12)

% ----------  F_0 ------------- %
ax5 = subplot(3,2,5);
t_title = 'Initial Soil Content';
z = Soil_Properties.F_0; z(idx_nan) = nan;
idx = z < 0;
z(idx) = nan;
idx = isinf(z);
z(idx) = nan;
xmax = size(z,2);
xend = xmax;
ymax = size(z,1);
yend = ymax;
h_min = min(min(z));
F = z;
zmax = max(max(z(~isnan(z))));
if isempty(zmax) || isinf(zmax) || zmax == 0
    zmax = 0.1;
end
map = surf(x_grid,y_grid,F);
set(map,'LineStyle','none'); axis tight; grid on; box on; % this ensures that getframe() returns a consistent size; axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
title((t_title),'Interpreter','Latex','FontSize',12)
view(0,90)
if h_min == zmax
    zmax = 2*h_min;
end
caxis([h_min zmax]);
colormap(jet)
hold on
k = colorbar ;
ylabel(k,'$F_0$ ($\mathrm{m}$)','Interpreter','Latex','FontSize',12)
xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
zlabel ('$F_0$ ($\mathrm{m}$)','Interpreter','Latex','FontSize',12)


% ----------  I_0 ------------- %
ax6 = subplot(3,2,6);
t_title = 'Initial Water Depth';
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
z = h_ef_w_0; z(idx_nan) = nan;
idx = z < 0;
z(idx) = nan;
idx = isinf(z);
z(idx) = nan;
xmax = size(z,2);
xend = xmax;
ymax = size(z,1);
yend = ymax;
h_min = min(min(z));
F = z;
zmax = max(max(z(~isnan(z))));
if isempty(zmax) || isinf(zmax) || zmax == 0
    zmax = 0.1;
end
map = surf(x_grid,y_grid,F);
set(map,'LineStyle','none'); axis tight; grid on; box on; % this ensures that getframe() returns a consistent size; axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
title((t_title),'Interpreter','Latex','FontSize',12)
view(0,90)
if h_min == zmax
    zmax = 2*h_min;
end
caxis([h_min zmax]);
colormap(jet)
hold on
k = colorbar ;
ylabel(k,'$h_{ef}^0$ ($\mathrm{m}$)','Interpreter','Latex','FontSize',12)
xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
zlabel ('$h_{ef}^0$ ($\mathrm{m}$)','Interpreter','Latex','FontSize',12)

exportgraphics(gcf,'Outputs\Input_Data\Input_Maps.TIF','ContentType','image','Colorspace','rgb','Resolution',300)
close all

%% Plotting Flow Direction and Slope Maps
Slope_Map = slope;
Flow_Dir_Map = f_dir;

subplot(1,2,1)
t_title = 'Slope';
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
z = Slope_Map; z(idx_nan) = nan;
idx = z < 0;
z(idx) = nan;
idx = isinf(z);
z(idx) = nan;
xmax = size(z,2);
xend = xmax;
ymax = size(z,1);
yend = ymax;
h_min = min(min(z));
F = z;
zmax = max(max(z(~isnan(z))));
if isempty(zmax) || isinf(zmax) || zmax == 0
    zmax = 0.1;
end
map = surf(x_grid,y_grid,F);
set(map,'LineStyle','none'); axis tight; grid on; box on; % this ensures that getframe() returns a consistent size; axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
title((t_title),'Interpreter','Latex','FontSize',12)
view(0,90)
if h_min == zmax
    zmax = 2*h_min;
end
caxis([h_min zmax]);
colormap(jet)
hold on
k = colorbar ;
ylabel(k,'$s_0$ ($\mathrm{m/m}$)','Interpreter','Latex','FontSize',12)
xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
zlabel ('$s_0$ ($\mathrm{m/m}$)','Interpreter','Latex','FontSize',12)


subplot(1,2,2)

t_title = 'Flow Direction';
axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
z = Flow_Dir_Map; z(idx_nan) = nan;
idx = z < 0;
z(idx) = nan;
idx = isinf(z);
z(idx) = nan;
xmax = size(z,2);
xend = xmax;
ymax = size(z,1);
yend = ymax;
h_min = min(min(z));
F = z;
zmax = max(max(z(~isnan(z))));
if isempty(zmax) || isinf(zmax) || zmax == 0
    zmax = 0.1;
end
map = surf(x_grid,y_grid,F);
set(map,'LineStyle','none'); axis tight; grid on; box on; % this ensures that getframe() returns a consistent size; axis tight; grid on; box on; % this ensures that getframe() returns a consistent size
title((t_title),'Interpreter','Latex','FontSize',12)
view(0,90)
if h_min == zmax
    zmax = 2*h_min;
end
caxis([h_min zmax]);
colormap(jet)
hold on
k = colorbar;
ylabel(k,'Flow Direction','Interpreter','Latex','FontSize',12)
xlabel(' x (m) ','Interpreter','Latex','FontSize',12)
ylabel ('y (m) ','Interpreter','Latex','FontSize',12)
zlabel ('Direção)','Interpreter','Latex','FontSize',12)

exportgraphics(gcf,'Outputs\Input_Data\Fdir_Slope.TIF','ContentType','image','Colorspace','rgb','Resolution',300)
close all
%% Post Processing Rasters
% -------- In case you want to plot it -------- %
% ax1 = subplot(1,3,1)
% close all
% area_km2 = GIS_Parameters.min_area; % km2
% area_cells = area_km2./((DEM_raster.cellsize/1000)^2); % pixels
% FD = FLOWobj(DEM_raster);
% S = STREAMobj(FD,'minarea',area_cells); % Streams with faccum > area_threshold
% imageschs(DEM_raster,[],'colormap',[1 1 1],'colorbar',false)
% hold on
% width = max(DEM_raster.size)/70;
% h = plotdbfringe(FD,S,'colormap',parula,'width',width);
% plot(S,'b','linewidth',2)
% xlabel('utm [x] (m)','interpreter','latex')
% ylabel('utm [y] (m)','interpreter','latex')
% ScaleBar
% hold off
% 
% imageschs(LULC_raster);
% h = colorbar;
% h.Label.String = 'DEM [m]';
% h.Label.Interpreter = 'latex';
% xlabel('utm [x]','interpreter','latex')
% ylabel('utm [y]','interpreter','latex')
% colormap('jet')
% ScaleBar
% 
% imageschs(SOIL_raster);
% h = colorbar;
% h.Label.String = 'DEM [m]';
% h.Label.Interpreter = 'latex';
% xlabel('utm [x]','interpreter','latex')
% ylabel('utm [y]','interpreter','latex')
% ScaleBar
% hold on
