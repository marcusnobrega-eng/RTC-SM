%% Watershed Physic Infomration
% Marcus Nóbrega & Matheus Schroeder 
% Goal: Estiamate watershed properties
% Last Modified: 09/2022

%% Data in txt format (Gis Data)

% Here you should upload your LULC, DEM, and SOIL maps
% LULC = load('LULC_50v04_int.asc'); % Land Use and Land Cover Classification
% DEM = load('MDE_50_v04.asc'); % Digital Elevation Model
% SOIL = load('SOIL_50v04.asc'); % Soil Map

% Rasters
LULC_raster = GRIDobj('LULC_BHARI.tif'); % Land Use and Land Cover Classification
DEM_raster = GRIDobj('DEM_BHARI.tif'); % Digital Elevation Model
SOIL_raster = GRIDobj('SOIL_BHARI.tif'); % Soil Map

%% Treating Values with Issues

% Extent Problem
if sum(size(DEM_raster.Z)) >= sum(size(LULC_raster.Z)) && sum(size(DEM_raster.Z)) >= sum(size(SOIL_raster.Z)) % DEM is larger
    raster_resample = DEM_raster;
    % Resample other two rasters
    LULC_raster = resample(LULC_raster,raster_resample);
    SOIL_raster = resample(SOIL_raster,raster_resample);
end

if sum(size(SOIL_raster.Z)) >= sum(size(DEM_raster.Z)) && sum(size(SOIL_raster.Z)) >= sum(size(LULC_raster.Z))  % SOIL is larger
    raster_resample = DEM_raster;
    % Resample other two rasters
    LULC_raster = resample(LULC_raster,raster_resample);
    DEM_raster = resample(DEM_raster,raster_resample);
end

if sum(size(LULC_raster.Z)) >= sum(size(DEM_raster.Z)) && sum(size(LULC_raster.Z)) >= sum(size(SOIL_raster.Z))  % SOIL is larger
    raster_resample = DEM_raster;
    % Resample other two rasters
    SOIL_raster = resample(SOIL_raster,raster_resample);
    DEM_raster = resample(DEM_raster,raster_resample);
end

% Raster Values
LULC = double(LULC_raster.Z);
DEM = double(DEM_raster.Z);
SOIL = double(SOIL_raster.Z);

neg_DEM = DEM <= 0;
neg_LULC = LULC < 0;
neg_SOIL = SOIL < 0; 
inf_nan_MAPS = isinf(DEM) + isnan(DEM) + neg_DEM + isnan(LULC) + isnan(SOIL) + neg_LULC + neg_SOIL + isinf(LULC) + isinf(SOIL); % Logical array
idx = inf_nan_MAPS > 0;

% Rebuilding Rasters to Lowest Extent
LULC_raster.Z = LULC;% Land Use and Land Cover Classification
DEM_raster.Z = DEM; % Digital Elevation Model
SOIL_raster.Z = SOIL; % Soil Map

% Replacing Values with Issues 
DEM_raster.Z(idx) = nan;

 %% Set up the Import Options and import the data
Input_LULC = readtable('Input_LULC_Data_Watershed.csv');
InputLULCDataRTC = table2array(Input_LULC(2:end,1:3));
LULC_Type = Input_LULC(2:end,4);
LULC_Type = string(table2array(LULC_Type));

Input_Soil = readtable('Input_Soil_Data_Watershed.csv');
InputSoilDataRTC = table2array(Input_Soil(:,1:6));
SOIL_Type = Input_Soil(:,8);
SOIL_Type = string(table2array(SOIL_Type));

Impervious_LULC = table2array(Input_LULC(1,6)); % Assigned value for impervious areas
idx_impervious = LULC == Impervious_LULC; % Logical Array

%% LULC Parameters
LULC_roughness = InputLULCDataRTC(:,1);
LULC_h_0 = InputLULCDataRTC(:,2);

%% Soil Parameters
Soil_map_sat = InputSoilDataRTC(:,1);
Soil_map_i = InputSoilDataRTC(:,2);
Soil_map_psi = InputSoilDataRTC(:,3);
Soil_map_ksat = InputSoilDataRTC(:,4);
Soil_map_F_0 = InputSoilDataRTC(:,5);
Soil_map_h_ef_w_0 = InputSoilDataRTC(:,6);

%% Assigning Values for Each Soil and LULC value
%% Theta Sat
teta_sat = SOIL;

teta_sat(teta_sat(:,:)==1) = Soil_map_sat(1,1);
teta_sat(teta_sat(:,:)==2) = Soil_map_sat(2,1);
teta_sat(teta_sat(:,:)==3) = Soil_map_sat(3,1);
teta_sat(teta_sat(:,:)==4) = Soil_map_sat(4,1);
teta_sat(teta_sat(:,:)==5) = Soil_map_sat(5,1);
teta_sat(teta_sat(:,:)==6) = Soil_map_sat(6,1);
teta_sat(teta_sat(:,:)==7) = Soil_map_sat(7,1);
% teta_sat(teta_sat(:,:)<0) = nan;

%% teta_i
teta_i = SOIL;

teta_i(teta_i(:,:)==1) = Soil_map_i(1,1);
teta_i(teta_i(:,:)==2) = Soil_map_i(2,1);
teta_i(teta_i(:,:)==3) = Soil_map_i(3,1);
teta_i(teta_i(:,:)==4) = Soil_map_i(4,1);
teta_i(teta_i(:,:)==5) = Soil_map_i(5,1);
teta_i(teta_i(:,:)==6) = Soil_map_i(6,1);
teta_i(teta_i(:,:)==7) = Soil_map_i(7,1);
% teta_i(teta_i(:,:)<0) = nan;
teta_i(teta_i(:,:)<0) = 0;

%% F_0
F_0 = SOIL;

F_0(F_0(:,:)==1) = Soil_map_F_0(1,1);
F_0(F_0(:,:)==2) = Soil_map_F_0(2,1);
F_0(F_0(:,:)==3) = Soil_map_F_0(3,1);
F_0(F_0(:,:)==4) = Soil_map_F_0(4,1);
F_0(F_0(:,:)==5) = Soil_map_F_0(5,1);
F_0(F_0(:,:)==6) = Soil_map_F_0(6,1);
F_0(F_0(:,:)==7) = Soil_map_F_0(7,1);
% F_0(F_0(:,:)<0) = nan;
F_0(F_0(:,:)<0) = 0;

%% ksat
ksat = SOIL;

ksat(ksat(:,:)==1) = Soil_map_ksat(1,1);
ksat(ksat(:,:)==2) = Soil_map_ksat(2,1);
ksat(ksat(:,:)==3) = Soil_map_ksat(3,1);
ksat(ksat(:,:)==4) = Soil_map_ksat(4,1);
ksat(ksat(:,:)==5) = Soil_map_ksat(5,1);
ksat(ksat(:,:)==6) = Soil_map_ksat(6,1);
ksat(ksat(:,:)==7) = Soil_map_ksat(7,1);
% ksat(ksat(:,:)<0) = nan;
ksat(ksat(:,:)<0) = 0;

%%%%%%% -------- Constraint at Impervious Areas -------- %%%%%%
ksat(idx_impervious) = 0; % This way, impervious areas have no infiltration capacity

%% psi
psi = SOIL;

psi(psi(:,:)==1) = Soil_map_psi(1,1);
psi(psi(:,:)==2) = Soil_map_psi(2,1);
psi(psi(:,:)==3) = Soil_map_psi(3,1);
psi(psi(:,:)==4) = Soil_map_psi(4,1);
psi(psi(:,:)==5) = Soil_map_psi(5,1);
psi(psi(:,:)==6) = Soil_map_psi(6,1);
psi(psi(:,:)==7) = Soil_map_psi(7,1);
% psi(psi(:,:)<0) = nan;
psi(psi(:,:)<0) = 0;

%% h_ef_w_0 - This is the initial water depth, you can enter a map of warm-up if you want
h_ef_w_0 = SOIL;

h_ef_w_0(h_ef_w_0(:,:)==1) = Soil_map_h_ef_w_0(1,1);
h_ef_w_0(h_ef_w_0(:,:)==2) = Soil_map_h_ef_w_0(2,1);
h_ef_w_0(h_ef_w_0(:,:)==3) = Soil_map_h_ef_w_0(3,1);
h_ef_w_0(h_ef_w_0(:,:)==4) = Soil_map_h_ef_w_0(4,1);
h_ef_w_0(h_ef_w_0(:,:)==5) = Soil_map_h_ef_w_0(5,1);
h_ef_w_0(h_ef_w_0(:,:)==6) = Soil_map_h_ef_w_0(6,1);
h_ef_w_0(h_ef_w_0(:,:)==7) = Soil_map_h_ef_w_0(7,1);
% h_ef_w_0(h_ef_w_0(:,:)<0) = nan;
h_ef_w_0(h_ef_w_0(:,:)<0) = 0;

%% roughness
roughness= LULC;

roughness(roughness(:,:)==0) = LULC_roughness(1,1);
roughness(roughness(:,:)==1) = LULC_roughness(2,1);
roughness(roughness(:,:)==2) = LULC_roughness(3,1);
roughness(roughness(:,:)==3) = LULC_roughness(4,1);
roughness(roughness(:,:)==4) = LULC_roughness(5,1);
roughness(roughness(:,:)==5) = LULC_roughness(6,1);
roughness(roughness(:,:)==6) = LULC_roughness(7,1);
roughness(roughness(:,:)==7) = LULC_roughness(8,1);
roughness(roughness(:,:)==8) = LULC_roughness(9,1);

%% h_0
h_0 = LULC;

h_0(h_0(:,:)==0) = LULC_h_0(1,1);
h_0(h_0(:,:)==1) = LULC_h_0(2,1);
h_0(h_0(:,:)==2) = LULC_h_0(3,1);
h_0(h_0(:,:)==3) = LULC_h_0(4,1);
h_0(h_0(:,:)==4) = LULC_h_0(5,1);
h_0(h_0(:,:)==5) = LULC_h_0(6,1);
h_0(h_0(:,:)==6) = LULC_h_0(7,1);
h_0(h_0(:,:)==7) = LULC_h_0(8,1);
h_0(h_0(:,:)==8) = LULC_h_0(9,1);

%%%% ----- Manual Input Data
% Delta_x = input_data(9,1);
% Delta_y = input_data(10,1);
slope_outlet = input_data(5,1);

% Automatic Input Data
Delta_x = DEM_raster.cellsize;
Delta_y = DEM_raster.cellsize;

%% Model Type
flag_diffusive = input_data(14,1);

% Filling Sinks if flag_diffusive ~= 0
if flag_diffusive == 0
    DEM_filled = fillsinks(DEM_raster); % Filling sinks to be able to run kinematic_wave
    DEM = DEM_filled.Z;
    DEM_raster = DEM_filled; % New Filled DEM
end

%% DEM Smoothening
flag_smoothening = 1;
if flag_smoothening == 1
    min_area = 1; % km2
    flag_trunk = 0; % 0 All streams, 1 only main river
    tau = 0.2; % Between 0 and 1
    K_value = 10; % 1 - 20
    [DEM_raster,DEM] = DEM_smoothening(DEM_raster,min_area,flag_trunk,tau,K_value);
end

%% Main Data of soil at watershed
% Changing names to match with the main file
n = roughness; dtheta = teta_sat - teta_i;

%% Mask in matrices to avoid numerical issues
mask = isnan(DEM);
% F0 constraint
F_0(mask) = 1e-6; % small number in mm
n(mask) = 1e-8; % small number in m^(1/3)/s
h_0(mask) = 1e-6; % small number in m^(1/3)/s
teta_sat(mask) = 1e-6; % small number in mm/h
teta_i(mask) = 1e-6; % small number in mm/h
psi(mask) = 1e-6;  % small number in cm3/cm3
ksat(mask) = 1e-6; % small number
dtheta(mask) = 1e-6; % small number
h_ef_w_0(mask) = 1e-6; % small number

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
Direction_Matrix_Zeros = sparse(-eye(length(f_dir(:)),length(f_dir(:)))); % Spase Matrix
[Direction_Matrix] = sparse(Find_D_Matrix(f_dir,coord_outlet,Direction_Matrix_Zeros));

f_dir(idx) = 0;
slope(idx) = 0;
Direction_Matrix(idx) = 0;

%% Post Processing Rasters
% -------- In case you want to plot it -------- %
% ax1 = subplot(1,3,1)
% close all
% area_km2 = min_area; % km2
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
