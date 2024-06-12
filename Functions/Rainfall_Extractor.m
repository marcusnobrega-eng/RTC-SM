%% Main File - MPC solver for the Watershed + Reservoir + Channel system
% Developer: Marcus Nobrega Gomes Junior
% 8/1/2021
% Main Script
% Goal: Create the main file containing:
% a) Watershed Model
% b) Connection between the Optimization Function and the Solvers
% c) Plant Models of the Reservoir and Channel
% d) Saving final results
clear all
clc

%% Defining Global Variables
global Qout_w Qout_w_horizon steps_control_horizon steps_horizon time_step n_steps Control_Vector Nvars i_reservoir h_c_0 h_r_t ur_eq_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP new_timestep Control_Interval Control_Horizon Prediction_Horizon g Cd flag_r number_of_orifices D l b relative_hmin Cds Lef hs porosity orifice_height x_length y_length roughness segments slope slope_outlet_channel max_iterations max_fun_eval n_randoms flag_c flag_r hmin s h_end slope_outlet_channel n_channel a_noise b_noise ur_eq hr_eq flag_save_data m L Cd_HI u B_p pperson pfluid h_max_channel L_channel stage_area flag_channel_modeling flag_human_instability flag_detention_time detention_time_max q_max_star q_max_star_star alpha_p rho_u rho_h rho_hmax rho_c rho_HI max_res_level y_ref flag_gatecontrol ks_gatecontrol alpha_s_gatecontrol number_of_controls previous_control_gate previous_control_valve uv_eq us_eq uv_eq_t us_eq_t u_s u_v

%% Reading Main Input Data
input_table = readtable('Input_Data.xlsx','Sheet','Input_Main_Data_RTC'); % Reading Input data from Excel

%% Rasters Path
input_data = readtable('Input_Data.xlsx','Sheet','Input_Main_Data_RTC');
Topo_path = string(table2array(input_data(1,29)));
DEM_path = string(table2array(input_data(2,29)));
LULC_path = string(table2array(input_data(3,29)));
SOIL_path = string(table2array(input_data(4,29)));
Warmup_path = string(table2array(input_data(5,29)));

%% Load ToolBoxes
addpath(genpath(Topo_path));

% Simulation Time
input_simulation_time = input_table(1:2,2);

% Plots and Recording Times
input_plots_and_recording = table2array(input_table(4:end,2));

% Watershed Data
input_watershed_data = table2array(input_table(1:end,8));

% Reservoir Data
input_reservoir_data = table2array(input_table(1:end,11));

% Channel Data
input_channel_data = table2array(input_table(1:end,14));

% MPC
input_mpc_data = table2array(input_table(1:end,17));

% Human Instability Data
input_human_instability_data = table2array(input_table(1:end,20));

% Smoothening
input_smoothening_data = table2array(input_table(1:end,23));

% GIS Processing
input_GIS_processing = table2array(input_table(1:end,26));

% 1) Running Control

record_time_maps = input_plots_and_recording(1,1); % Time where maps are recorded in (min)
tfinal = 24*60*(table2array(input_simulation_time(2,1)) - table2array(input_simulation_time(1,1))); % Simulation duration in (min)
time_begin = datetime(datestr(table2array(input_simulation_time(1,1))+datenum('30-Dec-1899')));
time_end = datetime(datestr(table2array(input_simulation_time(2,1))+datenum('30-Dec-1899')));
if tfinal < 0
    error('Please make sure Date End is later than Date Begin.')
end
% ETP and Hydrographs (Recording Time)
record_time_ETP = input_plots_and_recording(2,1); % Minutes
record_time_hydrographs = input_plots_and_recording(3,1); % Minutes; % We are assuming the same as the rainfall (Minutes)


% 2) Watershed Data

slope_outlet = input_watershed_data(1,1);
% ETP = input_watershed_data(2,1);
time_step = input_watershed_data(3,1);
flag_diffusive = input_watershed_data(4,1); % Type of shallow water model
flag_rainfall = input_watershed_data(5,1);
flag_spatial_rainfall = input_watershed_data(6,1);
flag_ETP = input_watershed_data(7,1);
flag_spatial_ETP = input_watershed_data(8,1);
flag_Rs = input_watershed_data(9,1); % What is it?
flag_Ur = input_watershed_data(10,1); % What is it?
flag_imposemin = input_watershed_data(11,1); % if == 1, we impose a minimum slope

% 3) Reservoir Data
g = input_reservoir_data(1,1);
Cd = input_reservoir_data(2,1);
flag_r = input_reservoir_data(3,1);
flag_c = input_reservoir_data(4,1);
number_of_orifices = input_reservoir_data(5,1);
D = input_reservoir_data(6,1);
l = input_reservoir_data(7,1);
b = input_reservoir_data(8,1);
relative_hmin = input_reservoir_data(9,1);
Cds = input_reservoir_data(10,1);
Lef = input_reservoir_data(11,1);
hs = input_reservoir_data(12,1);
porosity = input_reservoir_data(13,1);
orifice_height = input_reservoir_data(14,1);
ur_eq_t = input_reservoir_data(15,1);
h_r_t = input_reservoir_data(16,1);
hmin = input_reservoir_data(17,1);
flag_reservoir_wshed = input_reservoir_data(19,1); % If == 1, we subtract the reservoir area * i in the outlet flow
u_static = input_reservoir_data(20,1);


% 4) Channel Data
x_length = input_channel_data(1,1);
y_length = input_channel_data(2,1);
n_channel = input_channel_data(3,1);
segments = input_channel_data(4,1);
h_end = input_channel_data(5,1);
s = input_channel_data(6,1);
slope_outlet_channel = input_channel_data(7,1);
h_c_0 = zeros(segments,1);

% 5) MPC Control
new_timestep = input_mpc_data(1,1);
Control_Interval = input_mpc_data(2,1);
Control_Horizon = input_mpc_data(3,1);
Prediction_Horizon = input_mpc_data(4,1);
% 5.1) Interior Point Solver
max_iterations = input_mpc_data(5,1);
max_fun_eval = input_mpc_data(6,1);
a_noise = input_mpc_data(7,1);
b_noise = input_mpc_data(8,1);
% 5.1.1) Randoms
n_randoms = input_mpc_data(9,1);
% Initial Equilibrium Points for the Reservoir
ur_eq = input_mpc_data(10,1);
hr_eq = input_mpc_data(11,1);
% Data
flag_save_data = input_mpc_data(12,1);
% Objective Function Constraint
flag_channel_modeling = input_mpc_data(14,1);
flag_human_instability = input_mpc_data(15,1);
flag_detention_time = input_mpc_data(16,1);
detention_time_max = input_mpc_data(17,1);
q_max_star = input_mpc_data(18,1);
q_max_star_star = input_mpc_data(19,1);
alpha_p = input_mpc_data(20,1);
rho_u = input_mpc_data(21,1);
rho_h = input_mpc_data(22,1);
rho_hmax = input_mpc_data(23,1);
rho_c = input_mpc_data(24,1);
rho_HI = input_mpc_data(25,1);
max_res_level = input_mpc_data(26,1);
y_ref = input_mpc_data(27,1);
flag_optimization_solver = input_mpc_data(28,1); % -1 static with no optimization, 0 patternsearch, 1 fmincon
ks_gatecontrol = input_mpc_data(29,1);
alpha_s_gatecontrol = input_mpc_data(30,1);

% 6) Human instability

m = input_human_instability_data(1,1); %m
L = input_human_instability_data(2,1); %m
Cd_HI = input_human_instability_data(3,1); %adm
u_p = input_human_instability_data(4,1); %adm
B_p = input_human_instability_data(5,1); %m
pperson = input_human_instability_data(6,1); %Kg/m³
pfluid = input_human_instability_data(7,1); %Kg/m³
h_max_channel = input_human_instability_data(8,1); %m
L_channel = input_human_instability_data(9,1); %m

% 7) Smoothening

flag_smoothening = input_smoothening_data(1,1);
min_area = input_smoothening_data(3,1); % km2
flag_trunk = input_smoothening_data(2,1); % 0 All streams, 1 only main river
tau = input_smoothening_data(4,1); % Between 0 and 1
K_value = input_smoothening_data(5,1); % 1 - 20
sl = input_smoothening_data(6,1); % m/m minimum slope for imposemin
% 8) GIS Parameters

flag_resample = input_GIS_processing(1,1);
resolution_resampled = input_GIS_processing(2,1);
flag_fill = input_GIS_processing(3,1);

%% 1.0 - Defining simulation duration

time = (0:time_step:tfinal*60)/60; % time vector in min
n_steps = length(time); % number of steps

% --- MAPS --- %

number_of_records = floor((n_steps-1)*time_step/(record_time_maps*60)); % number of stored data (size of the vector)

% --- Hydrographs and Depths ---- %

number_of_records_hydrographs = floor((n_steps-1)*time_step/(record_time_hydrographs*60)); % number of stored data (size of the vector)
time_records = [0:record_time_maps:tfinal]; % time in minutes
time_records_hydrographs = [0:record_time_hydrographs:tfinal]; % time in minutes
time_store_hydrographs = time_records_hydrographs*60./time_step; % number of steps necessary to reach the recording vector
time_store_hydrographs(1) = 1;

% --- ETP ---- %

time_records_ETP = [0:record_time_ETP:tfinal]; % time in minutes
time_store = time_records*60./time_step; % number of steps necessary to reach the recording vector
time_store(1) = 1; % the zero is the firt time step
time_store_ETP = time_records_ETP*60./time_step; % number of steps necessary to reach the recording vector
time_store_ETP(1) = 1;

% --- Time --- %
t = time_step/60; % begin time-step (min)

%% 2.0 - Watershed Preprocessing

% Call sub - Here we read all watershed information
Watershed_Physics

%% 3.0 - Calculating Flow Constants

resolution = (Delta_x + Delta_y)/2; % Average cell resolution
Lambda = (resolution)*(1./n).*slope.^(0.5); % Lumped hydraulic properties for k = 1
zzz = isinf(DEM) + isnan(DEM); % Logical array with where infs or nans occur
idx = zzz > 0; % Logical array where zzz > 0
Lambda(idx) = nan; % Values with zzz > 0 have no hydraulic properties
Lambda = Lambda(:); % Concatenated Lambda resultin in a 1-D array

%% 4.0 - Preallocations of Watershed States

d = zeros(numel(DEM),length(time_store));
flow_t = d;
I = d; % zeros
flow_outlet = zeros(length(time_store),1);
h_ef_w(:,1) = h_ef_w_0(:);
ETP = zeros(size(DEM,1),size(DEM,2));
% Output function of 1 watershed
Qout_w = zeros(length(time_store_hydrographs),1); % Catchment Outflow (cms)
Qout_w(1,1) = flow_outlet(1,1);

% ETP saving
ETP_saving = zeros(length(time_store_ETP),1); % ETP at the Outlet (mm/h)

% Position of a cell (2-D) in a 1-D concatenated vector
% Given a 2-D position (row,col), where row is the row of the outlet and col, the
% collumn of the outlet, we can find pos, such that:
% pos = (y1-1)*rows + x1;

pos = (col-1)*rows + row; % Outlet vector containing the position of the outlet
Depth_out = zeros(length(time_store_hydrographs),1);

% Depth at the outlet - Not quite sure if it is 100% correct
Depth_out(1,1) = mean(h_ef_w(pos,1));

% Infiltration Rate ate the Outlet
f_rate = zeros(1,length(time_store_hydrographs));
f_rate(1,1) = 0;

% Inflow_Matrix
Inflow_Matrix = Direction_Matrix;
Inflow_Matrix(Inflow_Matrix == -1) = 0;

%% 5.0 Evapotranspiration

% Mask of the Watershed
neg_DEM = DEM < 0;
DEM(neg_DEM) = nan;
idx_cells = DEM >= 0;
idx_cells = double(idx_cells(:));
lat(neg_DEM) = nan;
lon(neg_DEM) = nan;

% Quantify ETP - Penman-Monteith
n_obs = 1; % Only for cases where we are not modeling ETP
ETP_save = zeros(size(DEM,1),size(DEM,2),n_obs);
ETR_save = ETP_save;
if flag_ETP == 1 % We only read it if we are actually modeling it
    % Read ETP data
    if flag_spatial_ETP == 1
        input_table = readtable('Input_Data.xlsx','Sheet','ETP_Spatial_Input'); % Input ETP data to interpolate from IDW method.
    else
        input_table = readtable('Input_Data.xlsx','Sheet','Concentrated_ETP_Data'); % Input ETP data to interpolate from IDW method.
    end
    Krs = 0.16; % Default.
    alpha_albedo_input = 0.23; % Default.
    flag_U_R = 1; % if flag == 1, use RU. Else, use simplification method.
    flag_Rs = 0; % if flag == 1, quantify Rs from the number of hour in the day. Else, quantify in function of the temperature.

    % Observations
    n_obs = sum((table2array(input_table(:,3))>=0)); % Number of observations
    time_begin_ETP = table2array(input_table(3,2));
    time_end_ETP = table2array(input_table(end,2));
    n_max_etp_stations = 50; % Maximum number of sations to interpolate
    time_step_etp = minutes(table2array(input_table(4,2)) - table2array(input_table(3,2))); % min
    end_etp = (n_obs-1)*time_step_etp;
    climatologic_spatial_duration = 0:time_step_etp:(end_etp); % Climate data time in minutes
    input_table_day = table2array(input_table(3:n_obs+2,2));
    time_ETP_01 = input_table_day(1:n_obs,1); % First day of ETP

    % Preallocating Arrays
    maxtemp_stations = zeros(n_obs,n_max_etp_stations);
    mintemp_stations = zeros(n_obs,n_max_etp_stations);
    avgtemp_stations = zeros(n_obs,n_max_etp_stations);
    u2_stations = zeros(n_obs,n_max_etp_stations);
    ur_stations = zeros(n_obs,n_max_etp_stations);
    G_stations = zeros(n_obs,n_max_etp_stations);
    coordinates = zeros(n_max_etp_stations,2);

    if flag_spatial_ETP == 1

        % ETP Data
        for i = 1:n_max_etp_stations
            try
                n_stations = 50;
                % Maximum Temperature
                maxtemp_stations(:,i) = table2array(input_table(3:end,6*(i-1) + 3));
                % Minimum Temperature
                mintemp_stations(:,i) = table2array(input_table(3:end,6*(i-1) + 4));
                % Average Temperature
                avgtemp_stations(:,i) = table2array(input_table(3:end,6*(i-1) + 5));
                % U2
                u2_stations(:,i) = table2array(input_table(3:end,6*(i-1) + 6));
                % UR
                ur_stations(:,i) = table2array(input_table(3:end,6*(i-1) + 7));
                % G
                G_stations(:,i) = table2array(input_table(3:end,6*(i-1) + 8));

                % Coordinates
                coordinates(i,1) = table2array(input_table(1,6*(i-1) + 6));
                coordinates(i,2) = table2array(input_table(1,6*(i-1) + 8));
            catch
                n_stations = i-1; % Number of statitons considered
                break
            end
        end
    else
        n_stations = 1;
        % Concentrated ETP
        maxtemp_stations = table2array(input_table(3:end,3));
        % Minimum Temperature
        mintemp_stations = table2array(input_table(3:end,4));
        % Average Temperature
        avgtemp_stations = table2array(input_table(3:end,5));
        % U2
        u2_stations = table2array(input_table(3:end, 6));
        % UR
        ur_stations = table2array(input_table(3:end,7));
        % G
        G_stations = table2array(input_table(3:end,8));
        % Coordinates
        coordinates(1,1) = table2array(input_table(1,6));
        coordinates(1,2) = table2array(input_table(1,8));
    end
    % Extract coordinates from the raster
    if flag_resample == 1
        GRIDobj2geotiff(DEM_raster,'DEM_resampled')
        fname = 'DEM_resampled.tif'; % Input DEM
    else
        fname = DEM_path; % Input DEM
    end
    %     [~,R_etp] = readgeoraster(fname);
    info = geotiffinfo(fname);
    DEM_etp = DEM;
    height = info.Height; % Integer indicating the height of the image in pixels
    width = info.Width; % Integer indicating the width of the image in pixels
    [cols_etp,rows_etp] = meshgrid(1:width,1:height);
    [x_etp,y_etp] = pix2map(info.RefMatrix, rows_etp, cols_etp);
    [lat,lon] = projinv(info, x_etp,y_etp);

    % Interpolate Climatalogic Data
    % ---- Grids ---- %
    x_coordinate = coordinates(1:n_stations,1);
    y_coordinate = coordinates(1:n_stations,2);

    input_table_day = table2array(input_table(3:n_obs+2,2));

    % Interpolation Data
    z1 = find(input_table_day >= time_begin,1,'first');
    z2 = find(input_table_day <= time_end,1,'last');
    n_obs_model = z2 - z1 + 1;
    ETP_save = zeros(size(DEM_etp,1),size(DEM_etp,2),n_obs_model);
    ETR_save = ETP_save;
    for k = 1:n_obs_model
        % ---- Evapotranspiration ---- %
        % Day
        day_of_year = day(datetime(time_ETP_01(k,1), 'InputFormat', 'dd-MMM-yyyy' ), 'dayofyear' );
        % Maximum Temperature
        var_obs = maxtemp_stations(k,1:n_stations)';
        [max_temp,~,~] = Interpolator(x_coordinate,y_coordinate,var_obs,x_etp(1,:)',y_etp(:,1));
        % Minimum Temperature
        var_obs = mintemp_stations(k,1:n_stations)';
        [min_temp,~,~] = Interpolator(x_coordinate,y_coordinate,var_obs,x_etp(1,:)',y_etp(:,1));
        % Average Temperature
        var_obs = avgtemp_stations(k,1:n_stations)';
        [avg_temp,~,~] = Interpolator(x_coordinate,y_coordinate,var_obs,x_etp(1,:)',y_etp(:,1));
        % U2
        var_obs = u2_stations(k,1:n_stations)';
        [u2,~,~] = Interpolator(x_coordinate,y_coordinate,var_obs,x_etp(1,:)',y_etp(:,1));
        % UR
        var_obs = ur_stations(k,1:n_stations)';
        [ur,~,~] = Interpolator(x_coordinate,y_coordinate,var_obs,x_etp(1,:)',y_etp(:,1));
        % G
        var_obs = G_stations(k,1:n_stations)';
        [G,~,~] = Interpolator(x_coordinate,y_coordinate,var_obs,x_etp(1,:)',y_etp(:,1));

        % ---- Run ETP code ---- %
        [ETP] = Evapotranspiration(DEM_etp, avg_temp,max_temp,min_temp,day_of_year,lat,u2, ur, Krs, alpha_albedo_input, G, flag_Rs, flag_U_R);

        % -- Mask -- %
        ETP(neg_DEM) = nan;
        ETP_save(:,:,k) = ETP;
        if flag_spatial_ETP == 1
            ETP_interpolation_percentage = k/n_obs_model*100
        else
            warning('We are just creating the spatial matrices.')
            Concentrated_ETP_interpolation_percentage = k/n_obs_model*100
        end
    end
end

% Plot if you want
% subplot(2,1,1);
% plt_01 = surf(ETP_save(:,:,1));
% set(plt_01,'LineStyle','none')
%
% subplot(2,1,2);
% plt_02 = surf(ETP_save(:,:,n_obs));
% set(plt_02,'LineStyle','none')

%% 6.0 - Groundwater Volume (SWMW like approach)

% Replenishing Coefficient

kr = (1/75*(sqrt(ksat/25.4))); % Replenishing rate (1/hr) (Check Rossman, pg. 110)
Tr = 4.5./sqrt(ksat/25.4); %  Recovery time (hr) Check rossman pg. 110
Lu = 4.*sqrt(ksat/25.4); % Inches - Uppermost layer of the soil
Lu = Lu*2.54/100; % Meters
k_out = (teta_sat - teta_i).*kr.*Lu*1000; % Rate of replenishing exfiltration from the saturated zone during recoverying times (mm/hr)
k_out = k_out(:); % Rate of aquifer replenishing (mm/hr)
k_out_max = k_out; % Maximum rate of replenishing (mm/hr)

%% 7.0 - Rainfall Model


% ------------ Rainfall Matrix ------------ %
if flag_rainfall == 0 % No rainfall
    rainfall_matrix = flag_rainfall*zeros(size(dem));
elseif  flag_spatial_rainfall == 1
    % Spatial Rainfall Case
    input_table = readtable('Input_Data.xlsx','Sheet','Rainfall_Spatial_Input');
    % Observations
    n_obs = sum((table2array(input_table(:,2))>=0)); % Number of observations
    n_max_raingauges = 50;
    time_step_spatial = table2array(input_table(7,2)) - table2array(input_table(6,2)); % min
    end_rain = (n_obs-1)*time_step_spatial;
    rainfall_spatial_duration = 0:time_step_spatial:(end_rain); % Rainfall data time in minutes

    % Rainfall Data
    for i = 1:n_max_raingauges
        rainfall_raingauges(:,i) = table2array(input_table(6:end,3 + (i-1)));
        coordinates(i,1) = table2array(input_table(3,3 + (i-1)));
        coordinates(i,2) = table2array(input_table(4,3 + (i-1)));
    end
    n_raingauges = sum(rainfall_raingauges(1,:) >= 0); % Number of raingauges
end


if flag_rainfall == 1 % We are modeing rainfall
    rain_outlet = zeros(n_steps,1); % Outlet Rainfall

    if flag_spatial_rainfall ~= 1
        % We need to calculate the step_rainfall from the rainfall data
        rainfall_data = readtable('Input_Data.xlsx','Sheet','Concentrated_Rainfall_Data');
        precipitation_data = table2array(rainfall_data(:,2));
        % We are assuming that the rainfall begins with the initial time of
        % the model
        step_rainfall = time2num(table2array(rainfall_data(2,1)) - table2array(rainfall_data(1,1)));
    else
        spatial_rainfall_maps = zeros(size(DEM,1),size(DEM,2),size(d,2)); % Maps of Rainfall
        % Spatial Rainfall
        if t == 0
            z = 1;
        else
            z = find(t >= rainfall_spatial_duration,1,'first'); % Duration
        end
        rainfall = rainfall_raingauges(z,1:n_raingauges)'; % Values of rainfall at t for each rain gauge
        x_grid = xulcorner + Resolution*[1:1:size(DEM_raster.Z,2)]'; % Pixel easting coordinates
        y_grid = yulcorner - Resolution*[1:1:size(DEM_raster.Z,1)]'; % Pixel northing coordinates
        x_coordinate = coordinates(1:n_raingauges,1); % Coordinates (easting) of each rain gauge
        y_coordinate = coordinates(1:n_raingauges,2); % Coordinates (northing) of each rain gauge
        [spatial_rainfall] = Rainfall_Interpolator(x_coordinate,y_coordinate,rainfall,x_grid,y_grid); % Interpolated Values
        spatial_rainfall_maps(:,:,1) = spatial_rainfall;
        step_rainfall = time_step_spatial;
    end
end

% Time Records for Rainfall
time_records_rainfall = [0:step_rainfall:tfinal]; % time in minutes

% Rainfall Time-Records
time_store_rainfall = time_records_rainfall*60./time_step; % number of steps necessary to reach the recording vector
time_store_rainfall(1) = 1;

%% 8.0 - Array Concatenation

% %%%% Creating concatenated 1D arrays %%%
n = n(:);
h_0 = h_0(:);
ksat = ksat(:);
dtheta = dtheta(:);
F_d = F_0(:);
psi = psi(:);
% Initial Inflow
[inflows] = non_lin_reservoir(Lambda,h_ef_w,h_0,Delta_x,Delta_y);
flow_outlet(1,1) = sum(inflows(pos)*(Delta_x*Delta_y)/(1000*3600)); % Initial Outflow
t_store = 1;

% Making Zeros at Outside Values
ksat = ksat(:).*idx_cells;
h_ef_w = h_ef_w.*idx_cells;
psi = psi.*idx_cells;

% Maximum Depth
max_depth = zeros(size(h_ef_w,1),1);

% Watershed Area
% --- It might change according to the raster resolution
drainage_area = sum(sum(double(DEM>0)))*resolution^2/1000/1000; % area in km2

% Read Reservoir Area
stage_area = table2array(readtable('Input_Data.xlsx','Sheet','Reservoir_Stage_Area'));
hmax = max((stage_area(:,1))); % Maximum reservoir area (You've got to enter it). The rationale with this is that rainfall occurs in the top area of the reservoir
[~,Area] = reservoir_area(hmax,stage_area,0); %


%% 9.0 - Solving the Water Balance Equations for the Watersheds

% Initial Diffusive or Kinematic Wave Models

% Determine the Water Surface Elevation Map
if flag_diffusive == 1
    wse = DEM + 1/1000*reshape(h_ef_w,[],size(DEM,2));
else
    wse = DEM;
end
% Call Flow Direction Sub
[f_dir,idx_fdir] = FlowDirection(wse,Delta_x,Delta_y,coord_outlet); % Flow direction matrix
% Call Dir Matrix
[Direction_Matrix] = sparse(Find_D_Matrix(f_dir,coord_outlet,Direction_Matrix_Zeros));
% Call Slope Sub
flag_slopeoutletmin = 1;
if flag_diffusive ~=1 && flag_slopeoutletmin == 1 % Only if we use the kinematic wave and we assume flag_slopeoutletmin == 1
    [slope] = max_slope8D(wse,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet); % wse slope
    cells = find(Direction_Matrix(pos,:) == 1);
    for i = 1:length(find(Direction_Matrix(pos,:) == 1))
        zzz = slope(:);
        slopes_outlet_inlet(i,1) = zzz(cells(1,i));
    end
    slope_outlet = min(slopes_outlet_inlet); % Average slope of Inlet Cells
end
[slope] = max_slope8D(wse,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet); % wse slope

% Calculate Lambda
Lambda = (resolution)*(1./n).*slope(:).^(0.5); % Lumped hydraulic properties
Lambda(idx_fdir ~= 1) = nan;

tic % Starts Counting Time
k = 0;
I_previous = F_d;
error_model = zeros(n_steps,1);
ETP = ETP(:);
z_prev = 0;
flooded_volume = nansum(Resolution^2*h_ef_w/1000.*idx_cells);   % Initial Flooded Vol.
S_t = nansum(Resolution^2.*F_d/1000.*idx_cells) + flooded_volume; % Initial Storage
error = 0;
% Main Loop
save('workspace_prior_watershed');
%%
load('workspace_prior_watershed')

dt = time_step/60;
time = 0;
time_end = n_steps*time_step/60; % Minutes
z_ETP_prev = 0;
t_store_hydrographs_prev = 0;
t_store_prev = 0;
tic
% Converting time_stores to time
time_store_rainfall_time = time_store_rainfall*time_step/60; % min
time_store_ETP_time = time_store_ETP*time_step/60; % min
time_store_time = time_store*time_step/60; % min
time_store_hydrographs = time_store_hydrographs*time_step/60; % min

inf_prev = 0;
flood_prev = 0;

while time <= (time_end - 1)
    k = k + 1; % Time-step index
    time = time + time_step/60; % Time in minutes (duration)
    % Rainfall Input
    z = find(time_store_rainfall_time <= time, 1,'last' ); % Position of rainfall
    if flag_spatial_rainfall ~=1
        if z > length(precipitation_data)
            z = length(precipitation_data);
            factor_rainfall = 0;
            warning('Not enough rainfall data. We are assuming that the rainfall is null.');
        else
            factor_rainfall = 1;
        end
    end
    if flag_rainfall == 1 && flag_spatial_rainfall ~=1
        i_0 = factor_rainfall*precipitation_data(z,1).*idx_cells; % Initial Rainfall for the next time-step
    elseif flag_rainfall == 1 && flag_spatial_rainfall == 1 && z > z_prev
        if time > end_rain - time_step/60
            warning('We are assuming the rainfall is null because the data is missing.')
            rainfall = zeros(n_raingauges,1); % Values of rainfall at t for each rain gauge
        else
            rainfall = rainfall_raingauges(z,1:n_raingauges)'; % Values of rainfall at t for each rain gauge
        end
        [spatial_rainfall] = Rainfall_Interpolator(x_coordinate,y_coordinate,rainfall,x_grid,y_grid); % Interpolated Values
        i_0 = spatial_rainfall(:).*idx_cells;  % mm/h
        spatial_rainfall_maps(:,:,z) = spatial_rainfall;  % Saving map
    elseif flag_rainfall == 0
        i_0 = 0.*idx_cells;
    end
    z_prev = z; % Previous value of z
    rain_outlet(k,1) = i_0(pos,1); % Outlet Rainfall

    % Shallow Water Model
%     if flag_diffusive == 1
%         % Predictor Corrector: First we calculate h for t + dt
%         % then we use this value to estimate a representative h, such that
%         % h^(k + 1/2) = h^k + h^{k+1}_*
%         % This depth is used to calculate the water surface elevation map
% 
%         % ---- Predictor Estimation of Water Depth ---- %
%         % ----- Diffusive Routine ---- %
%         depths = h_ef_w;
%         depths(idx_cells ~= 1) = nan;
%         wse = DEM + 1/1000*reshape(depths,[],size(DEM,2));
%         % Call Flow Direction Sub
%         [f_dir,idx_fdir] = FlowDirection(wse,Delta_x,Delta_y,coord_outlet); % Flow direction matrix
%         % Call Slope Sub
%         [slope] = max_slope8D(wse,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet); % wse slope
%         % Call Dir Matrix
%         [Direction_Matrix] = sparse(Find_D_Matrix(f_dir,coord_outlet,Direction_Matrix_Zeros));
%         % Calculate Lambda
%         Lambda = (resolution)*(1./n).*slope(:).^(0.5); % Lumped hydraulic properties
%         Mask = idx_fdir ~= 1;
%         Lambda(Mask) = 0;
%         [~,h_ef_w_predictor,~,~,~,~] = wshed_matrix(h_ef_w,h_0,inflows,time_step,Direction_Matrix,i_0,ksat,psi,dtheta,F_d,Lambda,Delta_x,Delta_y,ETP,idx_cells,k_out_max,pos);
% 
%         % ----- Diffusive Routine ---- %
%         % Calculation of Representative Water Depth
%         depths = (h_ef_w + h_ef_w_predictor)/2;
%         depths(idx_cells ~= 1) = nan;
%         wse = DEM + 1/1000*reshape(depths,[],size(DEM,2));
%         % Call Flow Direction Sub
%         [f_dir,idx_fdir] = FlowDirection(wse,Delta_x,Delta_y,coord_outlet); % Flow direction matrix
%         % Call Slope Sub
%         [slope] = max_slope8D(wse,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet); % wse slope
%         % Call Dir Matrix
%         [Direction_Matrix] = sparse(Find_D_Matrix(f_dir,coord_outlet,Direction_Matrix_Zeros));
%         % Calculate Lambda
%         Lambda = (resolution)*(1./n).*slope(:).^(0.5); % Lumped hydraulic properties
%         Mask = idx_fdir ~= 1;
%         Lambda(Mask) = 0;
%     end

    if flag_ETP == 1
        z_ETP = find(time_store_ETP_time <= time, 1,'last' ); % Position of rainfall
        % Refreshing ETP Data
        if k == 1
            % Do nothing
        elseif z_ETP > z_ETP_prev % Typically - Daily
            ETP = ETP_save(:,:,z_ETP);
            ETP = ETP(:);
            ETR_save(:,:,z_ETP) = reshape(ETR,[],size(DEM,2));
        end
        z_ETP_prev = z_ETP;
    end
  
    

    perc = time/(time_end)*100;

    % Check if a Steady Dry Condition is Reached
    %     if max(h_ef_w) == 0 && max(F_d) == 5 && flag_rainfall == 1 && flag_spatial_rainfall ~= 1
    %         z = find(time_store_rainfall <= k, 1,'last' ); % Position of rainfall
    %         zz = find(precipitation_data(z:end,1) > 0,1,'first') + z - 1;
    %         k = zz*step_rainfall*60/time_step - 2;
    %         t_new = find(time_store_ETP <=k,1,'last'); % Time that is being recorded in min
    %         I_previous = F_d;
    %         ETP = ETP_save(:,:,t_new); ETP = ETP(:);
    %         ETP_saving(t_new+1,1) = ETP(pos); % ETP value in mm/day
    %         ETR_saving(t_new+1,1) = ETR(pos); % ETP value in mm/day
    %     end

    %%% ------------- Mass Balance Routine  ------------------- %%%    
end
watershed_runningtime = toc/60; % Minutes

%% 10.0 - Watershed Post-Processing
% Call sub - Generate GIFs, Hydrographs, and so on
watershed_post_processing
save('workspace_after_watershed')
load workspace_after_watershed
zzz = [Depth_out + DEM(row,col),Qout_w];
%% 11.0 Modeling Reservoir + Channel Dynamics
% Pre allocating arrays
load workspace_after_watershed.mat
h_r = zeros(n_steps,1); % Depth in the reservoir (m)
i_reservoir = rain_outlet; % Rainfall in the reservoir (mm/h)
u_begin = ur_eq_t; % Initial control law

if flag_rainfall == 1
    if flag_spatial_rainfall ~= 1
        i_outlet = precipitation_data; % Constant rainfall in the catchment (mm/h)
    else
        i_outlet = squeeze(spatial_rainfall_maps(row,col,:)); % Constant rainfall in the catchment (mm/h)
    end
else
    i_outlet = 0;
end
% We recommend saving the workspace such as
% save('workspace_waterhsed');

%% 12.0 Agregating time-step to increase speed
% Agregating the Inflow to a larger time-step
% You can either enter with your observed outflow from the watershed and
% observed rainfall or use the ones calculated previously.
% To gain velocity, we can enter these values loading these files below:
global Qout_w Qout_w_horizon steps_control_horizon steps_horizon time_step n_steps Control_Vector Nvars i_reservoir h_c_0 h_r_t ur_eq_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP new_timestep Control_Interval Control_Horizon Prediction_Horizon g Cd flag_r number_of_orifices D l b relative_hmin Cds Lef hs porosity orifice_height x_length y_length roughness segments slope slope_outlet_channel max_iterations max_fun_eval n_randoms flag_c flag_r hmin s h_end slope_outlet_channel n_channel

% Downscalling Qout to model's time-step that might be different
x = time_store_hydrographs; % low sampled data
xq = time*60/time_step; % high sampled data
v = Qout_w; % result from low sampled data
Qout_w = interp1(x,v,xq,'pchip'); % high sampled data interpolation
% plot(x,v,'o',xq,vq2,':.');
% shg
% Downscalling ETP
if flag_ETP == 1
    x = time_store_ETP; % low sampled data
    xq = time*60/time_step; % high sampled data
    v = ETP_saving; % result from low sampled data
    if length(time_store_ETP) ~= 1
        E = max(interp1(x,v,xq,'pchip'),0); % high sampled data interpolation
    else
        E = ETP_saving(end);
    end
else
    E = zeros(1,n_steps);
end
% plot(x,v,'o',xq,E,':.');
% shg

%%%% Net Rainfall i'(k)
i_reservoir = i_reservoir - E'/24; % Rainfall intensity - Evaporation Intensity (mm/h)
% All of this previously done is to to avoid to run the watershed model, but
% you can run, of course, or you can also input it from HEC-HMS or other
% model

%%%% Agregating Time-Steps to the new_timestep for the reservoir model %%%%
flow_timestep = time_step;
inflow_timesteps = n_steps; % Number of time-steps using the watershed time-step
n_steps = (n_steps-1)*time_step/new_timestep; % adjusting the new number of time-steps
Qout_w_disagregated = Qout_w';
agregation_step = new_timestep/time_step; % number of time-steps in on agregation time-step

% Preallocating Arrays
n_steps_disagregation = ((inflow_timesteps-1)/agregation_step);
flow_agregated = zeros(n_steps_disagregation,1);
i_reservoir_agregated = zeros(n_steps_disagregation,1);

% Disagregating or Agreagating flows and rainfall
for i = 1:n_steps_disagregation
    if new_timestep >= time_step
        flow_agregated(i,1) =  mean(Qout_w_disagregated(1 + (i-1)*agregation_step:i*agregation_step,1));
        i_reservoir_agregated(i,1) = mean(i_reservoir(1 + (i-1)*agregation_step:i*agregation_step,1));
    else
        flow_agregated(i,1) =  Qout_w_disagregated(1 + floor((i-1)*agregation_step):ceil(i*agregation_step));
        i_reservoir_agregated(i,1) = i_reservoir(1 + floor((i-1)*agregation_step):ceil(i*agregation_step));
    end
end

% Defining updated outflows from the watershed, rainfall intensity and
% time_step
Qout_w = flow_agregated;
i_reservoir = i_reservoir_agregated;
time_step = new_timestep;

%% 13.0 Calling MPC control
% Let's clear variables we don't use to avoid computational burden
% We recommend loading the workspace containing all data up to line 561

clearvars -except Qout_w Qout_w_horizon steps_control_horizon steps_horizon time_step n_steps Control_Vector Nvars i_reservoir h_c_0 h_r_t ur_eq_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP new_timestep Control_Interval Control_Horizon Prediction_Horizon g Cd flag_r number_of_orifices D l b relative_hmin Cds Lef hs porosity orifice_height x_length y_length roughness segments slope slope_outlet_channel max_iterations max_fun_eval n_randoms flag_c flag_r hmin s h_end slope_outlet_channel n_channel a_noise b_noise m L Cd_HI u B pperson pfluid h_max_channel L_channel stage_area flag_channel_modeling flag_human_instability flag_detention_time detention_time_max q_max_star q_max_star_star alpha_p rho_u rho_h rho_hmax rho_c rho_HI max_res_level y_ref flag_optimization_solver u_static ur_eq
save('workspace_prior_MPC')
tic % Startg Counting MPC time

%%
load('workspace_prior_MPC')
% Disable Warnings
warning('off','all')
warning('query','all')
tic
% ------ Manual Inputs ----

% Delete Later
global Qout_w Qout_w_horizon steps_control_horizon steps_horizon time_step n_steps Control_Vector Nvars i_reservoir h_c_0 h_r_t ur_eq_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP new_timestep Control_Interval Control_Horizon Prediction_Horizon g Cd flag_r number_of_orifices D l b relative_hmin Cds Lef hs porosity orifice_height x_length y_length roughness segments slope slope_outlet_channel max_iterations max_fun_eval n_randoms flag_c flag_r hmin s h_end slope_outlet_channel n_channel a_noise b_noise ur_eq hr_eq flag_save_data m L Cd_HI u B_p pperson pfluid h_max_channel L_channel stage_area flag_channel_modeling flag_human_instability flag_detention_time detention_time_max q_max_star q_max_star_star alpha_p rho_u rho_h rho_hmax rho_c rho_HI max_res_level y_ref flag_gatecontrol ks_gatecontrol alpha_s_gatecontrol number_of_controls previous_control_gate previous_control_valve uv_eq us_eq uv_eq_t us_eq_t u_s u_v
max_fun_eval = 120;
q_max_star = 10;
q_max_star_star = 40;
detention_time_max = 18;

% Weights
rho_u = 1;
rho_hmax = 10^6;

Control_Interval = 60.00;
Control_Horizon = 120.00;
Prediction_Horizon = 720.00;
max_iterations = 50.00;
n_randoms = 5;
max_res_level = 6.8;
flag_gatecontrol = 1;
ks_gatecontrol = 27;
alpha_s_gatecontrol = 1.5;

if flag_gatecontrol == 1
    number_of_controls = 2; % Valves and Gates
else
    number_of_controls = 1; % Only the valves
end

% Water Quality Parameters
detention_time = 0; % Beginning it

flag_optimization_solver = 1;

% Patter Search Number of Initial Searches
number_of_initials = 5; % Number of initials for pattern search

% a) A few vector calculations
steps_control_horizon = Control_Horizon*60/time_step; % number of steps in the control horizon;
n_controls = Control_Horizon/Control_Interval; % number of controls to choose from an optimization problem
steps_horizon = Prediction_Horizon*60/time_step; % number of steps in one prediction horizon
Qout_w(length(Qout_w):(length(Qout_w)+steps_horizon),1) = 0; % adding more variables to the inflow to account for the last prediction horizon
i_reservoir(length(i_reservoir):(length(i_reservoir)+steps_horizon),1) = 0; % adding more variables to the inflow to account for the last prediction horizon
n_horizons = ceil((n_steps)/(Control_Horizon*60/time_step)); % number of control horizons
Control_Vector = [0:Control_Interval:Prediction_Horizon]*60/time_step;
Nvars = length(Control_Vector)*number_of_controls;
Vars_per_control = Nvars/number_of_controls; % Number of variables per controllable asset
U_random = zeros(Nvars,n_randoms);

% b) Objective Function
fun = @Optimization_Function;
OF_value = zeros(n_horizons,1);

% c) Optmizer - Solver
%%%% Interior Point %%%
options = optimoptions(@fmincon,'MaxIterations',max_iterations,'MaxFunctionEvaluations',max_fun_eval); % Previously
%%%% To increase the chances of finding global solutions, we solve the problem
% for n_randoms initial points %%%%

%%%% Estimate random initial points for the Random Search %%%%
matrix = rand(Nvars,n_randoms); % Only for interior points method with fmincon

% Matrices for FMINCON Optimization Problem
A = []; B = []; Aeq = [];Beq = []; Lb = zeros(Nvars,1); Ub = ones(Nvars,1);

%%% Prealocatting Arrays %%%
h_r_final = zeros(n_steps,1);

% Channel modeling
h_c_max_final = zeros(n_steps,1);
U = zeros(Nvars,1); % Concatenation of future control signals
if flag_gatecontrol ~=1
    u = ur_eq_t; % initial control (MAYBE DELETE)
else
    uv_eq_t = 0;
    us_eq_t = 0;
end

u_v = 0; % Valves Closed
u_s = 0; % Gates Closed
previous_control_valve = uv_eq_t;
previous_control_gate = us_eq_t;

% Orifice Properties
Aoc = pi()*D^2/4*number_of_orifices;
Aor = l*b*number_of_orifices;
if ((flag_c == 1) && (flag_r == 1))
    error('Please choose only one type of orifice')
elseif (flag_c == 1)
    D_h = D; % circular
    Ao = Aoc;
else
    D_h = 4*(l*b)/(2*(l+b)); % rectangular
    Ao = Aor;
end
Ko = Cd*Ao*sqrt(2*g);
check_detention = 0;
for i = 1:n_horizons
    perc = i/n_horizons*100;
    perc____timeremainhour = [perc, (toc/60/60)/(perc/100),max(Qout_w_horizon)]

    % Define inflow from the catchment during the prediction horizon
    time_begin = (i-1)*steps_control_horizon + 1; % step
    t_min = time_begin*time_step/60; %  time in min
    time_end = time_begin + steps_horizon;
    t_end = time_end*time_step/60; % final time of the current step
    time_end_saving = time_begin + steps_control_horizon-1; % for saving
    Qout_w_horizon = Qout_w(time_begin:time_end,1); % Result of the Watershed Model
    i_reservoir_horizon = i_reservoir(time_begin:time_end,1); % Result of the Rainfall Forecasting

    % Determining random initial points
    %%% Noise Generation - Prediction %%%
    % In case noise is generated in the states, you can model it by a
    % assuming a Gaussian noise with an average and variance specified
    % below
    average = 0.0; % average noise in (m)
    variance = 0; % variance in (m2). Remember that xbar - 2std has a prob of 93... %, std = sqrt(variance)


    %%% - FMINCON WITH RANDOM NUMBERS AS X0
    if flag_optimization_solver == 1
        for j=1:n_randoms
            % Orifice
            if u_v(end,1) == 0 % In case the previous control equals zero
                u0_v = matrix(1:(Vars_per_control),j); % Random numbers for orifice
                U_v = U(1:Nvars/2);
                U_s = U((Nvars/2 + 1):end);
                u0_v(1) = U(n_controls);
            else
                U_v = U(1:Nvars/2);
                U_s = U((Nvars/2 + 1):end);
                u0_v = max(U_v(n_controls)*(a_noise + matrix(1:(Vars_per_control),j)*(b_noise - a_noise)),0); % randomly select values within the adopted range
                u0_v = max(u0_v,0);
                u0_v(1) = U_v(n_controls); % the initial estimative is the previous adopted
            end
            % Gates
            if u_s(end,1) == 0 % In case the previous control equals zero
                u0_s = matrix(1:(Vars_per_control),j); % Random numbers for orifice
                u0_s(1) = U_s(n_controls);
            else
                u0_s = max(U_s(n_controls)*(a_noise + matrix((Vars_per_control+1):Nvars,j)*(b_noise - a_noise)),0); % randomly select values within the adopted range
                u0_s = max(u0_s,0);
                u0_s(1) = U_s(n_controls); % the initial estimative is the previous adopted
            end

            if max(Qout_w_horizon) > 0 % If the maximum predicted outflow is zero
                detention_time = 0;
                %%% Solving the Optimization Problem for the Random Initial
                %%% Points

                %%%% ------------ Fmincon Solution Use Below ------------------- %%%
                if flag_gatecontrol == 1
                    u0 =[u0_v; u0_s]; % Concatenating Decisions
                end
                [U,FVAL] = fmincon(fun,u0',A,B,Aeq,Beq,Lb,Ub,[],options);  % Fmincon solutions
                % ------------------------------------------------------------------------------
                OF(j) = FVAL; % saving the objective function value
                U_random(:,j) = U'; % saving the control vector
                position = find(OF == min(OF)); % position where the minimum value of OF occurs
                if length(position)>1
                    position = position(1);
                end
                % Saving OF value and Controls
                if flag_gatecontrol ~= 1
                    U = U_random(:,position); % Chosen control
                else
                    U = U_random(:,position); % Chosen control
                    U_v = U(1:n_controls);
                    U_s = U((Nvars/2 + 1):(Nvars/2 + n_controls));
                end
            end
        end

        % ---- Detention Time ---- %
        if max(Qout_w_horizon) == 0 % If the maximum predicted outflow is zero
            detention_time = detention_time + Control_Horizon/60; % hours
            if check_detention == 0 && h_r(end,1) > 0  % Saving the last depth
                h_r_end = h_r(end,1);
            elseif check_detention == 0 && h_r(end,1) == 0
                h_r_end = 1e-3; % 1 cm
            end
            % Detention Time Release
            if detention_time > detention_time_max
                check_detention = 1;
                % We start releasing water
                u_qmax_star = min(q_max_star/(Ko*sqrt(h_r_end)),1);
                U(1:Nvars/2,1) = u_qmax_star*(ones(Nvars/2,1)); % Open Partially
                U((Nvars/2+1):Nvars,1) = 0; % Close Gates
            else
                U = 0*(ones(Nvars,1)); % Close Valves and hold water
                U_v = U(1:n_controls);
                U_s = U((Nvars/2 + 1):(Nvars/2 + n_controls));
            end
        end

    elseif flag_optimization_solver == 0 % Pattern Search Approach
        for j=1:number_of_initials
            %%%%% - Search Algorithms - %%%%
            u0 = zeros(Nvars,1) + (j-1)*(1/(number_of_initials-1));
            if max(Qout_w_horizon) == 0 % If the maximum predicted outflow is zero
                detention_time = detention_time + Control_Horizon/60; % hours
                if check_detention == 0 && h_r(end,1) > 0  % Saving the last depth
                    h_r_end = h_r(end,1);
                elseif check_detention == 0 && h_r(end,1) == 0
                    h_r_end = 1e-3; % 1 cm
                end
                if detention_time > detention_time_max
                    check_detention = 1;
                    % We start releasing water
                    u_qmax_star = min(q_max_star/(Ko*sqrt(h_r_end)),1);
                    U = u_qmax_star*(ones(Nvars,1)); % Open 50% only
                else
                    U = 0*(ones(Nvars,1)); % Close Valves and hold water
                end
            else
                %%%% ------------ Global Search Solution, if you want to use ------------- %%%%
                %             rng default % For reproducibility
                %             ms = MultiStart('FunctionTolerance',2e-4,'UseParallel',true);
                %             gs = GlobalSearch(ms);
                %             gs.MaxTime = 60; % Max time in seconds
                %             gs.NumTrialPoints = 20;
                %             problem = createOptimProblem('fmincon','x0',u0','objective',fun,'lb',Lb,'ub',Ub);
                %             [U,FVAL] = run(gs,problem); % Running global search
                %%%% --------------- Pattern Search Solution ---------------- %%%%
                [U,FVAL] = patternsearch(fun,u0',A,B,Aeq,Beq,Lb,Ub,[],options); % Pattern search solutions
                OF(j) = FVAL;
                U_random(:,j) = U'; % saving the control vector
                position = find(OF == min(OF)); % position where the minimum value of OF occurs
                if length(position)>1 % More than 1 solution with same O.F
                    position = position(1);
                end
                % Saving OF value
                U = U_random(:,position); % Chosen control
                % ------------------------------------------------------------------------------
            end
        end
    else % Run With Known Valve Control - Static Operation
        U = u_static*(ones(Nvars,1)); % Here you can either choose to open the valves or close them, in case you want to increase detention time
        FVAL = Optimization_Function(U);
    end
    % Objective Function
    if detention_time > detention_time_max && max(Qout_w_horizon) == 0 % Releasing
        OF_value(i) = nan;
    elseif detention_time <= detention_time_max && max(Qout_w_horizon) == 0 % Holding
        OF_value(i) = nan;
    else % Optimizing
        if flag_optimization_solver ~= 1 && flag_optimization_solver ~= 0
            OF_value(i) = FVAL; % Value for known u(k) over time
        else
            OF_value(i) = OF(position); % We are solving the O.P problem
        end
    end

    if flag_gatecontrol ~= 1
        controls((i-1)*n_controls + 1:i*n_controls,1) = U(1:n_controls)';
    else
        controls((i-1)*n_controls + 1:i*n_controls,1) = U_v(1:n_controls)';
        controls((i-1)*n_controls + 1:i*n_controls,2) = U_s(1:n_controls)';
    end
    % Implement the Controls in the Plant and retrieving outputs
    %%% Run the Model with the estimated trajectory determined previously %%%
    % Disagregating u into the time-steps of the model
    for j=1:(steps_horizon)
        idx = find(Control_Vector < j,1,'last'); % U disagregated into time-steps
        if flag_gatecontrol ~= 1
            u(j,1) = U(idx);
        else
            U_v_total = U(1:(Nvars/2));
            U_s_total = U((Nvars/2 +1):end);
            u_v(j,1) = U_v_total(idx);
            u_s(j,1) = U_s_total(idx);
        end
    end
    previous_control_valve = U_v_total(n_controls);
    previous_control_gate = U_s_total(n_controls);
    %%% Noise Generation - Application %%%
    % In case noise is generated in the states, you can model it by a
    % assuming a Gaussian noise with an average and variance specified
    % below (This applies only to the plant)

    %%% Reservoir Plant %%%
    [h_r,out_r] = plant_reservoir(Qout_w_horizon,time_step,u,g,Cd,number_of_orifices,flag_c,D,flag_r,l,b,hmin,orifice_height,Cds,Lef,hs,porosity,average,variance,stage_area); % Reservoir Dynamics
    h_r_t = h_r(steps_control_horizon);
    if flag_gatecontrol ~=1
        ur_eq_t = U(n_controls); % Initial values for the next step
    else
        uv_eq_t = U_v_total(n_controls);
        us_eq_t = U_s_total(end);
    end

    %%% Channel Plant %%%
    if flag_channel_modeling == 1
        [max_water_level,h_c,out_c] = plant_channel(out_r,time_step,h_c_0,x_length,y_length,roughness,average,variance,segments,s,slope_outlet_channel,h_end); % Channel Dynamics
        h_c_0 = h_c(:,steps_control_horizon); % Initial Values for the next step
        % Saving Results
        h_c_max_final((time_begin:time_end_saving),1) = max(h_c(:,1:steps_control_horizon))';
        out_c_final(time_begin:time_end_saving,1) = out_c(1:steps_control_horizon,1);
    end

    %%% Saving Results %%%
    h_r_final(time_begin:time_end_saving,1) = h_r(1:steps_control_horizon,1);
    out_r_final(time_begin:time_end_saving,1) = out_r(1:steps_control_horizon,1);
    %     h_c_final(:,(time_begin:time_end_saving)) = h_c(:,1:steps_control_horizon);
end

% Enable Warnigns
warning('on','all')
warning('query','all')

% %% 12.0 Disagregating controls to the time-step unit
Control_Vector = [0:Control_Interval*60/time_step:n_steps];
for i=1:(n_steps-1)
    idx = find(Control_Vector <= i,1,'last'); % Position in U
    if flag_gatecontrol ~=1
        u(i,1) = controls(idx);
    else
        u_v(i,1) = controls(idx,1);
        u_s(i,1) = controls(idx,2);
    end
end


if flag_gatecontrol ~=1
    u_begin = ur_eq;
    u = [u_begin; u];
else
    u_begin = ur_eq;
    u_v = [uv_eq; u_v];
    u_s = [us_eq; u_s];
end
% graphs_wshed_reservoir_channel
time_MPC = toc;

save('workspace_after_MPC');

display_results_only_reservoir;
