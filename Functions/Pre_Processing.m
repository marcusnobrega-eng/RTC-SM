%% Pre-processing
% Developer: Marcus Nobrega Gomes Junior
% 12/19/2023
% Pre-processing File
% Goal: Create the pre-processing file

%% Load ToolBoxes
addpath(genpath(Topo_path));

% Simulation Time
input_simulation_time = input_table(1:2,2);

% Plots and Recording Times
input_plots_and_recording = table2array(input_table(4:7,2));

% Adaptive Time-Stepping
input_adaptive = table2array(input_table(8:10,2));

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

% Adaptive Time-Stepping
Adaptive_Time_Stepping.alpha = input_adaptive(1);
Adaptive_Time_Stepping.t_min = input_adaptive(2);
Adaptive_Time_Stepping.t_max = input_adaptive(3);

% 2) Watershed Data

slope_outlet = input_watershed_data(1,1);
% ETP = input_watershed_data(2,1);
time_step = input_watershed_data(3,1);
flags.flag_diffusive = input_watershed_data(4,1); % Type of shallow water model
flags.flag_rainfall = input_watershed_data(5,1);
flags.flag_spatial_rainfall = input_watershed_data(6,1);
flags.flag_ETP = input_watershed_data(7,1);
flags.flag_spatial_ETP = input_watershed_data(8,1);
flags.flag_Rs = input_watershed_data(9,1); % What is it?
flags.flag_Ur = input_watershed_data(10,1); % What is it?
flags.flag_imposemin = input_watershed_data(11,1); % if == 1, we impose a minimum slope

% 3) Reservoir Data
g = input_reservoir_data(1,1);
Reservoir_Parameters.Cd = input_reservoir_data(2,1);
flags.flag_r = input_reservoir_data(3,1);
flags.flag_c = input_reservoir_data(4,1);
Reservoir_Parameters.number_of_orifices  = input_reservoir_data(5,1);
Reservoir_Parameters.D = input_reservoir_data(6,1);
Reservoir_Parameters.l = input_reservoir_data(7,1);
Reservoir_Parameters.b = input_reservoir_data(8,1);
Reservoir_Parameters.relative_Reservoir_Parameters.hmin = input_reservoir_data(9,1);
Reservoir_Parameters.Cds = input_reservoir_data(10,1);
Reservoir_Parameters.Lef = input_reservoir_data(11,1);
Reservoir_Parameters.hs = input_reservoir_data(12,1);
Reservoir_Parameters.porosity = input_reservoir_data(13,1);
Reservoir_Parameters.orifice_height = input_reservoir_data(14,1);
MPC_Control_Parameters.ur_eq_t = input_reservoir_data(15,1);
h_r_t = input_reservoir_data(16,1);
Reservoir_Parameters.hmin = input_reservoir_data(17,1);
flags.flag_reservoir_wshed = input_reservoir_data(19,1); % If == 1, we subtract the reservoir area * i in the outlet flow
u_static = input_reservoir_data(20,1);
flags.flag_gatecontrol = input_reservoir_data(21,1);


% 4) Channel Data
Channel_Parameters.x_length = input_channel_data(1,1);
Channel_Parameters.y_length = input_channel_data(2,1);
Channel_Parameters.n_channel = input_channel_data(3,1);
Channel_Parameters.segments = input_channel_data(4,1);
Channel_Parameters.h_end = input_channel_data(5,1);
s = input_channel_data(6,1);
Channel_Parameters.slope_outlet_channel = input_channel_data(7,1);
Channel_Parameters.h_c_0 = zeros(Channel_Parameters.segments,1);

% 5) MPC Control
MPC_Control_Parameters.new_timestep = input_mpc_data(1,1);
MPC_Control_Parameters.Control_Interval = input_mpc_data(2,1);
MPC_Control_Parameters.Control_Horizon = input_mpc_data(3,1);
MPC_Control_Parameters.Prediction_Horizon = input_mpc_data(4,1);
% 5.1) Interior Point Solver
MPC_Control_Parameters.max_iterations = input_mpc_data(5,1);
MPC_Control_Parameters.max_fun_eval = input_mpc_data(6,1);
MPC_Control_Parameters.a_noise = input_mpc_data(7,1);
MPC_Control_Parameters.b_noise = input_mpc_data(8,1);
% 5.1.1) Randoms
MPC_Control_Parameters.n_randoms = input_mpc_data(9,1);
% Initial Equilibrium Points for the Reservoir
MPC_Control_Parameters.ur_eq = input_mpc_data(10,1);
MPC_Control_Parameters.hr_eq = input_mpc_data(11,1);
% Data
flags.flag_save_data = input_mpc_data(12,1);
% Objective Function Constraint
flags.flag_channel_modeling = input_mpc_data(14,1);
flags.flag_human_instability = input_mpc_data(15,1);
flags.flag_detention_time = input_mpc_data(16,1);
MPC_Control_Parameters.detention_time_max = input_mpc_data(17,1);
MPC_Control_Parameters.q_max_star = input_mpc_data(18,1);
MPC_Control_Parameters.q_max_star_star = input_mpc_data(19,1);
MPC_Control_Parameters.alpha_p = input_mpc_data(20,1);
MPC_Control_Parameters.rho_u = input_mpc_data(21,1);
MPC_Control_Parameters.rho_h = input_mpc_data(22,1);
MPC_Control_Parameters.rho_hmax = input_mpc_data(23,1);
MPC_Control_Parameters.rho_c = input_mpc_data(24,1);
MPC_Control_Parameters.rho_hI = input_mpc_data(25,1);
MPC_Control_Parameters.max_res_level = input_mpc_data(26,1);
MPC_Control_Parameters.y_ref = input_mpc_data(27,1);
flags.flag_optimization_solver = input_mpc_data(28,1); % -1 static with no optimization, 0 patternsearch, 1 fmincon
flags.flag_gatecontrol = input_mpc_data(29,1);
Reservoir_Parameters.ks_gatecontrol = input_mpc_data(30,1);
Reservoir_Parameters.alpha_s_gatecontrol = input_mpc_data(31,1);

% 6) Human instability

m = input_human_instability_data(1,1); %m
L = input_human_instability_data(2,1); %m
Human_Instability_Parameters.Cd_HI = input_human_instability_data(3,1); %adm
Human_Instability_Parameters.u_p = input_human_instability_data(4,1); %adm
Human_Instability_Parameters.B_p = input_human_instability_data(5,1); %m
Human_Instability_Parameters.pperson = input_human_instability_data(6,1); %Kg/m³
Human_Instability_Parameters.pfluid = input_human_instability_data(7,1); %Kg/m³
Human_Instability_Parameters.h_max_channel = input_human_instability_data(8,1); %m
Human_Instability_Parameters.L_channel = input_human_instability_data(9,1); %m

% 7) Smoothening

flags.flag_smoothening = input_smoothening_data(1,1);
GIS_Parameters.min_area = input_smoothening_data(3,1); % km2
flags.flag_trunk = input_smoothening_data(2,1); % 0 All streams, 1 only main river
GIS_Parameters.tau = input_smoothening_data(4,1); % Between 0 and 1
GIS_Parameters.K_value = input_smoothening_data(5,1); % 1 - 20
GIS_Parameters.sl = input_smoothening_data(6,1); % m/m minimum slope for imposemin
% 8) GIS Parameters

flags.flag_resample = input_GIS_processing(1,1);
GIS_Parameters.resolution_resampled = input_GIS_processing(2,1);
flags.flag_fill = input_GIS_processing(3,1);

%% 1.0 - Defining simulation duration

time = (0:time_step:tfinal*60)/60; % time vector in min
time_simulation = time;
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
Lambda = (resolution)*(1./LULC_Properties.n).*slope.^(0.5); % Lumped hydraulic properties for k = 1
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
if flags.flag_ETP == 1 % We only read it if we are actually modeling it
    % Read ETP data
    if flags.flag_spatial_ETP == 1
        input_table = readtable('Input_Data_Test','Sheet','ETP_Spatial_Input'); % Input ETP data to interpolate from IDW method.
    else
        input_table = readtable('Input_Data_Test','Sheet','Concentrated_ETP_Data'); % Input ETP data to interpolate from IDW method.
    end
    Krs = 0.16; % Default.
    alpha_albedo_input = 0.23; % Default.
    flags.flag_U_R = 1; % if flag == 1, use RU. Else, use simplification method.
    flags.flag_Rs = 0; % if flag == 1, quantify Rs from the number of hour in the day. Else, quantify in function of the temperature.

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

    if flags.flag_spatial_ETP == 1

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
    if flags.flag_resample == 1
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
        [ETP] = Evapotranspiration(DEM_etp, avg_temp,max_temp,min_temp,day_of_year,lat,u2, ur, Krs, alpha_albedo_input, G, flags.flag_Rs, flags.flag_U_R);

        % -- Mask -- %
        ETP(neg_DEM) = nan;
        ETP_save(:,:,k) = ETP;
        if flags.flag_spatial_ETP == 1
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

Soil_Properties.kr = (1/75*(sqrt(Soil_Properties.ksat/25.4))); % Replenishing rate (1/hr) (Check Rossman, pg. 110)
Soil_Properties.Tr = 4.5./sqrt(Soil_Properties.ksat/25.4); %  Recovery time (hr) Check rossman pg. 110
Soil_Properties.Lu = 4.*sqrt(Soil_Properties.ksat/25.4); % Inches - Uppermost layer of the soil
Soil_Properties.Lu = Soil_Properties.Lu*2.54/100; % Meters
Soil_Properties.k_out = (Soil_Properties.dtheta).*Soil_Properties.kr.*Soil_Properties.Lu*1000; % Rate of replenishing exfiltration from the saturated zone during recoverying times (mm/hr)
Soil_Properties.k_out = Soil_Properties.k_out(:); % Rate of aquifer replenishing (mm/hr)
Soil_Properties.k_out_max = Soil_Properties.k_out; % Maximum rate of replenishing (mm/hr)

%% 7.0 - Rainfall Model


% ------------ Rainfall Matrix ------------ %
if flags.flag_rainfall == 0 % No rainfall
    rainfall_matrix = flags.flag_rainfall*zeros(size(dem));
elseif  flags.flag_spatial_rainfall == 1
    % Spatial Rainfall Case
    input_table = readtable('Input_Data_Test','Sheet','Rainfall_Spatial_Input');
    % Observations
    Rainfall_Properties.n_obs = sum((table2array(input_table(:,2))>=0)); % Number of observations
    Rainfall_Properties.n_max_raingauges = 50;
    Rainfall_Properties.time_step_spatial = table2array(input_table(7,2)) - table2array(input_table(6,2)); % min
    Rainfall_Properties.end_rain = (Rainfall_Properties.n_obs-1)*Rainfall_Properties.time_step_spatial;
    Rainfall_Properties.rainfall_spatial_duration = 0:Rainfall_Properties.time_step_spatial:(Rainfall_Properties.end_rain); % Rainfall data time in minutes

    % Rainfall Data
    for i = 1:Rainfall_Properties.n_max_raingauges
        Rainfall_Properties.rainfall_raingauges(:,i) = table2array(input_table(6:end,3 + (i-1)));
        Rainfall_Properties.coordinates(i,1) = table2array(input_table(3,3 + (i-1)));
        Rainfall_Properties.coordinates(i,2) = table2array(input_table(4,3 + (i-1)));
    end
    Rainfall_Properties.n_raingauges = sum(Rainfall_Properties.rainfall_raingauges(1,:) >= 0); % Number of raingauges
end


if flags.flag_rainfall == 1 % We are modeing rainfall
    rain_outlet = zeros(n_steps,1); % Outlet Rainfall

    if flags.flag_spatial_rainfall ~= 1
        % We need to calculate the step_rainfall from the rainfall data
        rainfall_data = readtable('Input_Data_Test','Sheet','Concentrated_Rainfall_Data');
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
            z = find(t >= Rainfall_Properties.rainfall_spatial_duration,1,'first'); % Duration
        end
        rainfall = Rainfall_Properties.rainfall_raingauges(z,1:Rainfall_Properties.n_raingauges)'; % Values of rainfall at t for each rain gauge
        x_grid = xulcorner + Resolution*[1:1:size(DEM_raster.Z,2)]'; % Pixel easting coordinates
        y_grid = yulcorner - Resolution*[1:1:size(DEM_raster.Z,1)]'; % Pixel northing coordinates
        x_coordinate = Rainfall_Properties.coordinates(1:Rainfall_Properties.n_raingauges,1); % Coordinates (easting) of each rain gauge
        y_coordinate = Rainfall_Properties.coordinates(1:Rainfall_Properties.n_raingauges,2); % Coordinates (northing) of each rain gauge
        [spatial_rainfall] = Rainfall_Interpolator(x_coordinate,y_coordinate,rainfall,x_grid,y_grid); % Interpolated Values
        spatial_rainfall_maps(:,:,1) = spatial_rainfall;
        step_rainfall = Rainfall_Properties.time_step_spatial;
    end
end

% Time Records for Rainfall
Rainfall_Properties.time_records_rainfall = [0:step_rainfall:tfinal]; % time in minutes

% Rainfall Time-Records
Rainfall_Properties.time_store_rainfall = Rainfall_Properties.time_records_rainfall*60./time_step; % number of steps necessary to reach the recording vector
Rainfall_Properties.time_store_rainfall(1) = 1;

%% 8.0 - Array Concatenation

% %%%% Creating concatenated 1D arrays %%%
LULC_Properties.n = LULC_Properties.n(:);
LULC_Properties.h_0 = LULC_Properties.h_0(:);
Soil_Properties.ksat = Soil_Properties.ksat(:);
Soil_Properties.dtheta = Soil_Properties.dtheta(:);
Soil_Properties.F_d = Soil_Properties.F_0(:);
Soil_Properties.psi = Soil_Properties.psi(:);
% Initial Inflow
[inflows] = non_lin_reservoir(Lambda,h_ef_w,LULC_Properties.h_0,Delta_x,Delta_y);
flow_outlet(1,1) = sum(inflows(pos)*(Delta_x*Delta_y)/(1000*3600)); % Initial Outflow
t_store = 1;

% Making Zeros at Outside Values
Soil_Properties.ksat = Soil_Properties.ksat(:).*idx_cells;
h_ef_w = h_ef_w.*idx_cells;
Soil_Properties.psi = Soil_Properties.psi.*idx_cells;

% Maximum Depth
max_depth = zeros(size(h_ef_w,1),1);

% Watershed Area
% --- It might change according to the raster resolution
drainage_area = sum(sum(double(DEM>0)))*resolution^2/1000/1000; % area in km2


% Read Reservoir Area
stage_area = table2array(readtable('Input_Data_Test','Sheet','Reservoir_Stage_Area'));
% Reservoir Stage-Varying Functions
[Area_Functions,Volume_Function,h_stage] = reservoir_stage_varying_functions(stage_area);
Reservoir_Parameters.hmax = max((stage_area(:,1))); % Maximum reservoir area (You've got to enter it). The rationale with this is that rainfall occurs in the top area of the reservoir
[~,Reservoir_Parameters.Area] = reservoir_area(Reservoir_Parameters.hmax,stage_area,h_stage,Area_Functions,Volume_Function);

%% 9.0 - Solving the Water Balance Equations for the Watersheds
% Initial Diffusive or Kinematic Wave Models

% Determine the Water Surface Elevation Map
if flags.flag_diffusive == 1
    wse = DEM + 1/1000*reshape(h_ef_w,[],size(DEM,2));
else
    wse = DEM;
end
% Call Flow Direction Sub
[f_dir,idx_fdir] = FlowDirection(wse,Delta_x,Delta_y,coord_outlet); % Flow direction matrix
% Call Dir Matrix
[Direction_Matrix] = sparse(Find_D_Matrix(f_dir,coord_outlet,Direction_Matrix_Zeros));
% Call Slope Sub
flags.flag_slopeoutletmin = 1;
if flags.flag_diffusive ~=1 && flags.flag_slopeoutletmin == 1 % Only if we use the kinematic wave and we assume flags.flag_slopeoutletmin == 1
    [slope] = max_slope8D(wse,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet); % wse slope
    cells = find(Direction_Matrix(pos,:) == 1);
    for i = 1:length(find(Direction_Matrix(pos,:) == 1))
        zzz = slope(:);
        slopes_outlet_inlet(i,1) = zzz(cells(1,i));
    end
    slope_outlet = max(min(slopes_outlet_inlet),slope_outlet); % Maximum value between inlets to the outlet and outlet slope
end
[slope] = max_slope8D(wse,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet); % wse slope

% Calculate Lambda
slope(slope<0) = 0;
Lambda = (resolution)*(1./LULC_Properties.n).*slope(:).^(0.5); % Lumped hydraulic properties
Lambda(idx_cells ~= 1) = nan; % Cells outside the catchment

tic % Starts Counting Time
k = 0;
I_previous = Soil_Properties.F_d;
error_model = zeros(n_steps,1);
ETP = ETP(:);
z_prev = 0;
flooded_volume = nansum(Resolution^2*h_ef_w/1000.*idx_cells);   % Initial Flooded Vol.
S_t = nansum(Resolution^2.*Soil_Properties.F_d/1000.*idx_cells) + flooded_volume; % Initial Storage
error = 0;
% Main Loop
save(label_watershed_prior_modeling);
% save('workspace_prior_watershed');