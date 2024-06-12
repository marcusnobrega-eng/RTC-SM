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
global Qout_w flags Qout_w_horizon GIS_Parameters steps_horizon time_step n_steps Channel_Parameters Control_Vector Nvars i_reservoir h_r_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP g  D Reservoir_Parameters Reservoir_Parameters roughness slope s m L Human_Instability_Parameters u stage_area MPC_Control_Parameters Reservoir_Parameters number_of_controls previous_control_gate previous_control_valve uv_eq us_eq uv_eq_t us_eq_t u_s u_v

%% Label for Input Data
Input_Data_Label = 'Input_Data_Test.xlsx';

%% Reading Main Input Data
input_table = readtable(Input_Data_Label,'Sheet','Input_Main_Data_RTC'); % Reading Input data from Excel

%% Rasters Path
input_data = readtable(Input_Data_Label,'Sheet','Input_Main_Data_RTC');
Topo_path = string(table2array(input_data(1,29)));
DEM_path = string(table2array(input_data(2,29)));
LULC_path = string(table2array(input_data(3,29)));
SOIL_path = string(table2array(input_data(4,29)));
Warmup_path = string(table2array(input_data(5,29)));

%% Labels for workspaces
label_watershed_prior_modeling = 'workspace_prior_watershed';
label_watershed_post_processing = 'workspace_after_watershed';
label_MPC_prior = 'workspace_MPC_prior';
label_MPC_after = 'workspace_MPC_after';

save(label_watershed_prior_modeling);

%% Pre-Processing 
Pre_Processing

%% Watershed Model - Script
Watershed_Script

%% Watershed Post-Processing
% Call sub - Generate GIFs, Hydrographs, and so on
watershed_post_processing
% We recommend saving the workspace such as
save(label_watershed_post_processing);
load label_watershed_post_processing
zzz_stage_discharge = [Depth_out + DEM(row,col),Qout_w];
zzz_stage_discharge(1,3) = watershed_runningtime;

%% MPC Code
MPC_Code; % MPC Script