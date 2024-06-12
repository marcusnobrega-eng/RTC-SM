%% Channel Routing (Non-Linear Dynamics)
% Developer - Marcus Nobrega
% 8/1/2021
% Objective: Develop a 1-D model of Manning's equation for a open Channel
% using 1-D gridded cells and calculating the maximum water level for each
% time-step
function [max_water_level,h_c,out_c] = plant_channel(out_r,time_step,h_c_0,x_length,y_length,roughness,average,variance,segments,s,slope_out_channel,h_end)
global Qout_w Qout_w_horizon steps_control_horizon steps_horizon time_step n_steps Control_Vector Nvars i_reservoir h_c_0 h_r_t ur_eq_t i_reservoir_horizon previous_control average variance slope_outlet tfinal record_time_maps ETP new_timestep Control_Interval Control_Horizon Prediction_Horizon g Cd flag_r number_of_orifices D l b relative_hmin Cds Lef hs porosity orifice_height x_length y_length roughness segments slope slope_outlet_channel max_iterations max_fun_eval n_randoms flag_c flag_r hmin s h_end slope_outlet_channel n_channel
% Time-step
delta_t = time_step; % Seconds;
x_cells = repmat(x_length,[segments,1]);
y_cells = repmat(y_length,[segments,1]);
roughness = repmat(n_channel,[segments,1]);
% Bottom Elevation
elevation = zeros(segments,1);
for i = 1:segments
    if i == 1
        elevation(1) = sum(y_cells*s)+ h_end;
        r = (1 + average + sqrt(variance).*randn()); % Noise
        roughness(i) = roughness(i)*r;
    else
        elevation(i) = elevation(i-1) - s*y_cells(i-1);
        r = (1 + average + sqrt(variance).*randn()); % Noise
        roughness(i) = roughness(i)*r;
    end
end
% Cells Area
cell_area = x_cells.*y_cells; % m2
% Initial Water Level
H_0 = h_c_0;
n_cells = length(x_cells);
slope = zeros(n_cells,1);
%% 1.0 - Topographic Slopes
for i = 1:(n_cells-1)
    slope(i) = (elevation(i) - elevation(i+1))/(y_cells(i)); % m/m
end
slope(n_cells) = slope_outlet_channel; % outlet
%% 2.0 - Hydraulic Radius and Area Function
h_radius = @(h,b) ((b.*h)./(b + 2*h)); % User defined hydraulic radius function
wet_area = @(h,b) (b.*h); % User defined cross section area function
outflow = @(h,b,n,slope) (1./roughness.*wet_area(h,b).*(h_radius(h,b).^(2/3)).*(slope.^(0.5))); % Manning's
h_c = zeros(n_cells,steps_horizon); h_c(:,1) = H_0; h_0_c = H_0;
%% Topology Relationships
% Direction Matrix
B_d_c = (-1)*eye(n_cells,n_cells); % Direction Matrix
% Slope(k+1) = Aslope*h(k) + Bslope
% Bslope = alfa*theta*elevation
% Aslope = alfa*theta
% alfa = tau*y_cells
% Beta = [0; 0; ... 0; s_outlet];
theta(segments) = 0; % Outlet Cell
theta = eye(segments); % Slope Conection Matrix
tau = 0.5*eye(segments); % Slope Distance Matrix
tau(segments,segments) = 0; % Outlet Cell
beta = zeros(n_cells,1); 
beta(n_cells) = slope_outlet_channel; % Outlet Slope
for i = 1:(n_cells - 1)
        B_d_c(i+1,i) = 1;
        theta(i,i+1) = -1; % [1-1 0 0; 0 1 -1 0 ...]
        tau(i,i+1) = 0.5;
end
alfa = (tau*y_cells).^-1; % 1/Horizontal Distance between cells (m^-1);
alfa(segments) = 0;
alfa = diag(alfa); % Transforming into a diagonal format
B_slope = alfa*theta*elevation + beta;
A_slope = alfa*theta;
%% 3.0 - Routing Flow
% Setting Initial Conditions
max_water_level = zeros(steps_horizon,1); % Maximum Water Level in the Channel (m)
reservoir_inflow = zeros(n_cells,steps_horizon); reservoir_inflow(1,1) = out_r(1);
inflow = outflow(h_c(:,1),x_cells,roughness,slope);
out_c = zeros(steps_horizon,1);
    for k = 1:(steps_horizon-1)
        % Implementing Water Balance  Numerical Constraint of no negative
        % depth - h(k+1) = max(h(k+1),0)
        h_c(:,k+1) = max(h_0_c + delta_t*(1./cell_area).*(B_d_c*inflow + reservoir_inflow(:,k)),0);
        h_0_c = h_c(:,k+1); % Depths for the next time step
        % Slopes Calculation assuming the Water Level of (k-1)
        slope = A_slope*h_0_c + B_slope;
        % Inflows for the next time step
        inflow = outflow(h_0_c,x_cells,roughness,slope); % Inflow to each cell
        out_c(k,1) = inflow(segments,1); % Outflow function at the outlet of the channel
        reservoir_inflow(1,k+1) = out_r(k+1); % Assuming the inflow from the cathcment only for the 1st cell
        max_water_level(k+1) = max(max(h_c(:,k+1)));% Maximum water level in the channel in (m)
    end
end