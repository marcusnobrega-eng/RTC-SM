%% Reservoir Dynamics - Main Script
% 4/25/2021
% Note - I added rainfall in the area of the reservoir
% Developer: Marcus Nobrega
% Goal - Develop a linearized state-space represenntation of the Reservoir
% Dynamics
% Last Edited: 11/14/2021
function [h_r,Qout_r] = plant_reservoir(Qout_w_horizon,time_step,u,g,Cd,number_of_orifices,flag_c,D,flag_r,l,b,hmin,orifice_height,Cds,Lef,hs,porosity,average,variance,stage_area)
global Qout_w Area_Functions Volume_Function h_stage flags alfa_1_function alfa_2_function beta_1_function beta_2_function gamma_1_function gamma_2_function Reservoir_Parameters Channel_Parameters Qout_w_horizon MPC_Control_Parameters steps_horizon time_step n_steps Control_Vector Nvars i_reservoir Channel_Parameters h_r_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP g flag_r Reservoir_Parameters l b  hs roughness slope s m L u stage_area number_of_controls previous_control_gate previous_control_valve uv_eq us_eq uv_eq_t us_eq_t u_s u_v
%% Reservoir Parameters
h_eq = h_r_t; % m
% h_r_t = 0; % Initial Water Level in the Reservoir (m)
if flags.flag_gatecontrol ~=1
    ur_eq = MPC_Control_Parameters.ur_eq_t; % fuzzy-logic (0-1), 1 is fully opened, 0 is fully closed
else
    uv_eq = uv_eq_t;
    us_eq = us_eq_t;
end
T = time_step; % seconds
Aoc = pi()*D^2/4*number_of_orifices;
Aor = Reservoir_Parameters.l*Reservoir_Parameters.b*number_of_orifices;
if ((flags.flag_c == 1) && (flags.flag_r == 1))
    error('Please choose only one type of orifice')
elseif (flags.flag_c == 1)
    D_h = D; % circular
    Ao = Aoc;
else
    D_h = 4*(Reservoir_Parameters.l*Reservoir_Parameters.b)/(2*(Reservoir_Parameters.l + Reservoir_Parameters.b)); % rectangular
    Ao = Aor;
end
h_flow = max(hmin*D_h,orifice_height);
% Inflow
Qin = Qout_w_horizon; % m3/s
% Flow Equation
outflow_eq = @(Ko,Ks,h,hs,ho,u_v_eq,u_s_eq) (Ko.*u_v_eq.*sqrt(max(h - h_flow,0)) + max(u_s_eq*Reservoir_Parameters.ks_gatecontrol.*(h - Reservoir_Parameters.hs),0).^(Reservoir_Parameters.alpha_s_gatecontrol));

%% Reservoir Area
[~,Area,~,~] = reservoir_area(h_eq,stage_area,0); % A = f(h,flags.flag_volume) (handle) Function. If flags.flag_volume == 1, we don't integrate vol functions and it is faster

%% Calculating Constants
Ko = Cd*Ao*sqrt(2*g);
% Gate Operation
Ks = Reservoir_Parameters.ks_gatecontrol;
%% Symbolic Alfa and Beta Matrices (we took that out to gain velocity)
% gamma_1_function = [];
% [alfa_1_function,alfa_2_function,beta_1_function,beta_2_function,gamma_1_function,gamma_2_function] = symbolic_jacobians(stage_area,flags.flag_gatecontrol); %

%% Determining Jacobian Matrices
Qin_t = Qin(1);
i_0 = i_reservoir_horizon(1,1);
[alfa,beta,gamma] = alfabetagamma_matrices(Qin_t,Ko,MPC_Control_Parameters.ur_eq,h_eq,h_flow,Ks,Reservoir_Parameters.hs,h_r_t,alfa_1_function,alfa_2_function,beta_1_function,beta_2_function,D_h,flags.flag_gatecontrol,gamma_2_function,uv_eq,us_eq,Reservoir_Parameters.alpha_s_gatecontrol,Reservoir_Parameters.ks_gatecontrol);
epsilon = (1/(Area*porosity)*(Qin_t - outflow_eq(Ko,Ks,h_eq,Reservoir_Parameters.hs,h_flow,uv_eq, us_eq))) + i_0/1000/3600; % Adding Rainfall - Evaporation
epsilon(isnan(epsilon))=0; % Cases where the area might be equals 0
%% Discretizing the System
A = (1 + T*alfa);
B = (T*beta);
C = (T*gamma);
fi = (T*epsilon - T*alfa*h_eq - T*beta*uv_eq - T*gamma*us_eq);

% h(k+1) = A*h(k) + B*u + fi(k)
h = zeros(steps_horizon,1);
h(1,1) = h_r_t;
%% Main for Loop
for k = 1:(steps_horizon-1)
    % Generate a Gaussian Noise (if you want to generate, please uncomment)
%     [g_noise] = gaussian_noise_generator(variance,average);
    g_noise = 0;
    h(k+1) = max(A*h(k) + B*u_v(k,1) + C*u_s(k,1) + fi + g_noise ,0); % Constraint to make water level positive

    if h(k+1) > max(stage_area(:,1))
        warning('Overflow. We are assuming that the reservoir depth is the maximum depth.')
        h(k+1) = max(stage_area(:,1));
    end
    Qin_t = Qin(k+1);
    i_0 = i_reservoir_horizon(k,1); % Initial Rainfall for the next time-step
    h_t = h(k);
    % New Operation Point
    h_eq = h(k);
    uv_eq = u_v(k); % Valve
    us_eq = u_s(k); % Orifice

    % Jacobians
    if isinf(h_t) || isnan(h_t)
        error('Inf Depths or Nan Depths')
    end

    u_eq = [];
    [alfa,beta,gamma] = alfabetagamma_matrices(Qin_t,Ko,u_eq,h_eq,h_flow,Ks,Reservoir_Parameters.hs,h_t,alfa_1_function,alfa_2_function,beta_1_function,beta_2_function,D_h,flags.flag_gatecontrol,gamma_2_function,uv_eq,us_eq,Reservoir_Parameters.alpha_s_gatecontrol,Reservoir_Parameters.ks_gatecontrol);
    out_r = outflow_eq(Ko,Ks,h_eq,Reservoir_Parameters.hs,h_flow,uv_eq,us_eq);

    [~,Area,~,~] = reservoir_area(h_eq,stage_area,0); % A = f(h,flag_volume) (handle) Function. If flag_volume == 1, we don't integrate vol functions and it is faster
    epsilon = (1/(Area*porosity)*(Qin_t - out_r)) + i_0/1000/3600;
    epsilon(isnan(epsilon))=0;
    % New State-Space
    A = (1 + T*alfa);
    B = (T*beta);
    C = T*gamma;
    fi = (T*epsilon - T*alfa*h_eq - T*beta*uv_eq - T*gamma*us_eq);
end
Qout_r = outflow_eq(Ko,Ks,h,Reservoir_Parameters.hs,h_flow,u_v,u_s);
h_r = h;
end