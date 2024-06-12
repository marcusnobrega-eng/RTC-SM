%% Genetic Algorithm Optimization Approach
% 8/1/2021
% Developer: Marcus Nobrega
% Goal: The Objective Function of the MPC controller
% Notes: You can write an objective function as a non-linear combination
% within control deviations (dU), water depths in the reservoir (x_r), and
% maximum water levels in the channel and in the reservoir

function [OF] = Optimization_Function(U)
global Qout_w Area_Functions Volume_Function h_stage flags alfa_1_function alfa_2_function beta_1_function beta_2_function gamma_1_function gamma_2_function Reservoir_Parameters Channel_Parameters Qout_w_horizon MPC_Control_Parameters steps_horizon time_step n_steps Control_Vector Nvars i_reservoir Channel_Parameters h_r_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP g flag_r Reservoir_Parameters l b  hs roughness slope s m L u stage_area number_of_controls previous_control_gate previous_control_valve uv_eq us_eq uv_eq_t us_eq_t u_s u_v
%% O.F parameters

%%%% Function %%%%%
%   rho_u*dU*dU' + rho_x*x_r^T*x_r + rho_hmax*(max(max_hc - y_ref,0)) + rho_c*(max(max_hr - max_res_level,0));
% dU = [du(1), du(2) ... du(Np-1)]'
% du is delta U. For instance: du(t) = u(t+1) - u(t)

u_v = zeros(steps_horizon,1);
u_s = u_v;
dU_v = zeros(steps_horizon-1,1);
dU_s = zeros(steps_horizon-1,1);
%% Disagregating the Control into the time-steps
if flags.flag_gatecontrol == 1
    U_v = U(1:(Nvars/2));
    U_s = U((Nvars/2)+1:end);
else
    U_v = U(1:(Nvars));
    U_s = U(1:(Nvars));
end

% Orifice
for i=1:(steps_horizon)
    % Disagregating the control U into u
    idx = find(Control_Vector < i,1,'last'); % Position in U
    u_v(i,1) = U_v(idx);
    if i == 1
        dU_v(i) = u_v(i) - previous_control_valve;
    else
        dU_v(i,1) = u_v(i) - u_v(i-1);
    end
end

% Gate
for i=1:(steps_horizon)
    % Disagregating the control U into u
    idx = find(Control_Vector < i,1,'last'); % Position in U
    u_s(i,1) = U_s(idx);
    if i == 1
        dU_s(i) = u_s(i) - previous_control_gate;
    else
        dU_s(i,1) = u_s(i) - u_s(i-1);
    end
end
%% Call Scripts for the Reservoir and for the Channel
% Calls the Reservoir Plant
[x_r,out_r] = reservoir_dynamics(Qout_w,time_step,u,g,Reservoir_Parameters.Cd,Reservoir_Parameters.number_of_orifices,flags.flag_c,Reservoir_Parameters.D,flags.flag_r,Reservoir_Parameters.l,Reservoir_Parameters.b,Reservoir_Parameters.hmin,Reservoir_Parameters.orifice_height,Reservoir_Parameters.Cds,Reservoir_Parameters.Lef,Reservoir_Parameters.hs,Reservoir_Parameters.porosity,average,variance,stage_area,flags.flag_gatecontrol,u_v,u_s);
max_hr = max(x_r); % Max water level in the reservoir in meters
% ------- If you don't want to model the channel, you should deactivate below
if flags.flag_channel_modeling == 1
    [~,h_c] = channel_dynamics(out_r,time_step,h_c_0);
    max_hc = max(max(h_c)); % Max water level in the channel
end
%% Objective Function
% Original - {deviations in control} + {deviations in res. w. level) +
% {deviations in channel w.l) + penalizations in res and chan w.l
% Changed - We take out channel constraint and include a penalization in
% flows larger than a threshold
% q_max_star = 4; % Threshold for relatively small events in m3/s
% q_max_star_star = 4; % Threshold for relatively large events in m3/s
% alfa_p = 0.5; % Peak flow factor


% Flood Mitigation - Penalizations
if max(Qout_w_horizon) < MPC_Control_Parameters.q_max_star % Release the flows, if required, but not so much
    % ---- Minor Flood Forecast ---- %    
    % We want to mitigate alpha_p of the incoming peak flow, at least
    max_outflow = MPC_Control_Parameters.alpha_p*max(Qout_w_horizon); % m3/s - we are assuming we reduce at least alfa_p of all flows
    rho_outflow = 10*(MPC_Control_Parameters.rho_u); % Weight on maximum reservoir outflow in this case
else 
% ---- Large Flood Forecast ---- %    
    max_outflow = MPC_Control_Parameters.q_max_star; % m3/s - We want to minimize as best as we can
    rho_outflow = 100*(MPC_Control_Parameters.rho_u); % Weight on maximum reservoir outflow    
end

% ---- Very Large Flood Forecast --- %
if max(Qout_w_horizon) > MPC_Control_Parameters.q_max_star_star
    rho_outflow_penalization = 1000*(MPC_Control_Parameters.rho_u);
else
    rho_outflow_penalization = 0;
end
% ----------- Costs -----------------%
% Control Signal Variation
signal_cost = MPC_Control_Parameters.rho_u*norm(dU_v,2)^2 + MPC_Control_Parameters.rho_u*norm(dU_s,2)^2; % Be carefull here, we are adding the weir

% Reservoir Level Variation
% res_lev_cost = rho_x*norm(x_r,2)^2;

% Maximum Reservoir Level
res_lev_max_cost = MPC_Control_Parameters.rho_u*(max(max_hr - MPC_Control_Parameters.max_res_level,0));

% Outflow Penalization
max_res_outflow_cost = rho_outflow*(max(max(out_r - max_outflow,0)));

% Large Inflow Forecast Penalization
max_res_outflow_cost_penalization = rho_outflow_penalization*(max(max(out_r - MPC_Control_Parameters.q_max_star_star,0)));

% Channel Maximum Level
if flags.flag_channel_modeling == 1
    chan_max_lev_cost = MPC_Control_Parameters.rho_c*(max(max_hc - MPC_Control_Parameters.y_ref,0));
else
    chan_max_lev_cost = 0;
end

% Human Instability
if flags.flag_human_instability == 1
    human_instab_cost = MPC_Control_Parameters.rho_HI*max(0,instability);
else
    human_instab_cost = 0;
end

% --- Objective Function % ---
[OF] = signal_cost + res_lev_max_cost + max_res_outflow_cost + max_res_outflow_cost_penalization + chan_max_lev_cost + human_instab_cost;
% [OF] = signal_cost + res_lev_cost + res_lev_max_cost + max_res_outflow_cost + chan_max_lev_cost + human_instab_cost;
end