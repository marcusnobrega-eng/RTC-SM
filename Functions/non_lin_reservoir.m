%% Flow Equation - Non-Linear Reservoir
% Developer: Marcus Nobrega
% 3/2/2023
% Goal: Apply manning's overland flow equation considering initial
% abstraction

function [Q,Qcms,v] = non_lin_reservoir(Lambda,d_0,h_0,Delta_x,Delta_y)
exp_flow = 5/3; % 5/3 for Manning's equation
Qcms = ((Lambda.*(max(0,(d_0/1000 - h_0/1000))).^(exp_flow))); % m3/s'
% Convert to m3/s to m/s and then to mm/h
Q = (Qcms./(Delta_x*Delta_y))*1000*3600; % mm/h
% Velocity
v = (Lambda./(0.5*(Delta_x + Delta_y))).*(max(0,(d_0/1000 - h_0/1000))).^(exp_flow - 1); % Velocity in m/s
end