%% Symbolic Jacobian Matrices
% 4/26/2021
% Developer: Marcus Nobrega
% Goal - Define symbolic jacobian matrices for a given system of equations
% Last Editted: 11/08/2021
function [alfa_1_function,alfa_2_function,beta_1_function,beta_2_function,gamma_1_function,gamma_2_function] = symbolic_jacobians(stage_area,flag_gatecontrol)
syms Qin Kot u u_v u_s h hout Kst hsp ks_gatecontrol alpha_s_gatecontrol

[Area_Functions,~] = reservoir_stage_varying_functions(stage_area); % Functions of area in terms of stage

for i = 1:length(Area_Functions) % For all stages
    Area_Function = Area_Functions{i}; % Current area function
    % Phase 1 (h < hs)
    h_dot_eq =  (1/sym(Area_Function)).*(Qin - (Kot.*u_v.*sqrt(h - hout))); % dh/dt
    alfa_1{i} = jacobian(h_dot_eq,h); % Partial with respect to h
    
    beta_1{i} = jacobian(h_dot_eq,u_v); % Partial with respect to u_v
    gamma_1{i} = jacobian(h_dot_eq,u_s); % Partial with respect to u_s
    
    
    % Phase 2 (h >= hs)
    
    h_dot_eq =  (1/sym(Area_Function)).*(Qin - Kot.*u_v.*sqrt(h - hout)  -(u_s.*ks_gatecontrol.*((h - hsp)^(alpha_s_gatecontrol)))); % Spillway Coefficient
    
    alfa_2{i} = jacobian(h_dot_eq,h); % Partial with respect to h
    
    % Jacobians
    
    beta_2{i} = jacobian(h_dot_eq,u_v); % Partial with respect to u_v
    gamma_2{i} = jacobian(h_dot_eq,u_s); % Partial with respect to u_s
    
    
    % Functions - Creating alfa and beta function for each case
    alfa_1_function{i} = matlabFunction(alfa_1{i});
    alfa_2_function{i} = matlabFunction(alfa_2{i});
    beta_1_function{i} = matlabFunction(beta_1{i});
    beta_2_function{i} = matlabFunction(beta_2{i});
    
    gamma_1_function{i} = matlabFunction(gamma_1{i});
    gamma_2_function{i} = matlabFunction(gamma_2{i});
end
