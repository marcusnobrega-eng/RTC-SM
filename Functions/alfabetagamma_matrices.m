%% Numerical Jacobians Alfa and Beta
% 4/25/2021
% Developer - Marcus Nobrega
% Goal: Stablish a right criteria to define the jacobian matrices for the
% linear reservoir
function [alfa,beta,gamma] = alfabetagamma_matrices(Qin_t,Ko,ur_eq,hr_eq,ho,Ks,hs,h_t,alfa_1_function,alfa_2_function,beta_1_function,beta_2_function,D_h,flag_gatecontrol,gamma_2_function,uv_eq,us_eq,alpha_s_gatecontrol,ks_gatecontrol,h_stage)
% Evaluating the Jacobian with respect to h_t, u_t
% Assigning Values to each Symb. Variable
Qin = Qin_t;
u_v = uv_eq;
u_s = us_eq;
alpha_s = alpha_s_gatecontrol;
ks_gate = ks_gatecontrol;
hout = ho;
hsp = hs;
Kot = Ko;
Kst = Ks;
h = h_t;
% Orifice Minimum Height
%% Checking the correct alfa,beta,gamma functions
if h_t == 0
    pos = 1;
else
    pos = find(h_stage <= h_t,1,'last');
end

%% Solving the Jacobians Manually - It takes less time
if h_t < ho % ho represents the maximum between the orifice water level and h_min*Dh
    alfa = 0;
    beta = 0;
    gamma = 0;
elseif h_t < hs
    % Enter the Jacobian manually calculated or with the symbolic jacobian
    % for h < hs for alfa_1
    alfa = alfa_1_function{pos}(Kot,Qin,h,hout,u_v);
    beta = beta_1_function{pos}(Kot,h,hout);
    gamma = 0;
else
    % Enter the Jacobian manually calculated or with the symbolic jacobian
    % for h > hs, or alfa_2

    alfa = alfa_2_function{pos}(Kot,Qin,alpha_s,h,hout,hsp,ks_gate,u_s,u_v);
    beta = beta_2_function{pos}(Kot,h,hout);
    gamma = gamma_2_function{pos}(alpha_s,h,hsp,ks_gate); % Gate Control
end

