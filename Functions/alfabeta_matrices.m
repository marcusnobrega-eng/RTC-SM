%% Numerical Jacobians Alfa and Beta
% 4/25/2021
% Developer - Marcus Nobrega
% Goal: Stablish a right criteria to define the jacobian matrices for the
% linear reservoir
function [alfa,beta] = alfabeta_matrices(Qin_t,Ko,ur_eq,hr_eq,ho,Ks,hs,h_t,alfa_1_function,alfa_2_function,beta_1_function,beta_2_function,D_h)
% Evaluating the Jacobian with respect to h_t, u_t
% Assigning Values to each Symb. Variable
Qin = Qin_t;
u = ur_eq;
hout = ho;
hsp = hs;
Kot = Ko;
Kst = Ks;
h = h_t;
% Orifice Minimum Height
%% Solving the Jacobians Manually - It takes less time
if h_t < ho % ho represents the maximum between the orifice water level and h_min*Dh
    alfa = 0;
    beta = 0;
elseif h_t < hs
    % Enter the Jacobian manually calculated or with the symbolic jacobian
    % for h < hs for alfa_1
    alfa = alfa_1_function(Kot,Qin,h,hout,u);
    beta = beta_1_function(Kot,h,hout);
else
    % Enter the Jacobian manually calculated or with the symbolic jacobian
    % for h > hs, or alfa_2
    alfa = alfa_2_function(Kot,Kst,Qin,h,hout,hsp,u);
    beta = beta_2_function(Kot,h,hout);
end

