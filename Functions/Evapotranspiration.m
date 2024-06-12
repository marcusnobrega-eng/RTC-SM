%% Process Potential Evapotranspiration
% Matheus Schroeder dos Santos
% 15/10/2022
% Objective:  Developed a 2-D array with the potential evapotranspiration for all cells in a given watershed
% Bibliography <Marco Antônio Fonseca Conceição. (2006). Roteiro de cálculo da evapotranspiração de referência pelo método de Penman-Monteith-FAO. Circular Técnica - EMBRAPA,(ISSN 1808-6810), 1–8>.

function [ETP] = Evapotranspiration(DEM,Temp,Temp_max,Temp_min,Day_year,latitude,U_2,U_R,Krs,alpha_albedo_input,G,flag_Rs,flag_U_R)
%% Input data
n_rows = size(DEM,1);
n_cols = size(DEM,2);

% ---- Default data following the bibliography ---- %

% U_2 = 1.6; % Wind speed at 2m height (m/s) - (e.g fixed in 2.0, as default)
% U_R = 81.6; % Relative humidity (%)
% alpha_albedo = 0.23; % Crop reference (Grass)
% Krs = 0.16; % Local coefficient for determining incident solar radiation (e.g fixed in 0.16 - continental areas, and 0.19 - coastal areas, as default).
% G = 0.6; % Total daily soil heat flux (MJm^-2/day)
a = 0.25;% Local coefficient for determining incident solar radiation (e.g fixed in 0.25, as default)
b = 0.50; % Local coefficient for determining incident solar radiation (e.g fixed in 0.50, as default)
theta_S_B_default = 4.903*(10^-9); % Constant of Stefan-Boltzmann (MJm^-2/day)

%% Processing data %%
% DEM
neg_DEM = DEM < 0;
inf_nan_MAPS = isinf(DEM) + isnan(DEM) + neg_DEM; % Logical array
idx = inf_nan_MAPS > 0;

% Day of the year
J = zeros(n_rows, n_cols);
for i = 1:n_rows
    for j = 1:n_cols
        J(i, j) = Day_year;
    end
end

% alfa_albedo
alfa_albedo = zeros(n_rows, n_cols);
for i = 1:n_rows
    for j = 1:n_cols
        alfa_albedo(i, j) = alpha_albedo_input;
    end
end

% teta_S_B
teta_S_B = zeros(n_rows, n_cols);
for i = 1:n_rows
    for j = 1:n_cols
        teta_S_B(i, j) = theta_S_B_default;
    end
end
%% Equations
% clearvars -except G teta_S_B alfa_albedo U_2 U_R J latitude Rs Tmin Tmax T n_cols n_rows DEM Krs a b

phi = latitude*pi/180; % latitude (rad)
dec_sol = 0.409*sin((2*pi/365)*J-1.39); % Solar declination (rad)
if (1-((tan(phi)).^2).*((tan(dec_sol)).^2)) <= 0
    X = 0.00001;
else
    X = (1-((tan(phi)).^2).*((tan(dec_sol)).^2));
end
ws = (pi/2)-atan(((-tan(phi)).*tan(dec_sol))./(X.^0.5)); % Hour angle at sunrise (rad)
dr = 1+0.033*cos(((2*pi)/365)*J); % Relative distance between Earth and Sun (rad)
Ra = (118.08/pi)*dr.*(ws.*sin(phi).*sin(dec_sol)+cos(phi).*cos(dec_sol).*sin(ws)); % Solar radiation at the top of the atmosphere (MJm^-2/day)
if flag_Rs == 1
    N = (24/pi).*ws;
    Rs_input = (a+(b*(n/N)))*Ra; % Quantify the incident solar radiation (MJm^-2/dia) using the number of hours with solar radiation
else
    Rs_input = Krs*Ra.*sqrt((Temp_max-Temp_min)); % Quantify the incident solar radiation (MJm^-2/dia) using the maximum and minimum temperatures
end 
Rso = (0.75+2*(10^-5)*DEM).*Ra; % Incident solar radiation without clouds (MJm^-2/dia)
Rns = (1-alfa_albedo).*Rs_input; % Shortwave radiation balance (MJm^-2/dia)
e_s = 0.6108*exp((17.27*Temp)./(Temp+237.3)); % Vapor saturation pressure (kPa)
if flag_U_R == 1
    e_a = e_s.*U_R/100; % Actual pressure of vapor (kPa) using the relative humidity
else
    e_a = 0.61*exp((17.27*Temp_min)./(Temp_min+237.3)); % Actual pressure of vapor (kPa) using the minimum temperature
end
Rnl = teta_S_B.*((((Temp_max+273.16).^4)+((Temp_min+273.16).^4))./2).*(0.34-0.14.*sqrt(e_a)).*(1.35*(Rs_input./Rso)-0.35); % Saldo da radiação de ondas longas (MJm^-2/dia)
Rn = Rns-Rnl; % Daily radiation balance (MJm^-2/dia)
delta = (4098*(0.6108*exp((17.27*Temp)./(Temp+237.3))))./((Temp+237.3).^2); % Slope of the vapor pressure curve versus temperature (kPa/C)
Patm = 101.3*((((293-0.0065*DEM)/293)).^5.26); % Local atmospheric pressure (kPa)
gama = 0.665*(10^-3)*Patm; % Psychometric coefficient (kPa/C)
ETP = (0.408.*delta.*(Rn-G)+(gama.*900.*U_2.*(e_s-e_a))./(Temp+273))./(delta+gama.*(1+0.34.*U_2));

% Mask %
idx = isinf(DEM) + isnan(DEM) + (DEM < 0);
idx(idx>0) = 1;
idx = logical(idx);
ETP(idx) = nan;

end
