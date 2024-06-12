%% Reservoir Area
% 04/25/2021
% Developer: Marcus Nobrega
% Goal - Define the stage-storage function of the reservoir
% You can enter a function using the example below:
% Area = @(h)(coef1*h.^(exp1) + coef2*h.^(exp2) ... );
function [Area,Volume] = reservoir_area(h,stage_area,h_stage,Area_Functions,Volume_Functions)
% We just need to calculate a area function, area, volume_function, and
% volume

% Intitializing Outputs
Area = nan; Volume = nan; pos = nan;

% Read Input Data
input_data = stage_area; % From xlsx input

% Variable Area - Check Example in EWRI Paper
n_points = size(input_data,1);

for i = 1:(n_points)
    if h <= h_stage(1,i) && isnan(pos)
        if h == h_stage(1,i)
            pos = 2; % Finding the right stage for the entered value of h
        else
            pos = i; % Finding the right stage for the entered value of h
        end
    elseif h > max(input_data(:,1)) % Larger than the max
        error('Water depth larger than the maximum water depth. Overflow!!!')
    end
end

Area_Function = Area_Functions{pos-1}; % Choosing the right function
Area = Area_Function(h); % Value of the area

% Volume - We analytically integrate area function to derive volume
% function

Volume_Function = Volume_Functions{pos-1}; % Choosing the right function

Volume = Volume_Function(h); % m3 in terms of h in (m)

end