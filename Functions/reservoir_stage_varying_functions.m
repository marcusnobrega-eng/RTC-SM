function [Area_Functions,Volume_Function,h_stage] = reservoir_stage_varying_functions(stage_area)
%% Reservoir Functions
% 22/11/2023
% Developer: Marcus Nobrega
% Goal - Define the stage-storage function of the reservoir
% You can enter a function using the example below:
% Area = @(h)(coef1*h.^(exp1) + coef2*h.^(exp2) ... );
% We just need to calculate a area function, area, volume_function, and
% volume
syms h

% Intitializing Outputs
Area_Function = nan; Area = nan; Volume = nan; pos = nan;

% Read Input Data
input_data = stage_area; % From xlsx input

% Variable Area - Check Example in EWRI Paper
n_points = size(input_data,1);

for i = 1:(n_points)
    if i == 1
        amin = input_data(1,2); % Minimum area for h = 0
        slope_area(1,i) = 1;
    end
    h_stage(1,i) = input_data(i,1); % Height for each stage
    area(1,i) = input_data(i,2); % Area for each stage

    if i ~= n_points
        slope_area(1,i) = (input_data(i+1,2) - input_data(i,2))/(input_data(i+1,1) - input_data(i,1));
    end
end

for i = 1:(n_points-1)
    if i == 1
        a = amin;
        slope = slope_area(1,i);
        stage = h_stage(1,i);
        Area_Functions{i} = (a + slope*(h - stage));
        Area_Functions{i} = matlabFunction(Area_Functions{i});
    else
        a = area(1,i);
        slope = slope_area(1,i);
        stage = h_stage(1,i);
        Area_Functions{i} = (a + slope*(h - stage));
        Area_Functions{i} = matlabFunction(Area_Functions{i});
    end
end
% if pos == 1 || isempty(pos) || pos == 5
%     ttt = 1;
% end
% if pos - 1 == 0 || pos - 1 > length(Area_Functions) || pos == 0
%     ttt = 1;
% end
% Area_Function = Area_Functions{pos-1};

% Volume - We analytically integrate area function to derive volume
% function
for i = 1:(n_points-1)
    if i == 1
        Vol_delta(i,1) = 0;
        Volume_Function{i} = int(Area_Functions{1},h); % Integral with respect to the depth
        Volume_Function{i} = matlabFunction(Volume_Function{i});
    else
        Vol_delta(i,1) = Vol_delta(i-1,1) + integral(Area_Functions{i-1},h_stage(i-1),h_stage(i));
        Volume_Function{i} = (int(Area_Functions{i},(h)) + Vol_delta(i,1));
        Volume_Function{i} = matlabFunction(Volume_Function{i});
    end
end
end
