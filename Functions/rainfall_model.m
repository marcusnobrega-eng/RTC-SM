%% Rainfall Model
% Developer: Marcus Nobrega
% 4/23/2021
% Objective: Create a 2-D array with the rainfall for all cells in a given
% watershed
% Load the rainfall file in mm/h and insert the step_rainfall in min

function [i,i_0,time_real] = rainfall_model(rows,cols,time_store,number_of_records_hydrographs,record_time_maps,step_rainfall)
% We assume Hydrograph Output Time-Step is the same as Rainfall Input Time-Step
i = zeros(rows,cols,length(time_store)); % 2-D array with rainfall with multiple pages
dim = size(i); % size of rainfall
input_table = readtable('precipitation.xlsx'); % Read Rainfall from .xlsx file
prec = table2array(input_table(:,2));
time_real = table2array(input_table(:,1));
rainfall_end = length(prec)*step_rainfall; % Minutes
% ------- If you want add Noise at Rainfall --------- %
variance = 0.00; 
average = 0;
% Main for loop
for pp = 1:number_of_records_hydrographs % Filling i array in record_times_map intervals
    if pp*record_time_maps <= rainfall_end % We are inside of rainfall duration
        r = (1 + average + sqrt(variance).*randn(dim(1),dim(2))); % % Check random number generation on matlab
        idx = r < 0; r(idx) = 0;
        if record_time_maps <= step_rainfall
            i(:,:,pp) = r*prec(ceil((pp*record_time_maps/step_rainfall)));
        else
            ratio = ceil(record_time_maps/step_rainfall); % Case the rainfall is finer, we average it
            start = (pp-1)*ratio + 1;
            final = (pp)*ratio;
            i((1:end),(1:end),pp) = r*mean(prec(start:final));
        end
    else % We are outside of rainfall duration
        i((1:end),(1:end),pp) = 0;
    end
end
i_0 = i(:,:,1); % First entry of the array
end