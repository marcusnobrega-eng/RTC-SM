%% Maximum 8-D slope calculation
% Marcus Nobrega
% 4/12/2021
% Calculation of maximum 8-D slope knowing the grid cell elevations (DEM)
% Last update: 11/02/2021

function [slope] = max_slope8D(DEM,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet)
% This function calculates the maximum 8-D slope
n_rows = size(DEM,1);
n_cols = size(DEM,2);
slope = zeros(n_rows,n_cols);
distance_diagonal = sqrt(Delta_x^2 + Delta_y^2);
for i = 1:n_rows
    for j = 1:n_cols
        if DEM(i,j) == inf || isnan(DEM(i,j)) % Not part of the watershed
            slope(i,j) = 0;
        else
            cell_elevation = DEM(i,j); % elevation of the cell (i,j)
            if i == coord_outlet(1,1) && j == coord_outlet(1,2)
                slope(i,j) = slope_outlet;
            else
                if cell_elevation < 0
                    f_dir(i,j) = nan;
                else
                    if f_dir(i,j) == 1
                        slope(i,j) = (cell_elevation - DEM(i,j+1))/Delta_x ; % m/m to right
                    elseif f_dir(i,j) == 2
                        slope(i,j) = (cell_elevation - DEM(i+1,j+1))/distance_diagonal; % m/m to southeast
                    elseif f_dir(i,j) == 4
                        slope(i,j) = (cell_elevation - DEM(i+1,j))/Delta_y; % m/m to south
                    elseif f_dir(i,j) == 8
                        slope(i,j) = (cell_elevation - DEM(i+1,j-1))/distance_diagonal; % m/m to southwest
                    elseif f_dir(i,j) == 16
                        slope(i,j) = (cell_elevation - DEM(i,j-1))/Delta_x; % m/m to west
                    elseif f_dir(i,j) == 32
                        if i == 1 || j == 1
                            ttt = 1;
                        end
                        slope(i,j) = (cell_elevation - DEM(i-1,j-1))/distance_diagonal; % m/m to northwest
                    elseif f_dir(i,j) == 64
                        slope(i,j) = (cell_elevation - DEM(i-1,j))/Delta_y; % m/m to north
                    elseif f_dir(i,j) == 128
                        slope(i,j) = (cell_elevation - DEM(i-1,j+1))/distance_diagonal; % m/m to northeast
                    end
                end
            end
        end
    end
end
if min(min(slope)) < 0
    idx = slope >= 0;
    idx2 = slope <= 0;
    zzz = slope;
    zzz(idx) = 0;
    surf(zzz)
    xlabel('x (m)','interpreter','latex')
    ylabel('y (m)','interpreter','latex')
    colorbar
    colormap('jet')
    view(30,60);
    error('Please make sure your DEM is filled since we have negative slopes.')
end
end