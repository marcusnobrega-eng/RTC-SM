%% Watershed, LULC, and Soil Maps for a Plane
% Developer: Marcus Nobregs
% This is the code to create a 1-D plan with slope s0

function [DEM,SOIL,LULC] = plane_watershed(pixelsize,s0,width,length,elevation_end)

% Input Data
% pixelsize = 0.02; % Meters
% s0 = 0.01; % m/m
% width = 1.48; % width of the plane
% length = 2.96; % length of the plane
% elevation_end = 0; % elevation of the outlet cells (m)
nx = width/pixelsize;
ny = length/pixelsize;
ncells = nx*ny;

% Elevation Data
DEM = zeros(ny,nx);
for i = 1:nx
    for j = 1:ny
        if j == 1
            DEM(j,i) = elevation_end;
        else
            DEM(j,i) = DEM(j-1,i) + s0*pixelsize;
        end
    end
end

SOIL = ones(size(DEM));

LULC = ones(size(DEM));

% Creating Rasters
SaveAsciiRaster(DEM,pixelsize,0,0,'DEM_Plane',-9999);
SaveAsciiRaster(SOIL,pixelsize,0,0,'SOIL_Plane',-9999);
SaveAsciiRaster(LULC,pixelsize,0,0,'LULC_Plane',-9999);
end

