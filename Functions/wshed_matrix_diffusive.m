%% Watershed Finite-Difference Scheme Model
% Developer: Marcus Nobrega
% 4/23/2021
% Goal: Define a Model that Calculates the flow in the Outlet of a
% Watershed in a Matrix Fashion
% Observation: We added k_out to consider ground water

function [F_d,h_ef_w,inflows,ETR,k_out,v] = wshed_matrix_diffusive(d_0,h_0,inflows,time_step,i_0,ksat,psi,dtheta,I_0,Lambda,Delta_x,Delta_y,ETP,idx_fdir,k_out,coord_outlet,DEM,n,slope_outlet,Direction_Matrix_Zeros,resolution)
% Faster way
% Solving the discretized water balance in each cell
% d(k+1) = d(k) + T*(i(k) + (qin - qout) - inf)

h_ef_w = 0*d_0;

% Boundary Conditions
idx_fdir = logical(idx_fdir);
idx_not_fdir = idx_fdir == 0;
h_ef_w(idx_not_fdir) = nan;
I_0(idx_not_fdir) = nan;
k_out(idx_not_fdir) = nan;
k_out_max = k_out;
ksat(idx_not_fdir) = nan;
i_0(idx_not_fdir) = nan;
% Extra Parameters

theta_ETR = 0.5; theta_kout = 1 - theta_ETR; % Weights of distritbution of water under scarce availability
F_d_min_value = 5; % Minimum infiltrated depth in (mm)
ETR = ETP; % Assuming that ETR equals the ETP first
F_d_min = F_d_min_value*ones(length(h_ef_w),1);
F_d_min(idx_not_fdir) = nan;

%%%% ------- LOW INFLOWS AND OUTFLOWS SCENARIO --------- %%%%

if max(inflows) <= 1e-2 % mm/hr
    % No need for matrix multiplication because max inflow is very low
    inflow_rate = i_0  + d_0/(time_step/3600); % mm/h with NO ETP    
    inf_capacity = ksat.*(1 + ((psi + d_0).*dtheta)./I_0); % mm/h
    inf_rate = min(inflow_rate,inf_capacity); % mm/h (Vertical, downwards)

    % ---- Effective Depth of Surface Runoff --- %
    %     h_ef_w = round(d_0 + time_step/3600*(i_0 - ETP/24 - inf_rate),5);
    h_ef_w = round(d_0 + time_step/3600*(i_0 - ETP/24 - inf_rate),5);
    
    % Diffusive Calculations
    depths = h_ef_w;
    depths(idx_fdir ~= 1) = nan;
    wse = DEM + 1/1000*reshape(depths,[],size(DEM,2));
    % Call Flow Direction Sub
    [f_dir,~] = FlowDirection(wse,Delta_x,Delta_y,coord_outlet); % Flow direction matrix
    % Call Slope Sub
    [slope] = max_slope8D(wse,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet); % wse slope
    % Call Dir Matrix
    [Direction_Matrix] = sparse(Find_D_Matrix(f_dir,coord_outlet,Direction_Matrix_Zeros));
    % Calculate Lambda
    Lambda = (resolution)*(1./n).*slope(:).^(0.5); % Lumped hydraulic properties
    Mask = idx_fdir ~= 1;
    Lambda(Mask) = 0;

    if min(h_ef_w(idx_fdir)) < 0
        idx = h_ef_w < 0; % Cells with no water availability
        % d_0/dt - f + i = ETR/24 + k_out
        % Availability <= Demand
        %         availability = d_0/(time_step/3600) - inf_rate + i_0;
        availability = d_0/(time_step/3600) + I_0/(time_step/3600) + i_0 - inf_rate; % mm/hr
        idx2 =  availability <= 0;
        if max(max(ETP)) > 0 % We are modeling ETP
            ETR(idx2) = min((-1)*24*availability(idx2),ETP(idx2)); % mm/day (Real Evapotrasnpiration)
        else
            error('Instability. ETP issue. Probably model instability.')
        end
        h_ef_w(idx) = 0; % No water depth (mm)

    end
    % Soil Flux
    soil_flux = inf_rate - ETR/24; % mm/h oriented downwards
    % Mass Balance in the Soil Column
    F_d = I_0 + (soil_flux - k_out)*time_step/3600; % Stored depth in mm for (k+1); (assuming a min of 5 mm)

    if min(F_d) < F_d_min_value
        idx = F_d <= F_d_min_value; % Areas with Fd < Fdmin
        k_out(idx) = soil_flux(idx) + (F_d_min(idx) - I_0(idx))/(time_step/3600); % Availalbe groundwater replenish
        k_out(idx_not_fdir) = nan;
        F_d(idx) = I_0(idx) + (soil_flux(idx) - k_out(idx))*time_step/3600; % Stored depth in mm for (k+1); (assuming a min of 5 mm)        
        idx1 = k_out < 0 ; % Negative k_out means no groundwater replenishing
        if min(k_out) < 0
            ttt = 1;
        end
        soil_flux(idx1) = (F_d_min(idx1) - I_0(idx1))/(time_step/60);
        ETR(idx1) = round((-1)*(soil_flux(idx1) - inf_rate(idx1))*24,5); % Everything becomes ETP
        k_out(idx1) = 0; % No grundwater replenishing
        F_d(idx1) = I_0(idx1) + (soil_flux(idx1) - k_out(idx1))*time_step/3600; % Stored depth in mm for (k+1); (assuming a min of 5 mm)
    end

else
    %%%% ------- HIGH INFLOWS AND OUTFLOWS SCENARIO --------- %%%%

    % Diffusive Calculations
    depths = h_ef_w;
    depths(idx_fdir ~= 1) = nan;
    wse = DEM + 1/1000*reshape(depths,[],size(DEM,2));
    % Call Flow Direction Sub
    [f_dir,~] = FlowDirection(wse,Delta_x,Delta_y,coord_outlet); % Flow direction matrix
    % Call Slope Sub
    [slope] = max_slope8D(wse,Delta_x,Delta_y,coord_outlet,f_dir,slope_outlet); % wse slope
    % Call Dir Matrix
    [Direction_Matrix] = sparse(Find_D_Matrix(f_dir,coord_outlet,Direction_Matrix_Zeros));
    % Calculate Lambda
    slope(slope<0) = 0;
    Lambda = (resolution)*(1./n).*slope(:).^(0.5); % Lumped hydraulic properties
    Mask = idx_fdir ~= 1;
    Lambda(Mask) = 0;    

    % Outflows - Inflows

    delta_q = (Direction_Matrix)*(inflows).*idx_fdir; % This can be time expensive

    % No need for matrix multiplication because max inflow is very low
%     inflow_rate = i_0  + delta_q +  d_0/(time_step/3600); % mm/h with NO ETP
    inflow_rate = i_0  + delta_q +  d_0/(time_step/3600); % mm/h with NO ETP    
    inf_capacity = ksat.*(1 + ((psi + d_0).*dtheta)./I_0); % mm/h
    inf_rate = min(inflow_rate,inf_capacity); % mm/h (Vertical, downwards)

    % ---- Effective Depth of Surface Runoff --- %
    %     h_ef_w = round(d_0 + time_step/3600*(i_0 - ETP/24 - inf_rate),5);
    h_ef_w = round(d_0 + time_step/3600*(i_0 + delta_q - ETP/24 - inf_rate),5);
    if min(h_ef_w(idx_fdir)) < 0
        idx = h_ef_w < 0; % Cells with no water availability
        % d_0/dt - f + i = ETR/24 + k_out
        % Availability <= Demand
        %         availability = d_0/(time_step/3600) - inf_rate + i_0;
        availability = d_0/(time_step/3600) + I_0/(time_step/3600) + delta_q + i_0 - inf_rate; % mm/hr
        idx2 =  availability <= 0;
        if max(max(ETP)) > 0 % We are modeling ETP
            ETR(idx2) = min((-1)*24*availability(idx2),ETP(idx2)); % mm/day (Real Evapotrasnpiration)
        else
            error('Instability. ETP issue. Probably model instability.')
        end
        h_ef_w(idx) = 0; % No water depth (mm)

    end
    % Soil Flux
    soil_flux = inf_rate - ETR/24; % mm/h oriented downwards    
    % Mass Balance in the Soil Column
    F_d = I_0 + (soil_flux - k_out)*time_step/3600; % Stored depth in mm for (k+1); (assuming a min of 5 mm)

    if min(F_d) < F_d_min_value
        idx = F_d <= F_d_min_value; % Areas with Fd < Fdmin
        k_out(idx) = soil_flux(idx) + (F_d_min(idx) - I_0(idx))/(time_step/3600); % Availalbe groundwater replenish
        k_out(idx_not_fdir) = nan;
        F_d(idx) = I_0(idx) + (soil_flux(idx) - k_out(idx))*time_step/3600; % Stored depth in mm for (k+1); (assuming a min of 5 mm)        
        idx1 = k_out < 0 ; % Negative k_out means no groundwater replenishing
        if min(k_out) < 0
            ttt = 1;
        end
        soil_flux(idx1) = (F_d_min(idx1) - I_0(idx1))/(time_step/60);
        ETR(idx1) = round((-1)*(soil_flux(idx1) - inf_rate(idx1))*24,5); % Everything becomes ETP
        k_out(idx1) = 0; % No grundwater replenishing
        F_d(idx1) = I_0(idx1) + (soil_flux(idx1) - k_out(idx1))*time_step/3600; % Stored depth in mm for (k+1); (assuming a min of 5 mm) (assuming a min of 5 mm)
    end
end

% --- Non Linear Reservoir --- %
if min(F_d) < F_d_min_value
    ttt = 1;
end

[inflows,~,v] = non_lin_reservoir(Lambda,h_ef_w,h_0,Delta_x,Delta_y); % flow_t = flow(k+1) (outflow from each cell)
if isnan(max(max(inflows(idx_fdir))))
    error('nans')
end
if isinf(max(max(inflows(idx_fdir))))
    error('infs')
end
if min(min(h_ef_w(idx_fdir))) < 0
    error('Reduce the time-step.') % Depths are becoming negative - instability
end
if min(min(ETR(idx_fdir))) < 0
    error('ETR is becoming negative') % ETR are becoming negative - instability
end
if min(min(k_out(idx_fdir))) < 0
    %         error('kout is becoming negative') % Depths are becoming negative - instability
end
end