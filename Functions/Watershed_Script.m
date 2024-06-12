%% Watershed Model
% Developer: Marcus Nobrega Gomes Junior
% 12/19/2023
% Main Script
% Goal: Create the main file of the watershed modeling
% a) Watershed Model - Loading results and 
% b) Saving final results

% Loading Previous Results
load(label_watershed_prior_modeling)

% Calculating some modeling parameters
dt = time_step/60;
time = 0;
time_end = n_steps*time_step/60; % Minutes
z_ETP_prev = 0;
t_store_hydrographs_prev = 0;
t_store_prev = 0;
tic
% Converting time_stores to time
time_store_rainfall_time = Rainfall_Properties.time_store_rainfall*time_step/60; % min
time_store_ETP_time = time_store_ETP*time_step/60; % min
time_store_time = time_store*time_step/60; % min
time_store_hydrographs = time_store_hydrographs*time_step/60; % min

inf_prev = 0;
flood_prev = 0;
time_save = zeros(n_steps,1); % Vector saving time

%% Main While for Watershed Routing
while time <= (time_end - 1)
    k = k + 1; % Time-step index
    time = time + time_step/60; % Time in minutes (duration)
    time_save(k) = time; % Minutes
    % Rainfall Input
    z = find(time_store_rainfall_time <= time, 1,'last' ); % Position of rainfall
    if flags.flag_spatial_rainfall ~=1
        if z > length(precipitation_data)
            z = length(precipitation_data);
            factor_rainfall = 0;
            warning('Not enough rainfall data. We are assuming that the rainfall is null.');
        else
            factor_rainfall = 1;
        end
    end
    if flags.flag_rainfall == 1 && flags.flag_spatial_rainfall ~=1
        i_0 = factor_rainfall*precipitation_data(z,1).*idx_cells; % Initial Rainfall for the next time-step
    elseif flags.flag_rainfall == 1 && flags.flag_spatial_rainfall == 1 && z > z_prev
        if time > Rainfall_Properties.end_rain - time_step/60
            warning('We are assuming the rainfall is null because the data is missing.')
            rainfall = zeros(n_raingauges,1); % Values of rainfall at t for each rain gauge
        else
            rainfall = Rainfall_Properties.rainfall_raingauges(z,1:Rainfall_Properties.n_raingauges)'; % Values of rainfall at t for each rain gauge
        end
        [spatial_rainfall] = Rainfall_Interpolator(x_coordinate,y_coordinate,rainfall,x_grid,y_grid); % Interpolated Values
        i_0 = spatial_rainfall(:).*idx_cells;  % mm/h
        spatial_rainfall_maps(:,:,z) = spatial_rainfall;  % Saving map
    elseif flags.flag_rainfall == 0
        i_0 = 0.*idx_cells;
    end
    z_prev = z; % Previous value of z
    rain_outlet(k,1) = i_0(pos,1); % Outlet Rainfall

    if flags.flag_ETP == 1
        z_ETP = find(time_store_ETP_time <= time, 1,'last' ); % Position of rainfall
        % Refreshing ETP Data
        if k == 1
            % Do nothing
        elseif z_ETP > z_ETP_prev % Typically - Daily
            ETP = ETP_save(:,:,z_ETP);
            ETP = ETP(:);
            ETR_save(:,:,z_ETP) = reshape(ETR,[],size(DEM,2));
        end
        z_ETP_prev = z_ETP;
    end

    % Call the Watershed Matrix Model
    % idx_cells == idx_fdir
    if flags.flag_diffusive ~=1 
        [Soil_Properties.F_d,h_ef_w,inflows,ETR,Soil_Properties.k_out,v] = wshed_matrix(h_ef_w,LULC_Properties.h_0,inflows,time_step,Direction_Matrix,i_0,Soil_Properties.ksat,Soil_Properties.psi,Soil_Properties.dtheta,Soil_Properties.F_d,Lambda,Delta_x,Delta_y,ETP,idx_cells,Soil_Properties.k_out_max,pos);
    else
        [Soil_Properties.F_d,h_ef_w,inflows,ETR,Soil_Properties.k_out,v] = wshed_matrix_diffusive(h_ef_w,h_0,inflows,time_step,i_0,ksat,psi,dtheta,F_d,Lambda,Delta_x,Delta_y,ETP,idx_cells,k_out,coord_outlet,DEM,n,slope_outlet,Direction_Matrix_Zeros,resolution);
    end
    max_depth = max(max_depth,h_ef_w);
    max_velocity = max(v); % Maximum velocity in m/s
    max_velocity(max_velocity == 0) = 1e-6;

    % Adaptive Time-Step
    dt_calc = Adaptive_Time_Stepping.alpha*(resolution)/(max_velocity); % Time in seconds
    dt = min(dt_calc,Adaptive_Time_Stepping.t_max);
    dt = max(Adaptive_Time_Stepping.t_min,dt);
    time_step = dt; % Seconds

    % Saving variables with user defined recording time-step
    if k == 1
        % Do nothing, it is already solved, we just have to save the data
        % for the next time-step
    else
        t_store = find(time_store_time <= time,1,'last');
        if t_store > t_store_prev
            d(:,t_store) = h_ef_w(:); % Depths in mm
            I(:,t_store) = (Soil_Properties.F_d); % stored depth in mm
            t_store_prev = t_store;
        end
    end

    % Saving hydrographs and depths with user defined recording time-step
    if k == 1
        % Do nothing, it is already solved, we just have to save the data
        % for the next time-step
        t_store_hydrographs = 1;
    else
        t_store_hydrographs = find(time_store_hydrographs <= time,1,'last'); % Time that is being recorded in min
        if t_store_hydrographs > t_store_hydrographs_prev
            if flags.flag_reservoir_wshed == 1
                Qout_w(t_store_hydrographs,1) = max(sum(inflows(pos)*(Delta_x*Delta_y)/(1000*3600)) - Area*i_0(pos,1)/1000/3600,0); % Outflow (cms)
                Depth_out(t_store_hydrographs,1) = 1/1000*h_ef_w(pos); % m
            else
                Qout_w(t_store_hydrographs,1) = sum(inflows(pos)*(Delta_x*Delta_y)/(1000*3600)); % Outflow - Area * i (cms)
                Depth_out(t_store_hydrographs,1) = 1/1000*h_ef_w(pos); % m
                %Qout_w(k,1) = sum(inflows(pos)*(Delta_x*Delta_y)/(1000*3600)); % Outflow - Area * i (cms)
            end
            % Saving ETP with user defined recording time-step
            if k == 1
                % ETP
                t_store_ETP = 1;
                ETP_saving(t_store_ETP,1) = ETP(pos);
            elseif find(time_store_ETP == k) > 0
                t_store_ETP = find(time_store_ETP == k);
                ETP_saving(t_store_ETP,1) = ETP(pos); % ETP value in mm/day but at rain and flow time-step for the outlet
            end

            % Average Infiltration
            I_actual = Soil_Properties.F_d; % mm in each cell
            Vol_begin = sum(I_previous.*idx_cells)/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            Vol_end = sum(I_actual.*idx_cells)/(1000*1000*drainage_area/(resolution^2)); % mm per cell in average
            f_rate(1,t_store_hydrographs) = (Vol_end - Vol_begin)/(record_time_hydrographs/60); % mm/hr; % Soil flux
            I_previous = I_actual;

            t_store_hydrographs_prev = t_store_hydrographs;
        end
    end

    perc = time/(time_end)*100;

    if flags.flag_ETP == 1
        perc________dt_______qoutw______depthw___infexut___timeremain____ETP____errorm3s = [perc,time_step,Qout_w(t_store_hydrographs,1),Depth_out(t_store_hydrographs,1),I(pos,t_store),toc*100/perc/3600,ETP(pos),error] % Showing the depths at the outlet and the percentage of the calculations a
    else
        perc_______dt_______qoutw______depthw___infexut___timeremain____errorm3s = [perc,time_step,max(sum(inflows(pos)*(Delta_x*Delta_y)/(1000*3600))),1/1000*h_ef_w(pos),I(pos,t_store),toc*100/perc/3600,error,1/1000*max(h_ef_w)] % Showing the depths at the outlet and the percentage of the calculations a
    end
    % Check if a Steady Dry Condition is Reached
    %     if max(h_ef_w) == 0 && max(F_d) == 5 && flags.flag_rainfall == 1 && flags.flag_spatial_rainfall ~= 1
    %         z = find(time_store_rainfall <= k, 1,'last' ); % Position of rainfall
    %         zz = find(precipitation_data(z:end,1) > 0,1,'first') + z - 1;
    %         k = zz*step_rainfall*60/time_step - 2;
    %         t_new = find(time_store_ETP <=k,1,'last'); % Time that is being recorded in min
    %         I_previous = F_d;
    %         ETP = ETP_save(:,:,t_new); ETP = ETP(:);
    %         ETP_saving(t_new+1,1) = ETP(pos); % ETP value in mm/day
    %         ETR_saving(t_new+1,1) = ETR(pos); % ETP value in mm/day
    %     end

    %%% ------------- Mass Balance Routine  ------------------- %%%
    if flags.flag_spatial_rainfall == 1
        rain_vol = sum((Resolution^2*1/1000*spatial_rainfall(:).*idx_cells*(time_step/3600))); % m3
        rain = 1000*(rain_vol)/(drainage_area*1000*1000)/(time_step/3600); % mm/h
        ETR_vol = nansum((Resolution^2*1/1000*ETR/24).*idx_cells*(time_step/3600)); % m3
        etr_rate = 1000*(ETR_vol)/(drainage_area*1000*1000)/(time_step/3600); % mm/h
    else
        rain = i_0(pos); % mm/h
        ETR_vol = nansum((Resolution^2*1/1000*ETR/24).*idx_cells*(time_step/3600)); % m3
        etr_rate = 1000*(ETR_vol)/(drainage_area*1000*1000)/(time_step/3600); % mm/h
    end
    exf = nansum(Soil_Properties.k_out)/(sum(double(idx_cells))); % average k_out for all catchment (mm/h)
    Q = sum(inflows(pos)*(Delta_x*Delta_y)/(1000*3600)); % m3/s of outflow
    flooded_volume = nansum(Resolution^2*h_ef_w/1000.*idx_cells);
    S_t_1 = nansum(Resolution^2.*Soil_Properties.F_d/1000.*idx_cells) + flooded_volume; % m3
    %     delta_inf = nansum(Resolution^2.*F_d/1000.*idx_cells) - inf_prev
    %     delta_flood = nansum(Resolution^2*h_ef_w/1000.*idx_cells) - flood_prev
    %     inf_prev = nansum(Resolution^2.*F_d/1000.*idx_cells);
    %     flood_prev = nansum(Resolution^2*h_ef_w/1000.*idx_cells);
    % Mathematical Background for Error Evaluation
    % dS/dt = Inflows - Outflows
    % S(k+1) - S(k) = dt * (Inflows - Outflows)
    % Storage = Drainage_Area * (Surface Depth + Infiltrated Depth)
    % Inflows = Drainage_Area * (Rainfall)
    % Outlofws = Drainage_Area * (Exfiltration + ETR + Q)
    % Delta_Storage = S(k+1) - S(k)
    Delta_Storage = S_t_1 - S_t;
    Inflows_Massbalance = drainage_area*1000*1000*rain/1000/3600*time_step; % m3
    Outflows_Massbalance = Q*time_step + drainage_area*1000*1000*(exf + etr_rate)/1000/3600*time_step; % m3
    Delta_Flows = Inflows_Massbalance - Outflows_Massbalance; % m3
    error = Delta_Storage - Delta_Flows; % m3
    if error < 0 
        ttt = 1;
    end
    error_model(k) = error/(time_step); % m3/s
    %     t = t + time_step/60; % Minutes
    S_t = S_t_1;
    Soil_Properties.k_out = Soil_Properties.k_out_max;
end
watershed_runningtime = toc/60; % Minutes
time_save = time_save(1:k,1); % Only in values with k
