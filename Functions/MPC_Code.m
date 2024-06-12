%% MPC Code
zzz_stage_discharge = [Depth_out + DEM(row,col),Qout_w];
zzz_stage_discharge(1,3) = watershed_runningtime;

%% 11.0 Modeling Reservoir + Channel Dynamics
% Pre allocating arrays
h_r = zeros(n_steps,1); % Depth in the reservoir (m)
i_reservoir = rain_outlet(1:length(time_save),1); % Rainfall in the reservoir (mm/h)
u_begin = MPC_Control_Parameters.ur_eq_t; % Initial control law

if flags.flag_rainfall == 1
    if flags.flag_spatial_rainfall ~= 1
        i_outlet = precipitation_data; % Constant rainfall in the catchment (mm/h)
    else
        %         i_outlet = squeeze(spatial_rainfall_maps(row,col,:)); % Constant rainfall in the catchment (mm/h)
        i_outlet = squeeze(spatial_rainfall(row,col,:)); % Constant rainfall in the catchment (mm/h)
    end
else
    i_outlet = 0;
end
% We recommend saving the workspace such as
% save('workspace_waterhsed');

%% 12.0 Agregating time-step to increase speed
% Agregating the Inflow to a larger time-step
% You can either enter with your observed outflow from the watershed and
% observed rainfall or use the ones calculated previously.
% To gain velocity, we can enter these values loading these files below:
global Qout_w Qout_w_horizon MPC_Control_Parameters steps_horizon time_step n_steps Control_Vector Nvars i_reservoir h_r_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP g Reservoir_Parameters flags  l b hs roughness slope  
% Downscalling Qout to model's time-step that might be different
x = time_store_hydrographs; % low sampled data (min)
dt_model = MPC_Control_Parameters.new_timestep; % seconds %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% delete
xq = 0:dt_model/60:tfinal; % high sampled data (min)
v = Qout_w; % result from low sampled data
Qout_w = interp1(x,v,xq,'pchip'); % high sampled data interpolation (NEW)
i_reservoir = interp1(time_save,i_reservoir,xq,'pchip'); % high sampled data interpolation (NEW)
% plot(x,v,'o',xq,vq2,':.');
% shg
%% Downscalling ETP
if flags.flag_ETP == 1
    x = time_records_ETP; % low sampled data (min)
    xq = 0:dt_model/60:tfinal; % high sampled data (min)
    for i = 1:length(ETP_saving)
        zzz = ETP_save(:,:,i); zzz = zzz(:); % ETP at the outlet in mm/day
        ETP_saving(i,1) = zzz(pos);
    end
    v = ETP_saving; % result from low sampled data
    if length(time_store_ETP) ~= 1
        E = max(interp1(x,v,xq,'pchip'),0)'; % high sampled data interpolation
    else
        E = ETP_saving(end)';
    end
else
    E = zeros(1,n_steps);
end
% plot(x,v,'o',xq,E,':.');
% shg

%%%% Net Rainfall i'(k)
i_reservoir = i_reservoir - E'/24; % Rainfall intensity - Evaporation Intensity (mm/h)
% All of this previously done is to to avoid to run the watershed model, but
% you can run, of course, or you can also input it from HEC-HMS or other
% model

%%%% Agregating Time-Steps to the new_timestep for the reservoir model %%%%
flow_timestep = dt_model;
inflow_timesteps = n_steps; % Number of time-steps using the watershed time-step
n_steps = (n_steps-1)*flow_timestep/MPC_Control_Parameters.new_timestep; % adjusting the new number of time-steps
Qout_w_disagregated = Qout_w';
agregation_step = MPC_Control_Parameters.new_timestep/flow_timestep; % number of time-steps in on agregation time-step

% Preallocating Arrays
n_steps_disagregation = ((inflow_timesteps-1)/agregation_step);
flow_agregated = zeros(n_steps_disagregation,1);
i_reservoir_agregated = zeros(n_steps_disagregation,1);

% Disagregating or Agreagating flows and rainfall
for i = 1:n_steps_disagregation
    if MPC_Control_Parameters.new_timestep > flow_timestep
        flow_agregated(i,1) =  mean(Qout_w_disagregated(1 + (i-1)*agregation_step:i*agregation_step,1));
        i_reservoir_agregated(i,1) = mean(i_reservoir(1 + (i-1)*agregation_step:i*agregation_step,1));
    else
        flow_agregated(i,1) =  Qout_w_disagregated(1 + floor((i-1)*agregation_step):ceil(i*agregation_step));
        i_reservoir_agregated(i,1) = i_reservoir(1 + floor((i-1)*agregation_step):ceil(i*agregation_step));
    end
end

% Defining updated outflows from the watershed, rainfall intensity and
% time_step
Qout_w = flow_agregated;
i_reservoir = i_reservoir_agregated;
time_step = MPC_Control_Parameters.new_timestep;


%% 13.0 Calling MPC control
% Let's clear variables we don't use to avoid computational burden
% We recommend loading the workspace containing all data up to line 561

clearvars -except label_watershed_post_processing flags h_r Adaptive_Time_Stepping Channel_Parameters Qout_w Qout_w_horizon MPC_Control_Parameters.Control_Horizon steps_horizon time_step n_steps Control_Vector Nvars i_reservoir Channel_Parameters.h_c_0 h_r_t MPC_Control_Parameters.ur_eq_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP MPC_Control_Parameters.new_timestep MPC_Control_Parameters.Control_Interval MPC_Control_Parameters.Control_Horizon MPC_Control_Parameters.Prediction_Horizon g Reservoir_Parameters.Cd flags.flag_r Reservoir_Parameters.number_of_orifices  Reservoir_Parameters.D l b Reservoir_Parameters.relative_hmin Reservoir_Parameters.Cds Reservoir_Parameters.Lef hs Reservoir_Parameters.porosity Reservoir_Parameters.orifice_height Channel_Parameters.x_length Channel_Parameters.y_length roughness Channel_Parameters.segments slope Channel_Parameters.slope_outlet_channel MPC_Control_Parameters.max_iterations MPC_Control_Parameters.max_fun_eval MPC_Control_Parameters.n_randoms flags.flag_c flags.flag_r hmin s Channel_Parameters.h_end Channel_Parameters.slope_outlet_channel Channel_Parameters.n_channel MPC_Control_Parameters.a_noise MPC_Control_Parameters.b_noise m L Human_Instability_Parameters.Cd_HI u B Human_Instability_Parameters.pperson Human_Instability_Parameters.pfluid Human_Instability_Parameters.h_max_channel Human_Instability_Parameters.L_channel stage_area flags.flag_channel_modeling flags.flag_human_instability MPC_Control_Parameters.detention_time_max MPC_Control_Parameters.q_max_star MPC_Control_Parameters.q_max_star_star MPC_Control_Parameters.alpha_p MPC_Control_Parameters.rho_u MPC_Control_Parameters.rho_h MPC_Control_Parameters.rho_hmax MPC_Control_Parameters.rho_c MPC_Control_Parameters.rho_hI MPC_Control_Parameters.max_res_level flags.flag_optimization_solver u_static MPC_Control_Parameters.ur_eq
save('workspace_prior_MPC')
tic % Startg Counting MPC time

%%
load('workspace_prior_MPC')
% Disable Warnings
warning('off','all')
warning('query','all')
tic
% ------ Manual Inputs ----

% Delete Later
global Qout_w Area_Functions Volume_Function h_stage flags alfa_1_function alfa_2_function beta_1_function beta_2_function gamma_1_function gamma_2_function Reservoir_Parameters Channel_Parameters Qout_w_horizon MPC_Control_Parameters steps_horizon time_step n_steps Control_Vector Nvars i_reservoir Channel_Parameters h_r_t i_reservoir_horizon previous_control_valve average variance slope_outlet tfinal record_time_maps ETP g flag_r Reservoir_Parameters l b  hs roughness slope s m L u stage_area number_of_controls previous_control_gate previous_control_valve uv_eq us_eq uv_eq_t us_eq_t u_s u_v
MPC_Control_Parameters.max_fun_eval = 120;
% MPC_Control_Parameters.q_max_star = 10;
% MPC_Control_Parameters.q_max_star_star = 40;
% MPC_Control_Parameters.detention_time_max = 18;
Volume_Detention = 0;
Outflow_Volume = 0;

% Weights
% MPC_Control_Parameters.rho_u = 1;
% MPC_Control_Parameters.rho_hmax = 10^6;

% MPC_Control_Parameters.Control_Interval = 60.00;
MPC_Control_Parameters.Control_Horizon = 120.00;
% MPC_Control_Parameters.Prediction_Horizon = 720.00;
MPC_Control_Parameters.max_iterations = 120.00;
% MPC_Control_Parameters.n_randoms = 5;
% MPC_Control_Parameters.max_res_level = 6.8;
MPC_Control_Parameters.q_max_release = 2; % m3/s


flags.flag_gatecontrol = 1;
% Reservoir_Parameters.ks_gatecontrol = 27;
% Reservoir_Parameters.alpha_s_gatecontrol = 1.5;
MPC_Control_Parameters.Threshold_Detention = 1; % Minimum flow in m3/s to start an optimization

% Deleteeeee
% flags.flag_gatecontrol = 1;

u_static = 1; % DELETEE
flags.flag_optimization_solver = 1; % Static
flags.flag_gatecontrol = 1;
% flags.flag_optimization_solver = 1; % Static
% flags.flag_gatecontrol = 1;


Reservoir_Parameters.ks_gatecontrol = 27;
Reservoir_Parameters.alpha_s_gatecontrol = 1.5;

if flags.flag_gatecontrol == 1
    number_of_controls = 2; % Valves and Gates
else
    number_of_controls = 1; % Only the valves
end

% Water Quality Parameters
detention_time = 0; % Beginning it
detention_step = 1;
% Patter Search Number of Initial Searches
number_of_initials = 5; % Number of initials for pattern search

% a) A few vector calculations
n_controls = MPC_Control_Parameters.Control_Horizon/MPC_Control_Parameters.Control_Interval; % number of controls to choose from an optimization problem
MPC_Control_Parameters.Control_Horizon_steps = MPC_Control_Parameters.Control_Horizon*60/time_step; % number of steps in the control horizon;
steps_horizon = MPC_Control_Parameters.Prediction_Horizon*60/time_step; % number of steps in one prediction horizon
Qout_w(length(Qout_w):(length(Qout_w)+steps_horizon),1) = 0; % adding more variables to the inflow to account for the last prediction horizon
i_reservoir(length(i_reservoir):(length(i_reservoir)+steps_horizon),1) = 0; % adding more variables to the inflow to account for the last prediction horizon
n_horizons = ceil((n_steps)/(MPC_Control_Parameters.Control_Horizon_steps)); % number of control horizons
Control_Vector = [0:MPC_Control_Parameters.Control_Interval:MPC_Control_Parameters.Prediction_Horizon]*60/time_step;
Nvars = length(Control_Vector)*number_of_controls;
Vars_per_control = Nvars/number_of_controls; % Number of variables per controllable asset
U_random = zeros(Nvars,MPC_Control_Parameters.n_randoms);

dim = (1:MPC_Control_Parameters.Control_Horizon_steps);
detention_interval = (1:length(dim))'*time_step/3600; % hour;

% b) Objective Function
fun = @Optimization_Function;
OF_value = zeros(n_horizons,1);

% c) Optmizer - Solver
%%%% Interior Point %%%
options = optimoptions(@fmincon,'MaxIterations',MPC_Control_Parameters.max_iterations,'MaxFunctionEvaluations',MPC_Control_Parameters.max_fun_eval); % Previously
%%%% To increase the chances of finding global solutions, we solve the problem
% for MPC_Control_Parameters.n_randoms initial points %%%%

%%%% Estimate random initial points for the Random Search %%%%
matrix = rand(Nvars,MPC_Control_Parameters.n_randoms); % Only for interior points method with fmincon

% Matrices for FMINCON Optimization Problem
A = []; B = []; Aeq = [];Beq = []; Lb = zeros(Nvars,1); Ub = ones(Nvars,1);

%%% Prealocatting Arrays %%%
h_r_final = zeros(n_steps,1);

% Channel modeling
h_c_max_final = zeros(n_steps,1);
U = zeros(Nvars,1); % Concatenation of future control signals
if flags.flag_gatecontrol ~=1
    u = MPC_Control_Parameters.ur_eq_t; % initial control (MAYBE DELETE)
    uv_eq_t = 0;
    us_eq_t = 0;
else
    uv_eq_t = 0;
    us_eq_t = 0;
end

u_v = 0; % Valves Closed
u_s = 0; % Gates Closed
previous_control_valve = uv_eq_t;
previous_control_gate = us_eq_t;

% Orifice Properties
Aoc = pi()*Reservoir_Parameters.D^2/4*Reservoir_Parameters.number_of_orifices ;
Aor = Reservoir_Parameters.l*Reservoir_Parameters.b*Reservoir_Parameters.number_of_orifices ;
if ((flags.flag_c == 1) && (flags.flag_r == 1))
    error('Please choose only one type of orifice')
elseif (flags.flag_c == 1)
    D_h = D; % circular
    Ao = Aoc;
else
    D_h = 4*(Reservoir_Parameters.l*Reservoir_Parameters.b)/(2*(Reservoir_Parameters.l + Reservoir_Parameters.b )); % rectangular
    Ao = Aor;
end
Ko = Reservoir_Parameters.Cd*Ao*sqrt(2*g);
check_detention = 0;

%% Symbolic Alfa and Beta Matrices
if flags.flag_gatecontrol ~= 1
    [alfa_1_function,alfa_2_function,beta_1_function,beta_2_function] = symbolic_jacobians(stage_area,flag_gatecontrol); % 
else
    [alfa_1_function,alfa_2_function,beta_1_function,beta_2_function,gamma_1_function,gamma_2_function] = symbolic_jacobians(stage_area,flags.flag_gatecontrol); % 
end

%% Reservoir Stage-Varying Functions
[Area_Functions,Volume_Function,h_stage] = reservoir_stage_varying_functions(stage_area);
%% Main Routing - For all control horizons
for i = 1:n_horizons
    perc = i/n_horizons*100;
    perc____timeremainhour = [perc, (toc/60/60)/(perc/100),max(Qout_w_horizon)]

    % Define inflow from the catchment during the prediction horizon
    time_begin = (i-1)*MPC_Control_Parameters.Control_Horizon_steps + 1; % step
    t_min = time_begin*time_step/60; %  time in min
    time_end = time_begin + steps_horizon;
    t_end = time_end*time_step/60; % final time of the current step
    time_end_saving = time_begin + MPC_Control_Parameters.Control_Horizon_steps-1; % for saving
    Qout_w_horizon = Qout_w(time_begin:time_end,1); % Result of the Watershed Model
    i_reservoir_horizon = i_reservoir(time_begin:time_end,1); % Result of the Rainfall Forecasting

    % Determining random initial points
    %%% Noise Generation - Prediction %%%
    % In case noise is generated in the states, you can model it by a
    % assuming a Gaussian noise with an average and variance specified
    % below
    average = 0.0; % average noise in (m)
    variance = 0; % variance in (m2). Remember that xbar - 2std has a prob of 93... %, std = sqrt(variance)


    %%% - FMINCON WITH RANDOM NUMBERS AS X0
    if flags.flag_optimization_solver == 1
        for j=1:MPC_Control_Parameters.n_randoms
            % Orifice
            if u_v(end,1) == 0 % In case the previous control equals zero
                u0_v = matrix(1:(Vars_per_control),j); % Random numbers for orifice
                U_v = U(1:Nvars/2);
                U_s = U((Nvars/2 + 1):end);
                u0_v(1) = U(n_controls);
            else
                U_v = U(1:Nvars/2);
                U_s = U((Nvars/2 + 1):end);
                u0_v = max(U_v(n_controls)*(MPC_Control_Parameters.a_noise + matrix(1:(Vars_per_control),j)*(MPC_Control_Parameters.b_noise - MPC_Control_Parameters.a_noise)),0); % randomly select values within the adopted range
                u0_v = max(u0_v,0);
                u0_v(1) = U_v(n_controls); % the initial estimative is the previous adopted
            end
            % Gates
            if u_s(end,1) == 0 % In case the previous control equals zero
                u0_s = matrix(1:(Vars_per_control),j); % Random numbers for orifice
                u0_s(1) = U_s(n_controls);
            else
                u0_s = max(U_s(n_controls)*(MPC_Control_Parameters.a_noise + matrix((Vars_per_control+1):Nvars,j)*(MPC_Control_Parameters.b_noise - MPC_Control_Parameters.a_noise)),0); % randomly select values within the adopted range
                u0_s = max(u0_s,0);
                u0_s(1) = U_s(n_controls); % the initial estimative is the previous adopted
            end

            if max(Qout_w_horizon) > MPC_Control_Parameters.Threshold_Detention % If the maximum predicted outflow is greater than a value in m3/s
                detention_time = 0;
                
                
                %%% Solving the Optimization Problem for the Random Initial
                %%% Points

                %%%% ------------ Fmincon Solution Use Below ------------------- %%%
                if flags.flag_gatecontrol == 1
                    u0 =[u0_v; u0_s]; % Concatenating Decisions
                end
                [U,FVAL] = fmincon(fun,u0',A,B,Aeq,Beq,Lb,Ub,[],options);  % Fmincon solutions
                % ------------------------------------------------------------------------------
                OF(j) = FVAL; % saving the objective function value
                U_random(:,j) = U'; % saving the control vector
                position = find(OF == min(OF)); % position where the minimum value of OF occurs
                if length(position)>1
                    position = position(1);
                end
                % Saving OF value and Controls
                if flags.flag_gatecontrol ~= 1
                    U = U_random(:,position); % Chosen control
                else
                    U = U_random(:,position); % Chosen control
                    U_v = U(1:n_controls);
                    U_s = U((Nvars/2 + 1):(Nvars/2 + n_controls));
                end
            end
        end

        % ---- Detention Time ---- %
        if max(Qout_w_horizon) <= MPC_Control_Parameters.Threshold_Detention && max(h_r(1:MPC_Control_Parameters.Control_Horizon_steps,1)) > Reservoir_Parameters.hmin 
            % If the maximum predicted outflow is smaller than a threshold 
            % We have water in the reservoir for the control step
            detention_time = detention_time + MPC_Control_Parameters.Control_Horizon/60; % hours
            if check_detention == 0 && h_r(end,1) > 0  % Saving the last depth
                h_r_end = h_r(end,1);
            elseif check_detention == 0 && h_r(end,1) == 0
                h_r_end = 1e-3; % 1 cm
            end
            % Detention Time Release
            if detention_time > MPC_Control_Parameters.detention_time_max
                check_detention = 1;
                % We start releasing water
                u_qmax_star = min(MPC_Control_Parameters.q_max_release/(Ko*sqrt(h_r_end)),1);
                U(1:Nvars/2,1) = u_qmax_star*(ones(Nvars/2,1)); % Open Partially
                U((Nvars/2+1):Nvars,1) = 0; % Close Gates
            else
                U = 0*(ones(Nvars,1)); % Close Valves and hold water
                U_v = U(1:n_controls);
                U_s = U((Nvars/2 + 1):(Nvars/2 + n_controls));
            end
        else
            detention_time = 0;
            check_detention = 0;
        end

    elseif flags.flag_optimization_solver == 0 % Pattern Search Approach
        for j=1:number_of_initials
            %%%%% - Search Algorithms - %%%%
            u0 = zeros(Nvars,1) + (j-1)*(1/(number_of_initials-1));
            if max(Qout_w_horizon) <= MPC_Control_Parameters.Threshold_Detention % If the maximum predicted outflow is zero
                detention_time = detention_time + MPC_Control_Parameters.Control_Horizon/60; % hours
                if check_detention == 0 && h_r(end,1) > 0  % Saving the last depth
                    h_r_end = h_r(end,1);
                elseif check_detention == 0 && h_r(end,1) == 0
                    h_r_end = 1e-3; % 1 cm
                end
                if detention_time > MPC_Control_Parameters.detention_time_max
                    check_detention = 1;
                    % We start releasing water
                    u_qmax_star = min(MPC_Control_Parameters.q_max_release/(Ko*sqrt(h_r_end)),1);
                    U = u_qmax_star*(ones(Nvars,1)); % Open 50% only
                else
                    U = 0*(ones(Nvars,1)); % Close Valves and hold water
                end
            else
                %%%% ------------ Global Search Solution, if you want to use ------------- %%%%
                %             rng default % For reproducibility
                %             ms = MultiStart('FunctionTolerance',2e-4,'UseParallel',true);
                %             gs = GlobalSearch(ms);
                %             gs.MaxTime = 60; % Max time in seconds
                %             gs.NumTrialPoints = 20;
                %             problem = createOptimProblem('fmincon','x0',u0','objective',fun,'lb',Lb,'ub',Ub);
                %             [U,FVAL] = run(gs,problem); % Running global search
                %%%% --------------- Pattern Search Solution ---------------- %%%%
                [U,FVAL] = patternsearch(fun,u0',A,B,Aeq,Beq,Lb,Ub,[],options); % Pattern search solutions
                OF(j) = FVAL;
                U_random(:,j) = U'; % saving the control vector
                position = find(OF == min(OF)); % position where the minimum value of OF occurs
                if length(position)>1 % More than 1 solution with same O.F
                    position = position(1);
                end
                % Saving OF value
                U = U_random(:,position); % Chosen control
                % ------------------------------------------------------------------------------
            end
        end
    else % Run With Known Valve Control - Static Operation
        U = u_static*(ones(Nvars,1)); % Here you can either choose to open the valves or close them, in case you want to increase detention time
        U_v = U(1:Nvars);
        U_s = U(1:Nvars);
        FVAL = Optimization_Function(U);
        if max(h_r(1:MPC_Control_Parameters.Control_Horizon_steps,1)) > Reservoir_Parameters.hmin % We have water in the reservoir
            detention_time = detention_time + MPC_Control_Parameters.Control_Horizon/60; % hours
            check_detention = 1; % We are releasing water
        else
            detention_time = 0;
            check_detention = 0;
        end
    end
    % Objective Function
    if detention_time > MPC_Control_Parameters.detention_time_max && max(Qout_w_horizon) <= MPC_Control_Parameters.Threshold_Detention % Releasing
        OF_value(i) = nan;
    elseif detention_time <= MPC_Control_Parameters.detention_time_max && max(Qout_w_horizon) <= MPC_Control_Parameters.Threshold_Detention % Holding
        OF_value(i) = nan;
    else % Optimizing
        if flags.flag_optimization_solver ~= 1 && flags.flag_optimization_solver ~= 0
            OF_value(i) = FVAL; % Value for known u(k) over time
        else
            OF_value(i) = OF(position); % We are solving the O.P problem
        end
    end

    if flags.flag_gatecontrol ~= 1
        controls((i-1)*n_controls + 1:i*n_controls,1) = U(1:n_controls)';
    else
        controls((i-1)*n_controls + 1:i*n_controls,1) = U_v(1:n_controls)';
        controls((i-1)*n_controls + 1:i*n_controls,2) = U_s(1:n_controls)';
    end
    % Implement the Controls in the Plant and retrieving outputs
    %%% Run the Model with the estimated trajectory determined previously %%%
    % Disagregating u into the time-steps of the model
    for j=1:(steps_horizon)
        idx = find(Control_Vector < j,1,'last'); % U disagregated into time-steps
        if flags.flag_gatecontrol == 1
            U_v_total = U(1:(Nvars/2));
            U_s_total = U((Nvars/2 +1):end);
        else
            U_v_total = U(1:(Nvars));
            U_s_total = U(1:(Nvars));
        end
        u_v(j,1) = U_v_total(idx);
        u_s(j,1) = U_s_total(idx);
    end
    previous_control_valve = U_v_total(1);
    previous_control_gate = U_s_total(1);
    %%% Noise Generation - Application %%%
    % In case noise is generated in the states, you can model it by a
    % assuming a Gaussian noise with an average and variance specified
    % below (This applies only to the plant)

    %%% Reservoir Plant %%%
    [h_r,out_r] = reservoir_dynamics(Qout_w_horizon,time_step,u,g,Reservoir_Parameters.Cd,Reservoir_Parameters.number_of_orifices ,flags.flag_c,Reservoir_Parameters.D,flags.flag_r,Reservoir_Parameters.l,Reservoir_Parameters.b,Reservoir_Parameters.hmin,Reservoir_Parameters.orifice_height,Reservoir_Parameters.Cds,Reservoir_Parameters.Lef,Reservoir_Parameters.hs,Reservoir_Parameters.porosity,average,variance,stage_area,flags.flag_gatecontrol,u_v,u_s); % Reservoir Dynamics
    h_r_t = h_r(MPC_Control_Parameters.Control_Horizon_steps);
    if flags.flag_gatecontrol ~=1
        MPC_Control_Parameters.ur_eq_t = U(n_controls); % Initial values for the next step
    else
        uv_eq_t = U_v_total(1);
        us_eq_t = U_s_total(1);
    end

    %%% Calculating Treated Volume
    if check_detention == 1
        detention_step = detention_step + 1;
        detention_interval = detention_interval + MPC_Control_Parameters.Control_Horizon/60; % hours
        Volume_Detention(detention_step,1) = Volume_Detention(detention_step-1) + sum(out_r(dim)*time_step.*detention_interval); % m3 * h
        Outflow_Volume(detention_step,1) = Outflow_Volume(detention_step-1,1) + sum(out_r(dim)*time_step); % m3
        average_detention_time(detention_step,1) = Volume_Detention(detention_step,1)/Outflow_Volume(detention_step,1); % hour
        if isnan(average_detention_time(detention_step,1))
            average_detention_time(detention_step,1) = 0;
        end
        time_detention(detention_step,1) = i*MPC_Control_Parameters.Control_Horizon/60/24; % Days
    else
        detention_interval = (1:length(dim))'*time_step/3600; % hour;
    end

    %%% Channel Plant %%%
    if flags.flag_channel_modeling == 1
        [max_water_level,h_c,out_c] = plant_channel(out_r,time_step,Channel_Parameters.h_c_0,Channel_Parameters.x_length,Channel_Parameters.y_length,roughness,average,variance,Channel_Parameters.segments,s,Channel_Parameters.slope_outlet_channel,Channel_Parameters.h_end); % Channel Dynamics
        Channel_Parameters.h_c_0 = h_c(:,MPC_Control_Parameters.Control_Horizon_steps); % Initial Values for the next step
        % Saving Results
        h_c_max_final((time_begin:time_end_saving),1) = max(h_c(:,1:MPC_Control_Parameters.Control_Horizon_steps))';
        out_c_final(time_begin:time_end_saving,1) = out_c(1:MPC_Control_Parameters.Control_Horizon_steps,1);
    end

    %%% Saving Results %%%
    h_r_final(time_begin:time_end_saving,1) = h_r(1:MPC_Control_Parameters.Control_Horizon_steps,1);
    out_r_final(time_begin:time_end_saving,1) = out_r(1:MPC_Control_Parameters.Control_Horizon_steps,1);
    %     h_c_final(:,(time_begin:time_end_saving)) = h_c(:,1:MPC_Control_Parameters.Control_Horizon_steps);
end

% Enable Warnigns
warning('on','all')
warning('query','all')

% %% 12.0 Disagregating controls to the time-step unit
Control_Vector = [0:MPC_Control_Parameters.Control_Interval*60/time_step:n_steps];
for i=1:(n_steps-1)
    idx = find(Control_Vector <= i,1,'last'); % Position in U
    if flags.flag_gatecontrol ~=1
        u(i,1) = controls(idx);
    else
        u_v(i,1) = controls(idx,1);
        u_s(i,1) = controls(idx,2);
    end
end


if flags.flag_gatecontrol ~=1
    u_begin = MPC_Control_Parameters.ur_eq;
    u = [u_begin; u];
else
    u_begin = MPC_Control_Parameters.ur_eq;
    u_v = [uv_eq; u_v];
    u_s = [us_eq; u_s];
end
% graphs_wshed_reservoir_channel
time_MPC = toc;

save('workspace_after_MPC');

