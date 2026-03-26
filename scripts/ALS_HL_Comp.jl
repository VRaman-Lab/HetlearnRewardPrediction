 include("setup.jl")

function steady_state_KG(stoch, vol)
    Pinf = 0.5*(vol + sqrt(vol^2 + 4*vol*stoch))
    Kinf = (Pinf) / (Pinf + stoch)
    return Kinf
end

"""
Works for the fixed stochasticity and volatility of the ALS figure
"""
function opt_kg_trajectory(c0, stoch, vol, T)
    covs = zeros(T)
    kgs = zeros(T)

    Ppred = c0
    for i in 1:T
        # Kalman gain based on prior
        K = Ppred / (Ppred + stoch)
        kgs[i] = K

        # Update posterior
        Ppost = (1 - K) * Ppred
        covs[i] = Ppost

        # Predict next step
        Ppred = Ppost + vol
    end
    return kgs
end

function run_ALS_comparisons(T, np_het, STOCHASTICITY, VOLATILITY)

    env = FixedStochVolEnv(0.0, STOCHASTICITY, VOLATILITY, T)
    als_fixed = Hetlearner(env, 0.5)
    alsi = ALSEstimator(als_fixed, T, W)
    kf = KalmanFilter(env)

    ran = range(0.35, 1.0, np_het)
    hetlearners = [Hetlearner(env.valences[1], kg, 0.0) for kg in ran]
    hc =  HetlearnCollection(hetlearners, [0.0,0.0,0.0], [2.0, 1.0, 0.1])

    function state_updates(c::Collection, t)
        foreach(c.particles) do particle
            StateUpdate!(particle, env.rewards[t])
        end
    end

    hu = BuildHetlearnUpdate(hc; τ = 0.1) #QQQ note tau here
    function weight_updates(c::HetlearnCollection, t) #QQQ i'm  not seeing why I need T here?
        hu(c)
    end

    ##### Things to record ##########
    upd, sv_record = RecordedOnly(
        t -> cleaned_stoch_vol(alsi),
        1:T,
        :ALSstochasticity_and_volatility 
    )()

    upd2, sv_sum_rec = RecordedOnly(
        t -> cleaned_stoch_vol(alsi) |> sum,
        1:T,
        :summed_stochvol
    )()

    kg_het_upd, kg_het_record = RecordedOnly(
        t -> kalman_gain(hc),
        # t -> kalman_gain_top(hc),
        1:T,
        :kg_het
    )(; data_type=Float64)

    kg_ALS_upd, kg_ALS_record = RecordedOnly(
        t -> ALS_kalman_gain(kf, alsi), #should check that this function is ok
        1:T,
        :kg_ALS
    )(; data_type=Float64)

    kg_het_opt_upd, kg_het_opt_record = RecordedOnly(
        t -> OptimalKalmanGain(env, hc, t),
        1:T,
        :kg_het_opt
    )(; data_type=Float64)

    weights_het_upd, weights_het_record = RecordedOnly(
        t -> weights(hc), # weights(hc) is just hc.weights and is KalmanFilter.jl
        1:T,
        :weights_het
    )()

    SquaredError_het, SquaredError_het_record = RecordedOnly(
        t -> SquaredError(env, hc, t),
        1:T,
        :SquaredError_het
    )(; data_type=Float64)

    SquaredError_alsi, SquaredError_alsi_record = RecordedOnly(
        t -> SquaredError_part(env, kf, t),
        1:T,
        :SquaredError_ALS
    )(; data_type=Float64)

    ##################

    records = CompositeRecord(
        [sv_record, sv_sum_rec,

        kg_het_record, kg_ALS_record,

        kg_het_opt_record,

        weights_het_record, 

        SquaredError_het_record,
        SquaredError_alsi_record,
        ]
    )

    foreach(1:T) do t
        
        upd(t) #record only
        upd2(t) #record only

        kg_het_upd(t) #record only
        kg_ALS_upd(t) #record only
        kg_het_opt_upd(t) #record only

        weights_het_upd(t) #record only
        
        StateUpdate!(kf, env.rewards[t])          # 1. filter with reward[t] using t-1 noise params (no look-ahead)
        StateUpdate!(als_fixed, env.rewards[t])   # 2. update Hetlearner innovation with reward[t]
        StateUpdate!(alsi, als_fixed, t)          # 3. update ALS correlations from updated als_fixed
        ALSParamsUpdate!(kf, alsi, t)             # 4. update kf noise params for use at t+1

        weight_updates(hc, t)
        state_updates(hc, t)

        SquaredError_het(t)  #record only
        SquaredError_alsi(t)  #record only
        
    end

    return records
end


##################################
### Example Usage ###############

# T = 1000
# W = 5 #window length for ALS Estimator
# STOCHASTICITY = 1.2
# VOLATILITY = 2.3
# np_het = 3
# records_als = run_ALS_comparisons(T, np_het, STOCHASTICITY, VOLATILITY)
