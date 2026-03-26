include("setup.jl")

function run_pd_comparisons(T, np_pd, np_het, stoch_trace, vol_trace; η = 0.9, τ = 0.1, min_lr= 0.3, max_lr=0.85)
    env = GaussianStochVolEnv(0.0, stoch_trace, vol_trace)
    pc = KalmanFilterCollection([KalmanFilter(env) for i in 1:np_pd])
    hc = HetlearnCollection(env; min=min_lr, max=max_lr, n=np_het)

    idealHet = Hetlearner(env, 0.5)

    function environmental_kg(sublearner, env, t)
        cov = (env.valences[t] - sublearner.state)^2
        optimal_kg = (cov + env.volatility[t]) / (cov + env.volatility[t] + env.stochasticity[t])
        return optimal_kg
    end

    function ideal_update(hetlearner, t)
        hetlearner.kalman_gain = environmental_kg(hetlearner, env, max(1, t))
        StateUpdate!(hetlearner, env.rewards[t])
    end

    # SystematicResampler cutoff
    sr = Dict(
        :hc => SystematicResampler(hc, cutoff=0.7),
        :pc => SystematicResampler(pc, cutoff=0.7)
    )

    function state_updates(c::Collection, t)
        foreach(c.particles) do particle
            StateUpdate!(particle, env.rewards[t])
        end
    end

    function weight_updates(c::KalmanFilterCollection, t)
        PirayDawWeightUpdate!(c, env.rewards[t])
    end

    hu = BuildHetlearnUpdate(hc; τ=τ) #QQQ here is where Tau is getting passed in

    function weight_updates(c::HetlearnCollection, t) #QQQ i'm not seeing why I need T here?
        hu(c)
    end

    function params_update(pc::KalmanFilterCollection)
        foreach(pc.particles) do particle
            PirayDawParamsUpdate!(particle, η)
        end
    end

    ##### Things to record ##########
    erupd, env_record = RecordedOnly(
        t -> [env.volatility[t], env.stochasticity[t], env.valences[t], env.rewards[t]],
        1:T,
        :volatility_and_stochasticity_and_valence_and_rewards
    )(; data_type=Vector{Float64})

    kg_het_upd, kg_het_record = RecordedOnly(
        t -> kalman_gain_top(hc), 
        1:T,
        :kg_het
    )(; data_type=Float64)

    kg_pd_upd, kg_pd_record = RecordedOnly(
        t -> kalman_gain_top(pc), 
        1:T,
        :kg_pd
    )(; data_type=Float64)

    kg_het_opt_upd, kg_het_opt_record = RecordedOnly(
        t -> OptimalKalmanGain(env, hc, t),
        1:T,
        :kg_het_opt
    )(; data_type=Float64)

    kg_pd_opt_upd, kg_pd_opt_record = RecordedOnly(
        t -> OptimalKalmanGain(env, pc, t),
        1:T,
        :kg_pd_opt
    )(; data_type=Float64)

    SquaredError_het, SquaredError_het_record = RecordedOnly(
        t -> SquaredError_top(env, hc, t), 
        1:T,
        :SquaredError_het
    )(; data_type=Float64)

    SquaredError_pd, SquaredError_pd_record = RecordedOnly(
        t -> SquaredError_top(env, pc, t), 
        1:T,
        :SquaredError_pd
    )(; data_type=Float64)

    SquaredError_ideal, SquaredError_ideal_record = RecordedOnly(
        t -> SquaredError_part(env, idealHet, t), 
        1:T,
        :SquaredError_ideal
    )(; data_type=Float64)

    valence_het, valence_het_record = RecordedOnly(
        t -> collection_average(hc, p -> p.state),
        1:T,
        :valence_het
    )(; data_type=Float64)

    valence_pd, valence_pd_record = RecordedOnly(
        t -> collection_average(pc, p -> p.state),
        1:T,
        :valence_pd
    )(; data_type=Float64)

    _resampler_updates_hc, _resampler_updates_hc_record = RecordedUpdate(
        (t) -> sr[:hc](hc),
        1:T,
        1:T,
        t -> sr[:hc].resampled,
        :resampler_hc
    )(; data_type=Bool)

    _resampler_updates_pc, _resampler_updates_pc_record = RecordedUpdate(
        t -> sr[:pc](pc),
        1:T,
        1:T,
        t -> sr[:pc].resampled,
        :resampler_pc
    )(; data_type=Bool)

    ##################

    records = CompositeRecord(
        [env_record,
        kg_het_record, 
        kg_pd_record,

        kg_het_opt_record,
        kg_pd_opt_record,

        SquaredError_het_record,
        SquaredError_pd_record,
        SquaredError_ideal_record,

        _resampler_updates_hc_record,
        _resampler_updates_pc_record,

        valence_het_record, 
        valence_pd_record,
        ]
    )

    foreach(1:T) do t
        erupd(t) 

        kg_het_upd(t)
        kg_pd_upd(t)
        kg_het_opt_upd(t)
        kg_pd_opt_upd(t)

        valence_het(t)
        valence_pd(t)
        # valence_topHet(t)

        params_update(pc)
        weight_updates(pc, t)
        weight_updates(hc, t)

        _resampler_updates_hc(t)
        _resampler_updates_pc(t)

        state_updates(pc, t)
        state_updates(hc, t)
        ideal_update(idealHet, t)

        # error between valence(T) and predict(T+1) 
        SquaredError_het(t) 
        SquaredError_pd(t)
        SquaredError_ideal(t)

    end

    return records
end


# T = 1000
# np_pd = 10 # num_particles_piraydaw
# np_het = 3 # num_particles_hetlearn
# η = 0.9
# stoch_trace = 4 .+ sin.(0.1 .* (1:T))
# vol_trace = 3 .+ 2 .* cos.(0.01 .* (1:T))

# records = run_pd_comparisons(T, np_pd, np_het, stoch_trace, vol_trace; η = 0.9)
# vol_stoch_val = records(:volatility_and_stochasticity_and_valence).summary
# volatility = [v[1] for v in vol_stoch_val]
# stochasticity = [v[2] for v in vol_stoch_val]
# # valence = [v[3] for v in vol_stoch_val]

# kg_het =  records(:kg_het).summary
# kg_pd = records(:kg_pd).summary
# kg_het_opt = records(:kg_het_opt).summary
# kg_pd_opt = records(:kg_pd_opt).summary

# se_het = records(:SquaredError_het).summary
# se_pd = records(:SquaredError_pd).summary
