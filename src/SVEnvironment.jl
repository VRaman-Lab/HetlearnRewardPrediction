 struct StochVolEnv{T}
    valences::Vector{T}
    rewards::Vector{T}
    stochasticity::Vector{T}
    volatility::Vector{T}
end

volatility(env::StochVolEnv, time) = env.volatility[time]
stochasticity(env::StochVolEnv, time) = env.stochasticity[time]

"""
Generate a matrix of trial outputs
"""
function GaussianStochVolEnv(initial_valence, stochasticity_trace, volatility_trace)
    num_trials = length(stochasticity_trace)
    valences   = zeros(eltype(initial_valence), num_trials)
    
    get_reward(i) = rand(Normal(valences[i], sqrt(stochasticity_trace[i])))
    function get_valence(i)
        (i == 1) && return initial_valence
        return rand(Normal(valences[i-1], sqrt(volatility_trace[i-1])))
    end

    foreach(axes(valences,1)) do t
        valences[t] = get_valence(t)
    end

    rewards = get_reward.(1:num_trials)

    return StochVolEnv(valences, rewards, stochasticity_trace, volatility_trace)
end

function FixedStochVolEnv(initial_valence, stochasticity::Number, volatility::Number, num_trials)
    stochasticity_trace = fill(stochasticity, num_trials)
    volatility_trace = fill(volatility, num_trials)
    return GaussianStochVolEnv(initial_valence, stochasticity_trace, volatility_trace)
end

