# include("PD_Hetlearn_helpers.jl") will need to include this for the collection_average function (unless it moves)

mutable struct KalmanFilter{V}
    state::V
    variance::V
    volatility::V
    stochasticity::V
end
state(kf::KalmanFilter) = kf.state

function KalmanFilter(sv::StochVolEnv)
    state = sv.valences[1]
    variance = zero(eltype(state))
    volatility = sv.volatility[1]
    stochasticity = sv.stochasticity[1]
    return KalmanFilter(state, variance, volatility, stochasticity)
end

function kalman_gain(KF::KalmanFilter)
    x̂ₜ, ĉₜ, vₜ, wₜ = KF.state, KF.variance, KF.volatility, KF.stochasticity
    αₜ = (ĉₜ + vₜ) / (ĉₜ + vₜ + wₜ)
    return αₜ 
end 

function StateUpdate!(pd::KalmanFilter, yₜ)
    x̂ₜ, ĉₜ, vₜ, wₜ = pd.state, pd.variance, pd.volatility, pd.stochasticity
    δₜ = yₜ - x̂ₜ
    αₜ = (ĉₜ + vₜ) / (ĉₜ + vₜ + wₜ)
    pd.state = x̂ₜ + αₜ * δₜ
    pd.variance = (1.0 - αₜ) * (ĉₜ + vₜ)
    nothing
end

################### Particle Filter ################################

mutable struct KalmanFilterCollection{P,N<:Number} <: Collection
    particles::Vector{P}
    weights::Vector{N}

end
KalmanFilterCollection(ps::Vector{P}) where {P} = KalmanFilterCollection(ps, ones(length(ps)) ./ length(ps))
weights(PC::Collection) = PC.weights
particles(PC::Collection) = PC.particles

"""
kalman gain for particle collection
"""
function kalman_gain(pc::Collection)
    return collection_average(pc, kalman_gain)
end

"""
custom kalman gain for LR particles: don't do weighted average over particles. Listen to top 3 weights (or only 2 weights if the best or worst weight is the first or last index)
"""
function kalman_gain_t3(pc::Collection)
    indx = argmax(weights(pc))
    N = length(pc.particles)
    
    # Ensure the range doesn't go out of bounds
    lo = max(indx - 1, 1)
    hi = min(indx + 1, N)
    only_consider = lo:hi

    relevant_weights = weights(pc)[only_consider]
    relevant_weights_norm = relevant_weights / norm(relevant_weights, 1)
    return sum(relevant_weights_norm .* kalman_gain.(pc.particles[only_consider]))
end

"""
custom kalman gain for LR particles: don't do weighted average over particles. Listen to top weight
"""
function kalman_gain_top(pc::Collection)
    indx = argmax(weights(pc))    
    return kalman_gain(pc.particles[indx])
end











