mutable struct Hetlearner{X<:Number}
    state::X
    kalman_gain::X
    current_rpe::X
end
state(h::Hetlearner) = h.state
Hetlearner(env::StochVolEnv, kg) = Hetlearner(env.valences[1], kg, 0.0)


function StateUpdate!(h::Hetlearner, yₜ)
    x̂ₜ, αₜ = h.state, h.kalman_gain
    δₜ = yₜ - x̂ₜ
    update = αₜ * δₜ
    h.state = x̂ₜ + update
    h.current_rpe = δₜ
    nothing
end

kalman_gain(H::Hetlearner) = H.kalman_gain


################### Particle Collection for Hetlearn ################################

mutable struct HetlearnCollection{P,N} <:Collection
    particles::Vector{P}
    weights::Vector{N}
    mse_estimates::Vector{N}
end

function HetlearnCollection(env::StochVolEnv, ran::AbstractRange)
    state = env.valences[1]
    return HetlearnCollection(state, ran)
end


function HetlearnCollection(state::Float64, ran)
    n = length(ran)
    hetlearners = [Hetlearner(state, kg, 0.0) for kg in ran]
    return HetlearnCollection(
            hetlearners,
            fill(1.0 / n, n),
            fill(0.0, n)
        )
end

function HetlearnCollection(env; min=0.0, max=1.0, n=3)
    ran = range(min, max, n)
    HetlearnCollection(env, ran)
end


function BuildHetlearnUpdate(h::HetlearnCollection; kwargs...)
    return BuildHetlearnUpdate(; kwargs...)
end


function softmax(x::AbstractVector{T}; beta::Real=5.0) where T<:Real
    scaled_x = x .* beta
    
    e = exp.(scaled_x .- maximum(scaled_x))
    
    return e ./ sum(e)
end

function BuildHetlearnUpdate(; mse = p -> p.current_rpe^2 , τ=0.1)

    function HetlearnUpdate!(h::HetlearnCollection)
        iszero(h.mse_estimates) && (h.mse_estimates = fill(mean(mse.(h.particles)), length(h.mse_estimates)))
        old_mses = deepcopy(h.mse_estimates)
        h.mse_estimates = map(h.particles, h.mse_estimates) do particle, mse_past
            (1.0 - τ) * mse_past + τ * mse(particle)
        end
        if iszero(h.mse_estimates) 
            if iszero(old_mses)
                new_weights = ones(length(h.mse_estimates))
            else
                new_weights = 1.0 ./ old_mses
            end
        else
            new_weights = 1.0 ./ (h.mse_estimates)
        end

        map!(new_weights, new_weights) do x
            isnan(x) ? zero(x) : x
        end
        h.weights = new_weights / norm(new_weights, 1)
        nothing

    end
end

function bestAlpha(hc::HetlearnCollection)
    LRs = [p.kalman_gain for p in hc.particles]
    errors = hc.mse_estimates

    # Fit quadratic MSE ≈ a*α² + b*α + c via least squares
    A = hcat(LRs .^ 2, LRs, ones(length(LRs)))
    a, b, _ = A \ errors

    # Vertex of upward-opening parabola; fall back to argmin if fit is invalid
    if a > 0
        min_LR = clamp(-b / (2a), minimum(LRs), maximum(LRs))
    else
        min_LR = LRs[argmin(errors)]
    end
    return min_LR
end
