"""

Take Hetlearn particle with fixed Kalman Gain as ALS learner. 
Take Kalman Filter with ALS_State_Update as Full learner

ALSEstimator owns:
    Hetlearn Particle
    innovation_holder = Vector of all past RPEs
    buildScalarAlS = FIXED Matrix 2xWindow_length
    R: VEctor/matrix of lagged correlations that gets UPDATED


    1. Hetlearn Particle state update
Order of updates (inside ALSEstimator): 

    2. add column to innovation_holder
    3. calculate R

Order of updates (inside Kalman Filter) ALS_update!(t, pd, alse)
    1. Calculate volatility and stochasticity from alse
    2. seed into kalman gain
    3. normal state update

"""

struct ALSEstimator{T<:Number}
    innovation_holder::Vector{T} # holds ALL RPEs
    ALS::AbstractMatrix{T} # fixed matrix 
    ℛ::Vector{T} #dynamically calculated
end


function ALSEstimator(h::Hetlearner, final_time::Integer, window_length::Integer)
    α = h.kalman_gain
    return ALSEstimator(
        zeros(final_time),
        buildScalarALS(α, window_length),
        zeros(window_length)
    )
end

window_length(a::ALSEstimator) = length(a.ℛ)

stoch_vol(a::ALSEstimator) = a.ALS \ a.ℛ |> reverse # A \ b solves the linear system A * x = b for x. Hence this returns x

function cleaned_stoch_vol(a::ALSEstimator)
    s, v = stoch_vol(a)
    if min(s, v) < 0.0
        tot = s + v
        (s < v) && (s = 0; v = tot)
        (s > v) && (v = 0; s = tot)
    end


    return [s, v]
end


function StateUpdate!(a::ALSEstimator, h::Hetlearner, t)
    window_length = length(a.ℛ)
    a.innovation_holder[t] = h.current_rpe
    if t > window_length
         a.ℛ[:] = build_lagged_correlations(a::ALSEstimator, t)
    end

    nothing
end


"""
calculates kalman gain of ALS estimaotr. uses cleaned stoch vol filtering out negative values
"""
function ALS_kalman_gain(kf::KalmanFilter, a::ALSEstimator)
    try
        x_t, c_t  = kf.state, kf.variance
        s_t, v_t = cleaned_stoch_vol(a) # cleans out unphysical negative values
        αₜ = (c_t + v_t) / (c_t + s_t + v_t)
        return αₜ
    catch
        return 0.5  # this is the initial KG set for the first trials when the ALSEstimator doesn't have an estimate yet
    end
end

function KronSum(A, B)
    Nₐ = size(A, 1)
    Nᵦ = size(B, 1)
    kron(A, I(Nᵦ)) + kron(I(Nₐ), B)
end

"""
My own worked version of building the ALS matrix, in the easy scalar case. It works correctly.
"""
function buildScalarALS(kalman_gain, N)
    L = kalman_gain
    function lagged_correlation_coefficients(j)
        if j == 0
            return [
                1.0 / (L * (2.0 - L))
                (L / (2.0 - L)) + 1.0
            ]
        elseif j > 0
            return [
                ((1.0 - L)^j) / (L * (2.0 - L))
                (L * (1.0 - L)^j) / (2.0 - L) - L * (1.0 - L)^(j - 1)
            ]
        end
    end

    return hcat([lagged_correlation_coefficients(j) for j in 0:N-1]...)'
end

function build_lagged_correlations(a::ALSEstimator, t)
    K = t
    N = window_length(a)
    (K <= N) && error("trying to build lagged correlations too early")
    ih = a.innovation_holder
    lag(j) = (1.0 / (K - j)) * sum(1:(K-j)) do i
        ih[i] * ih[i+j]
    end
    return [abs(i) |> lag for i in 0:N-1]
end


"""
Updates stochasticity and volatility of a Kalman Filter by using the ALS updates of alsi
"""
function ALSParamsUpdate!(kf, alse::ALSEstimator, t::Integer)
    if t > window_length(alse)
        s,v = cleaned_stoch_vol(alse)
        kf.stochasticity, kf.volatility = s, v
    end
    nothing
end
