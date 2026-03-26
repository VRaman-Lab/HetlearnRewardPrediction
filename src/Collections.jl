"""
This will hold either hetlearn particles or PD particles

Need 
SystematicResampler
PirayDawWeightUpdate
HetlearnWeightUpdate
"""

"""
    Any concrete collections  must have following fields:
        weights
        ``
"""
abstract type Collection end


function collection_average(pc::Collection, f::Function)
    sum(zip(pc.particles, pc.weights)) do (p, w)
        f(p) * w
    end
end

function collection_variance(pc::Collection, f::Function)
    second_moment = sum(zip(pc.particles, pc.weights)) do (p, w)
        f(p)^2 * w
    end
    return second_moment - collection_average(pc, f)^2
end

function OptimalKalmanGain(env::StochVolEnv, c::Collection, t)
    
    cov = sum(abs2, env.valences[t] - collection_average(c, state)) #sum is not doing anything, but if later they were vectors it would

    optimal_kg = (cov + env.volatility[t]) / (cov + env.volatility[t] + env.stochasticity[t])
    return optimal_kg
end

function SquaredError(env::StochVolEnv, c::Collection, t)
    abs2(env.valences[t] - collection_average(c, state))
end

function SquaredError_top(env::StochVolEnv, c::Collection, t)
    indx = argmax(weights(c))    
    abs2(env.valences[t] - c.particles[indx].state)
end

function SquaredError_part(env::StochVolEnv, part, t)
    abs2(env.valences[t] - part.state)
end




