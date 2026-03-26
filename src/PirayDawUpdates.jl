function MultiplicativeBetaUpdate!(pd::KalmanFilter, η::Number, name::Symbol)

    default_β(η) = Beta(0.5 * η * inverse(1 - η), 0.5)
    inverse(x) = x^(-1)

    old_variance = getfield(pd, name)
    new_variance = (inverse(η) * inverse(old_variance) * rand(default_β(η))) |> inverse
    setfield!(pd, name, new_variance)
    nothing
end


function PirayDawParamsUpdate!(kf::KalmanFilter, η1::Number, η2::Number)
    MultiplicativeBetaUpdate!(kf, η1, :volatility)
    MultiplicativeBetaUpdate!(kf, η2, :stochasticity)
end

PirayDawParamsUpdate!(kf::KalmanFilter, η::Number) = PirayDawParamsUpdate!(kf, η, η)



function PirayDawWeightUpdate!(pc::KalmanFilterCollection, yₜ)
    pc.weights = map(pc.particles) do p
        pdf(Normal(
                p.state,
                p.variance + p.volatility + p.stochasticity
            ),
            yₜ) # i.e. likelihoods of each particle
    end
    pc.weights = pc.weights / norm(pc.weights, 1)
    nothing
end
