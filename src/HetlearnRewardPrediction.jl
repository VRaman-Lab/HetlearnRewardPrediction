module HetlearnRewardPrediction

# Write your package code here.

using LinearAlgebra
using Statistics
using Distributions

include("SVEnvironment.jl")
export StochVolEnv, GaussianStochVolEnv, FixedStochVolEnv

include("Collections.jl")
export Collection
export collection_average, collection_variance, OptimalKalmanGain, SquaredError, SquaredError_top, SquaredError_part

include("KalmanFilter.jl")
export KalmanFilter, StateUpdate!
export KalmanFilterCollection
export kalman_gain, kalman_gain_top
export weights, particles #these might make more sense in HetlearnParticles.jl but it's fine

include("PirayDawUpdates.jl")
export PirayDawParamsUpdate!, PirayDawWeightUpdate!

include("HetlearnParticles.jl")
export Hetlearner, HetlearnCollection, BuildHetlearnUpdate, bestAlpha

include("ALS.jl")
export ALSEstimator, ALSParamsUpdate!, cleaned_stoch_vol, ALS_kalman_gain

include("SystematicResampler.jl")
export SystematicResampler


end
