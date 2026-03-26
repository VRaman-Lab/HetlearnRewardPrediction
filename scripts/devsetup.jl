import Pkg

"""
This should be run before any scripts at the start of each session

"""

project_root = dirname(@__DIR__)
Pkg.activate(joinpath(project_root, "scriptenv"))
Pkg.develop(path = project_root)

using HetlearnRewardPrediction
using SimulationHelper, Random 

