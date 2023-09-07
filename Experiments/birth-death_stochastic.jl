using Distributed
using Revise
# if nprocs()==1; addprocs(10) end
using PlaceholderLikelihood
@everywhere using Revise
@everywhere using Random, Distributions
@everywhere using KernelDensity, HighestDensityRegions
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "birth-death_stochastic.jl"));

# do experiments
model = initialise_LikelihoodModel(loglhood, predictFunc, data, θnames, θG, lb, ub, par_magnitudes);

# NEED TO TEST errorFunc

