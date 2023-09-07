using Distributed
using Revise
# if nprocs()==1; addprocs(10) end
using PlaceholderLikelihood
@everywhere using Revise
@everywhere using Random, Distributions, DifferentialEquations, StaticArrays
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "lotka_volterra.jl"))

# do experiments
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes);