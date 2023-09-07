using Distributed
using Revise
using PlaceholderLikelihood
if nprocs()==1; addprocs(7) end
@everywhere using Revise
@everywhere using Random, Distributions, DifferentialEquations
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "logistic_twospecies.jl"))

# do experiments
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes);

uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 100, θ_true, collect(1:7), show_progress=true, distributed_over_parameters=false)
println(uni_coverage_df)