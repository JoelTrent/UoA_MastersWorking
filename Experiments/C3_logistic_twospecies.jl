using Distributed
using Revise
using CSV, DataFrames
if nprocs()==1; addprocs(10) end
using PlaceholderLikelihood
@everywhere using Revise
@everywhere using Random, Distributions, DifferentialEquations
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "logistic_twospecies.jl"))
output_location = joinpath("Experiments", "Outputs", "logistic_twospecies");

# do experiments
opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=5,))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
    uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 100, θ_true, collect(1:7), show_progress=true, distributed_over_parameters=false)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
end