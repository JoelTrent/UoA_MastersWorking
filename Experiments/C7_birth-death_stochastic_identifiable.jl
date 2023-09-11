using Distributed
using Revise
using CSV, DataFrames
# if nprocs()==1; addprocs(10) end
using PlaceholderLikelihood
@everywhere using Revise
@everywhere using Random, Distributions
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "birth-death_stochastic_identifiable.jl"));
output_location = joinpath("Experiments", "Outputs", "stochastic_identifiable")

# do experiments
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model)
# get_points_in_intervals!(model, 30, additional_width=0.2)

# generate_predictions_univariate!(model, t_pred, 1.0, use_distributed=false)

# using Plots; pyplot()

# plots = plot_univariate_profiles(model, 0.2, 0.4, palette_to_use=:Spectral_8)
# for i in eachindex(plots); display(plots[i]) end

if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
    uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 1000, θ_true, collect(1:2), show_progress=true, distributed_over_parameters=false)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
end