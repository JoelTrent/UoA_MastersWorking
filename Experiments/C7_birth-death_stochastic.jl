using Distributed
using Revise
using CSV, DataFrames
# if nprocs()==1; addprocs(10) end
using PlaceholderLikelihood
@everywhere using Revise
@everywhere using Random, Distributions
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "birth-death_stochastic.jl"));
output_location = joinpath("Experiments", "Outputs", "stochastic");

# do experiments
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model)
get_points_in_intervals!(model, 30, additional_width=0.2)

generate_predictions_univariate!(model, t_pred, 1.0)

using Plots; pyplot()
# using StatsPlots

# plots = plot_univariate_profiles(model, 0.2, 0.4, palette_to_use=:Spectral_8)
# for i in eachindex(plots); display(plots[i]) end
plot = plot_predictions_union(model, t_pred)
display(plot)

model_sip = initialise_LikelihoodModel(loglhood_XYtoxy_sip, data, θnames_sip, θG_sip, lb_sip, ub_sip);
# univariate_confidenceintervals!(model_sip)
# get_points_in_intervals!(model_sip, 30, additional_width=0.2)

# plots = plot_univariate_profiles(model_sip, 0.2, 0.4, palette_to_use=:Spectral_8)
# for i in eachindex(plots); display(plots[i]) end


if !isfile(joinpath(output_location, "univariate_parameter_coverage_sip.csv"))
    uni_coverage_df = check_univariate_parameter_coverage(data_generator_XYtoxy_sip, training_gen_args, model_sip, 1000, xytoXY_sip(θ_true), [1], show_progress=true, distributed_over_parameters=false)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage_sip.csv"), uni_coverage_df)
end