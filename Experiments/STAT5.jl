using Distributed
using Revise
# if nprocs()==1; addprocs(10) end
using PlaceholderLikelihood
@everywhere using Revise
@everywhere using DifferentialEquations, LSODA, Random, Distributions
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "STAT5.jl"))

# do experiments
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model)
get_points_in_intervals!(model, 100, additional_width=0.2)

generate_predictions_univariate!(model, t_pred, 1.0, use_distributed=false)

using Plots; pyplot()

plots = plot_univariate_profiles(model, 0.2, 0.4, palette_to_use=:Spectral_8)
for i in eachindex(plots); display(plots[i]) end
plot = plot_predictions_union(model, t_pred)