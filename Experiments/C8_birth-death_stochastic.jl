using Distributed
using Revise
using CSV, DataFrames
# if nprocs()==1; addprocs(10) end
using LikelihoodBasedProfileWiseAnalysis
@everywhere using Revise
@everywhere using Random, Distributions
@everywhere using LikelihoodBasedProfileWiseAnalysis

include(joinpath("Models", "birth-death_stochastic.jl"));
output_location = joinpath("Experiments", "Outputs", "stochastic");

# do experiments
model = initialise_LikelihoodModel(loglhood, predictfunction, errorfunction, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model)
get_points_in_intervals!(model, 30, additional_width=0.2)

# require t_pred = data.t in this case
generate_predictions_univariate!(model, t_pred, 1.0)

using Plots, LaTeXStrings; gr()
# using StatsPlots

plts = plot_univariate_profiles(model, 0.2, 0.2)

for i in eachindex(plts)
    vline!(plts[i], [θ_true[i]], lw=3, linestyle=:dash, xlims=(0.01, 1.9), 
    label="true value", title="", xticks=0.2:0.4:2.0, legend_position=ifelse(i==1, false, :topright)) 
    xlabel!(plts[i], ifelse(i==1, L"\beta", L"\delta"))

    display(plts[i])
end

plt = plot(plts..., layout=(1,2), size=(450,400), dpi=(300))
savefig(plt, joinpath(output_location, "uni_profiles_nonidentifiable.pdf"))

plt = plot_predictions_union(model, t_pred)
display(plt)

plt = plot_realisations_union(model, t_pred)
display(plt)

model_sip = initialise_LikelihoodModel(loglhood_XYtoxy_sip, predictfunction_XYtoxy_sip, errorfunction_XYtoxy_sip, data, θnames_sip, θG_sip, lb_sip, ub_sip);
univariate_confidenceintervals!(model_sip)
get_points_in_intervals!(model_sip, 30, additional_width=0.2)
# require t_pred = data.t in this case
generate_predictions_univariate!(model_sip, t_pred, 1.0)


plts = plot_univariate_profiles(model_sip, 0.2, 0.4, palette_to_use=:Spectral_8)
for i in eachindex(plts); display(plts[i]) end

plts = plot_predictions_individual(model_sip, t_pred)
for i in eachindex(plts); display(plts[i]) end

plts = plot_realisations_individual(model_sip, t_pred)
for i in eachindex(plts); display(plts[i]) end

# if isfile(joinpath(output_location, "univariate_parameter_coverage_sip.csv"))
#     uni_coverage_df = check_univariate_parameter_coverage(data_generator_XYtoxy_sip, training_gen_args, model_sip, 2000, xytoXY_sip(θ_true), [1], distributed_over_parameters=false)
#     display(uni_coverage_df)
#     CSV.write(joinpath(output_location, "univariate_parameter_coverage_sip.csv"), uni_coverage_df)
# end