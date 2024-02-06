using Distributed
using Revise
using CSV, DataFrames
using Integrals
# if nprocs()==1; addprocs(10) end
using LikelihoodBasedProfileWiseAnalysis
using ProgressMeter
@everywhere using Revise
@everywhere using Random, Distributions
@everywhere using LikelihoodBasedProfileWiseAnalysis

include(joinpath("Models", "birth-death_stochastic_identifiable.jl"));
output_location = joinpath("Experiments", "Outputs", "stochastic_identifiable")

# do experiments
model = initialise_LikelihoodModel(loglhood, predictfunction, errorfunction, data, θnames, θG, lb, ub, par_magnitudes);

# the parameter range used to create surrogate impacts Σ_εgθ and thus all other things:
# confidence intervals get wider as more parameter range is included AND the 
# surrogate correction used to create reference tolerance intervals also gets inflated
univariate_confidenceintervals!(model)
get_points_in_intervals!(model, 30, additional_width=0.2)

generate_predictions_univariate!(model, t_pred, 1.0, use_distributed=false)

using Plots, LaTeXStrings; gr()

plts = plot_univariate_profiles(model, 0.2, 0.2)

for i in eachindex(plts)
    vline!(plts[i], [θ_true[i]], lw=3, linestyle=:dash,
        label="true value", title="", legend_position=ifelse(i == 1, false, :topright))
    xlabel!(plts[i], ifelse(i == 1, L"\beta", L"\delta"))

    display(plts[i])
end

plt = plot(plts..., layout=(1, 2), size=(500, 400), dpi=(300))
savefig(plt, joinpath(output_location, "uni_profiles_identifiable.pdf"))

plt = plot_predictions_union(model, t_pred)
display(plt)

plt = plot_realisations_union(model, t_pred)
display(plt)

for _ in 1:100
    new_y_obs = hcat(birth_death_firstreact(training_gen_args.t_single, θ_true..., N0)...)
    plot!(plt[1], t_pred, new_y_obs[:,1], label=false)
    plot!(plt[2], t_pred, new_y_obs[:,2], label=false)
end
display(plt)

# if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
#     uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 1000, θ_true, collect(1:2), show_progress=true, distributed_over_parameters=false)
#     display(uni_coverage_df)
#     CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
# end