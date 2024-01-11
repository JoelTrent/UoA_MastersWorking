using Random, Distributions
using LaTeXStrings
using Revise
using PlaceholderLikelihood

output_location = joinpath("Experiments", "Outputs", "tolerance_intervals_demo");

θ_true = [2,1]
true_dist = Normal(θ_true[1], θ_true[2])
n = 100
Random.seed!(3)
y_obs = rand(true_dist, n) 

ref_interval = quantile(true_dist, [0.025, 0.975])

data = (y_obs=y_obs, dist=Normal(0, θ_true[2]), t=["z"])

function lnlike(θ, data)
    return sum(loglikelihood(Normal(0, θ[2]), data.y_obs .- θ[1]))
end

function predictfunction(θ, data, t=["z"])
    return [θ[1]*1.0]
end

errorfunction(a,b,c) = normal_error_σ_estimated(a,b,c, 2)

model = initialise_LikelihoodModel(lnlike, predictfunction, errorfunction, data, [:μ, :σ], [2.,1.], [-1., 0.01], [5., 5.], [1.,1.]);

univariate_confidenceintervals!(model, num_points_in_interval=300)
dimensional_likelihood_samples!(model, 2, 1000000)

using Plots; gr()
format=(size=(400,400), dpi=300, title="", legend_position=:topright)


plt = plot_univariate_profiles(model; format...)
vline!(plt[1], [θ_true[1]], label=L"\theta^M", xlabel=L"\theta^M", lw=2, linestyle=:dash)
vline!(plt[2], [θ_true[2]], label=L"\theta^\textrm{o}", xlabel=L"\theta^M", lw=2, linestyle=:dash)
display(plt[1])
display(plt[2])
savefig(plt[1], joinpath(output_location, "theta1_profile.pdf"))
savefig(plt[2], joinpath(output_location, "theta2_profile.pdf"))

plt = plot_bivariate_profiles(model; for_dim_samples=true, markeralpha=0.4, max_internal_points=10000, ylabel=latexstring("\\theta^\\textrm{o}"), xlabel=latexstring("\\theta^M"), format...)
scatter!(plt[1], [θ_true[1]], [θ_true[2]], label="θtrue", color="black", ms=5, msw=0)
display(plt[1])
savefig(plt[1], joinpath(output_location, "theta_confidence_set.pdf"))

generate_predictions_univariate!(model, ["z"], 1.0)

# 1D
lq = model.uni_predictions_dict[1].realisations.lq
uq = model.uni_predictions_dict[1].realisations.uq

plt = plot(1:length(lq), transpose(uq); xlabel="Confidence Set Sample", ylabel="Interval", label="Upper", palette=:Paired_6, format...)
plot!(1:length(lq), transpose(lq), label="Lower")
savefig(plt, joinpath(output_location, "all_interval_ranges_profile1.pdf"))

lq = model.uni_predictions_dict[2].realisations.lq
uq = model.uni_predictions_dict[2].realisations.uq

plt = plot(1:length(lq), transpose(uq); xlabel="Confidence Set Sample", ylabel="Interval", label="Upper", palette=:Paired_6, format...)
plot!(1:length(lq), transpose(lq), label="Lower")
savefig(plt, joinpath(output_location, "all_interval_ranges_profile2.pdf"))

extrema1 = model.uni_predictions_dict[1].realisations.extrema
extrema2 = model.uni_predictions_dict[2].realisations.extrema
extrema = [min(extrema1[1],extrema2[1]) max(extrema1[2], extrema2[2])]

using StatsPlots
plt = plot(true_dist; xlabel=latexstring("y"), label="Density", fill=(0, 0.3), palette=:Paired_6, format...)
vline!(ref_interval, label="Reference", lw=2)
vline!(transpose(extrema1), label="Tolerance, "*L"\psi=\theta^M", linestyle=:dash, lw=2)
vline!(transpose(extrema2), label="Tolerance, "*L"\psi=\theta^\textrm{o}", linestyle=:dash, lw=3)
vline!(transpose(extrema), label="Tolerance, union", linestyle=:dashdot, lw=2, alpha=0.7)
savefig(plt, joinpath(output_location, "interval_ranges_over_density_profile.pdf"))

# 2D
generate_predictions_dim_samples!(model, ["z"], 1.0)
lq = model.dim_predictions_dict[1].realisations.lq
uq = model.dim_predictions_dict[1].realisations.uq
extrema=model.dim_predictions_dict[1].realisations.extrema

plt = plot(1:length(lq), transpose(uq); xlabel="Confidence Set Sample", ylabel="Interval", label="Upper", palette=:Paired_6, format...)
plot!(1:length(lq), transpose(lq), label="Lower")
savefig(plt, joinpath(output_location, "all_interval_ranges.pdf"))

plt = plot(true_dist; xlabel=latexstring("y"),label="Density", fill=(0, 0.3), palette=:Paired_6, format...)
vline!(ref_interval, label="Reference", lw=2)
vline!(transpose(extrema), label="Tolerance", linestyle=:dash, lw=2)
savefig(plt, joinpath(output_location, "interval_ranges_over_density.pdf"))