using Random, Distributions
using LaTeXStrings
using Revise
using PlaceholderLikelihood

output_location = joinpath("Experiments", "Outputs", "profile_paths_demo");

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

model = initialise_LikelihoodModel(lnlike, predictfunction, errorfunction, data, [:μ, :σ_squared], [2.,1.], [-1., 0.01], [5., 5.], [1.,1.]);

univariate_confidenceintervals!(model, num_points_in_interval=300)
equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, 2, 1)
univariate_confidenceintervals!(model, num_points_in_interval=300, confidence_level=equiv_simul_conf_level)


equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, 1, 2)
bivariate_confidenceprofiles!(model, 200, method=RadialMLEMethod(), confidence_level=equiv_simul_conf_level)
bivariate_confidenceprofiles!(model, 200, method=RadialMLEMethod())
sample_bivariate_internal_points!(model, 200)

# dimensional_likelihood_samples!(model, 2, 1000000)

using Plots; gr()
format = (size=(550, 400), dpi=300, title="", legend_position=:outerright, background_color_legend=RGBA(1, 1, 1, 0.9))

plt = plot(; palette=:Paired, xlims=(1.8, 2.45), ylims=(0.75, 1.25), ylabel=latexstring("\\theta^\\textrm{o}"), xlabel=latexstring("\\theta^M"), format...)


for i in 1:2
    points = model.biv_profiles_dict[i].confidence_boundary
    points = hcat(points, points[:,1])
    plot!(plt, points[1, :], points[2, :], label=[L"\ell_{c,1}=-\Delta_{1, \,0.95}/2", L"\ell_{c,2}=-\Delta_{2, \,0.95}/2"][i], colour=[1, 2][i], lw=3)
end

for j in 1:2
    for i in 1:2
        points = model.uni_profiles_dict[ifelse(i==1, j, j+2)].interval_points.points

        plot!(plt, points[1, :], points[2, :], 
            label=[L"\psi\in\{\theta^M\}, \ell_{c,1}", L"\psi\in\{\theta^\textrm{o}\}, \ell_{c,1}",
                L"\psi\in\{\theta^M\}, \ell_{c,2}", L"\psi\in\{\theta^\textrm{o}\}, \ell_{c,2}"][ifelse(i == 1, j, j + 2)],
            colour=[3, 5, 4, 6][ifelse(i == 1, j, j + 2)], linestyle=[:solid, :dashdot][i], lw=[3, 4][i])
    end
end


for j in 1:2
    for i in 2:-1:1
        points = model.uni_profiles_dict[ifelse(i==1, j, j+2)].interval_points.points

        plot!(plt, points[1, :], points[2, :], 
            label=nothing,
            colour=[3, 5, 4, 6][ifelse(i == 1, j, j + 2)], linestyle=[:solid, :dashdot][i], lw=[3,4][i])
    end
end

display(plt)

savefig(plt, joinpath(output_location, "theta_confidence_set.pdf"))

generate_predictions_univariate!(model, ["z"], 1.0, region=0.95)
generate_predictions_bivariate!(model, ["z"], 1.0, region=0.95)

using StatsPlots
# 2D
# generate_predictions_dim_samples!(model, ["z"], 1.0)
extrema1 = model.uni_predictions_dict[1].realisations.extrema
extrema2 = model.uni_predictions_dict[2].realisations.extrema
extrema_uni1 = [min(extrema1[1], extrema2[1]) max(extrema1[2], extrema2[2])]

extrema1 = model.uni_predictions_dict[3].realisations.extrema
extrema2 = model.uni_predictions_dict[4].realisations.extrema
extrema_uni2 = [min(extrema1[1], extrema2[1]) max(extrema1[2], extrema2[2])]

extrema_full = model.biv_predictions_dict[2].realisations.extrema

plt = plot(true_dist; xlabel=latexstring("y"), label="Density", fill=(0, 0.3), palette=:Paired_6, format..., legend_position=:topright, size=(400,400))
vline!(ref_interval, label="Reference", lw=2)
vline!(transpose(extrema_uni1), label="Tolerance, univariate "*L"\ell_{c,1}", linestyle=:dash, lw=2)
vline!(transpose(extrema_uni2), label="Tolerance, univariate " * L"\ell_{c,2}", linestyle=:dash, lw=2)
vline!(transpose(extrema_full), label="Tolerance, full " * L"\ell_{c,2}", linestyle=:dash, lw=2)
# savefig(plt, joinpath(output_location, "interval_ranges_over_density.pdf"))