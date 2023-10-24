using Distributed
using Revise
using CSV, DataFrames
using PlaceholderLikelihood
using Random, Distributions
using LaTeXStrings

include(joinpath("..", "Experiments", "Models", "logistic.jl"))
seed_num=1

model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes);

bivariate_confidenceprofiles!(model, [[1,2]], 150, method=IterativeBoundaryMethod(20, 5,5, 0.15, 1.0, use_ellipse=true))
true_boundary = model.biv_profiles_dict[1].confidence_boundary
PlaceholderLikelihood.minimum_perimeter_polygon!(true_boundary)
true_boundary = hcat(true_boundary, true_boundary[:, 1])

model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, [1,1,1]);
Random.seed!(seed_num)
bivariate_confidenceprofiles!(model, [[1,2]], 4, method=RadialRandomMethod(4))

using Plots; gr()
using Plots.PlotMeasures

format = (size=(500, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    xlims=(0.005, 0.025),
    # ylims=(0.1,0.8),
    xlabel=θnames[1],
    ylabel=θnames[2],
    title="",
    rightmargin=3mm,
    # aspect_ratio=:equal, 
    legend_position=:topright, palette=:Paired_7)

plt = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
PlaceholderLikelihood.minimum_perimeter_polygon!(pnts)
pnts = pnts[:, [1:4..., 1]]
plot!(pnts[1, :], pnts[2, :], label="Undistorted search directions", marker=(:circle), color=3, msw=0, ms=5)
# scatter!([model.core.θmle[1]], [model.core.θmle[2]], label="MLE point", ms=5, msw=0, color=6)


model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes);
Random.seed!(seed_num)
bivariate_confidenceprofiles!(model, [[1,2]], 4, method=RadialRandomMethod(4))

pnts = model.biv_profiles_dict[1].confidence_boundary
PlaceholderLikelihood.minimum_perimeter_polygon!(pnts)
pnts = pnts[:, [1:4..., 1]]
plot!(pnts[1, :], pnts[2, :], label="Distorted search directions", marker=(:diamond), color=4, msw=0, ms=5)

function likelihoodFuncLog10(Θ, data) 
    return loglhood(exp10.(Θ), data) 
end

opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
model = initialise_LikelihoodModel(likelihoodFuncLog10, data, θnames, log10.(θG), log10.(lb .+ 0.001), log10.(ub), [1, 1, 1], optimizationsettings=opt_settings);
Random.seed!(seed_num)
bivariate_confidenceprofiles!(model, [[1,2]], 4, method=RadialRandomMethod(4))

pnts = model.biv_profiles_dict[1].confidence_boundary
PlaceholderLikelihood.minimum_perimeter_polygon!(pnts)
pnts = exp10.(pnts[:, [1:4..., 1]])

plot!(pnts[1, :], pnts[2, :], label=L"\log_{10}"*" parameter space", marker=(:square), color=5, msw=0, ms=5)
scatter!([exp10(model.core.θmle[1])], [exp10(model.core.θmle[2])], label="MLE point", ms=5, msw=0, color=6)

display(plt)

output_location = joinpath("Bespoke graphics", "4point_radialrandom_par_mags");
savefig(plt, joinpath(output_location, "logistic.pdf"))