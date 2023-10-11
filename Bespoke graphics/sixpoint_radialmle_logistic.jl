using Distributed
using Revise
using CSV, DataFrames
using PlaceholderLikelihood
using Random, Distributions

include(joinpath("..", "Experiments", "Models", "logistic.jl"))

model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes);

bivariate_confidenceprofiles!(model, [[1,2]], 150, method=IterativeBoundaryMethod(20, 5,5, 0.15, 1.0, use_ellipse=true))
true_boundary = model.biv_profiles_dict[1].confidence_boundary
PlaceholderLikelihood.minimum_perimeter_polygon!(true_boundary)
true_boundary = hcat(true_boundary, true_boundary[:, 1])

model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes);
bivariate_confidenceprofiles!(model, [[1,2]], 6, confidence_level=0.2, method=AnalyticalEllipseMethod(0.0, 1.0))
bivariate_confidenceprofiles!(model, [[1,2]], 6, method=RadialMLEMethod(0.0, 1.0))

using Plots; gr()
using Plots.PlotMeasures

format = (size=(400, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    # xlims=(10.0, 100),
    # ylims=(0.1,0.8),
    xlabel=θnames[1],
    ylabel=θnames[2],
    title="",
    rightmargin=3mm,
    # aspect_ratio=:equal, 
    legend_position=:topright, palette=:Paired_7)

plt = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts = pnts[:, [1:6..., 1]]
plot!(pnts[1, :], pnts[2, :], label="Ellipse, α=0.8", linestyle=:dash, marker=(:diamond), color=2, msw=0, ms=5)

pnts = model.biv_profiles_dict[2].confidence_boundary
pnts = pnts[:, [1:6..., 1]]
plot!(pnts[1, :], pnts[2, :], label="Boundary polygon", marker=(:circle), color=3, msw=0, ms=5)
scatter!([model.core.θmle[1]], [model.core.θmle[2]], label="MLE point", ms=5, msw=0, color=6)

output_location = joinpath("Bespoke graphics", "sixpoint_radialmle");
savefig(plt, joinpath(output_location, "logistic.pdf"))