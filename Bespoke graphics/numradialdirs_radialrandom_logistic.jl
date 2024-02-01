using Distributed
using Revise
using CSV, DataFrames
using LikelihoodBasedProfileWiseAnalysis
using Random, Distributions
using LaTeXStrings

output_location = joinpath("Bespoke graphics", "8point_radialrandom_numdirs");
include(joinpath("..", "Experiments", "Models", "logistic.jl"))
seed_num=3

model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes);

bivariate_confidenceprofiles!(model, [[1,2]], 150, method=IterativeBoundaryMethod(20, 5,5, 0.15, 1.0, use_ellipse=true))
true_boundary = model.biv_profiles_dict[1].confidence_boundary
LikelihoodBasedProfileWiseAnalysis.minimum_perimeter_polygon!(true_boundary)
true_boundary = hcat(true_boundary, true_boundary[:, 1])

model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes);

using Plots; gr()
using Plots.PlotMeasures

format = (size=(400, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    xlims=(0.005, 0.025),
    # ylims=(0.1,0.8),
    xlabel=θnames[1],
    ylabel=θnames[2],
    title="",
    rightmargin=3mm,
    # aspect_ratio=:equal, 
    legend_position=:topright, palette=:Paired_7)

Random.seed!(seed_num)
bivariate_confidenceprofiles!(model, [[1,2]], 8, method=RadialRandomMethod(2, false))

plt = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary

pnts = model.biv_profiles_dict[1].confidence_boundary
internal_pnts = model.biv_profiles_dict[1].internal_points.points
LikelihoodBasedProfileWiseAnalysis.minimum_perimeter_polygon!(pnts)
pnts = pnts[:, [1:8..., 1]]
scatter!(internal_pnts[1, :], internal_pnts[2, :], label="Internal points", marker=(:circle), color=4, msw=0, ms=5, opacity=0.75)
plot!(pnts[1, :], pnts[2, :], label="Number of directions = 2", marker=(:circle), color=3, msw=0, ms=5)
display(plt)

savefig(plt, joinpath(output_location, "2directions.pdf"))

Random.seed!(seed_num)
bivariate_confidenceprofiles!(model, [[1,2]], 8, method=RadialRandomMethod(4, false))
plt = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);

pnts = model.biv_profiles_dict[2].confidence_boundary
internal_pnts = model.biv_profiles_dict[2].internal_points.points
LikelihoodBasedProfileWiseAnalysis.minimum_perimeter_polygon!(pnts)
pnts = pnts[:, [1:8..., 1]]
scatter!(internal_pnts[1, :], internal_pnts[2, :], label="Internal points", marker=(:circle), color=4, msw=0, ms=5, opacity=0.75)
plot!(pnts[1, :], pnts[2, :], label="Number of directions = 4", marker=(:circle), color=3, msw=0, ms=5)


display(plt)
savefig(plt, joinpath(output_location, "4directions.pdf"))


Random.seed!(seed_num)
bivariate_confidenceprofiles!(model, [[1,2]], 8, method=RadialRandomMethod(8, false))
plt = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);

pnts = model.biv_profiles_dict[3].confidence_boundary
internal_pnts = model.biv_profiles_dict[3].internal_points.points
LikelihoodBasedProfileWiseAnalysis.minimum_perimeter_polygon!(pnts)
pnts = pnts[:, [1:8..., 1]]
scatter!(internal_pnts[1, :], internal_pnts[2, :], label="Internal points", marker=(:circle), color=4, msw=0, ms=5, opacity=0.75)
plot!(pnts[1, :], pnts[2, :], label="Number of directions = 8", marker=(:circle), color=3, msw=0, ms=5)

display(plt)

savefig(plt, joinpath(output_location, "8directions.pdf"))