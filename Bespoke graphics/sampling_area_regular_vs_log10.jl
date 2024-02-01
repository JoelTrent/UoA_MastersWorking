using Distributed
using Revise
using CSV, DataFrames
using LikelihoodBasedProfileWiseAnalysis
using Random, Distributions
using LaTeXStrings

include(joinpath("..", "Experiments", "Models", "logistic.jl"))
output_location = joinpath("Bespoke graphics", "log10_sampling_area_impact");

model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes);

bivariate_confidenceprofiles!(model, [[1,2]], 150, method=IterativeBoundaryMethod(20, 5,5, 0.15, 1.0, use_ellipse=true))
true_boundary = model.biv_profiles_dict[1].confidence_boundary
LikelihoodBasedProfileWiseAnalysis.minimum_perimeter_polygon!(true_boundary)
true_boundary = hcat(true_boundary, true_boundary[:, 1])

using Plots; gr()
using Plots.PlotMeasures

format = (size=(500, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    xlims=(lb[1], ub[1]),
    ylims=(lb[2], ub[2]),
    xticks=lb[1]:0.01:ub[1],
    xlabel="λ",
    ylabel="K",
    title="",
    lw=4,
    rightmargin=3mm,
    # aspect_ratio=:equal, 
    legend_position=:topright, palette=:Paired_7)

plt = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);

# bivariate_confidenceprofiles!(model, [[1, 2]], 100, confidence_level=0.95, method=AnalyticalEllipseMethod(0.0, 1.0))
# pnts = model.biv_profiles_dict[2].confidence_boundary
# pnts = pnts[:, [1:100..., 1]]
# plot!(pnts[1, :], pnts[2, :], label="Ellipse, α=0.05", linestyle=:dash, color=2, msw=0, ms=5)

savefig(plt, joinpath(output_location, "logistic_standard.pdf"))
display(plt)

format = (size=(500, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    xlims=(log10(lb[1]+0.00000000000000000001), log10(ub[1])),
    ylims=(log10(lb[2]), log10(ub[2])),
    xticks=-20:2:0,
    xlabel=L"\log_{10}"*"λ",
    ylabel=L"\log_{10}"*"K",
    title="",
    rightmargin=3mm,
    lw=4,
    # aspect_ratio=:equal, 
    legend_position=:topright, palette=:Paired_7)

plt = plot(log10.(true_boundary[1, :]), log10.(true_boundary[2, :]); label=L"\log_{10}"*" true boundary",color=2, format...);



# function likelihoodFuncLog10(Θ, data)
#     return loglhood(exp10.(Θ), data)
# end

# opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
# model = initialise_LikelihoodModel(likelihoodFuncLog10, data, θnames, log10.(θG), log10.(lb .+ 0.001), log10.(ub), [1, 1, 1], optimizationsettings=opt_settings);

# bivariate_confidenceprofiles!(model, [[1, 2]], 100, confidence_level=0.95, method=AnalyticalEllipseMethod(0.0, 1.0))
# pnts = model.biv_profiles_dict[1].confidence_boundary
# pnts = pnts[:, [1:100..., 1]]
# plot!(pnts[1, :], pnts[2, :], label="Ellipse, α=0.05", linestyle=:dash, color=2, msw=0, ms=5)

display(plt)

savefig(plt, joinpath(output_location, "logistic_log10.pdf"))