using Revise
using LikelihoodBasedProfileWiseAnalysis

using Random, Distributions, StaticArrays

# parameter -> data dist (forward) mapping
distrib_xy(xy) = Normal(xy[1] * xy[2], sqrt(xy[1] * xy[2] * (1 - xy[2]))) # 
# variables
varnames = Dict("x" => "n", "y" => "p")
θnames = [:n, :p]


# initial guess for optimisation
xy_initial = [50, 0.3]# x (i.e. n) and y (i.e. p), starting guesses
# parameter bounds
xy_lower_bounds = [0.0001, 0.0001]
xy_upper_bounds = [500.0, 1.0]
# true parameter
xy_true = [200.0, 0.2] #x,y, truth. N, p
N_samples = 10 # measurements of model
# generate data
#data = rand(distrib_xy(xy_true),N_samples)
data = (samples=SA[21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4],)

par_magnitudes = [100, 1]

# ---- use above to construct log likelihood in original parameterisation given (iid) data
function loglhood(xy, data)
    return sum(logpdf.(distrib_xy(xy), data.samples))
end

model = initialise_LikelihoodModel(loglhood, data, θnames, xy_initial, xy_lower_bounds, xy_upper_bounds, par_magnitudes);

bivariate_confidenceprofiles!(model, 1000, method=RadialRandomMethod(3))
true_boundary = model.biv_profiles_dict[1].confidence_boundary
LikelihoodBasedProfileWiseAnalysis.minimum_perimeter_polygon!(true_boundary)
true_boundary = hcat(true_boundary, true_boundary[:, 1])

model = initialise_LikelihoodModel(loglhood, data, θnames, xy_initial, xy_lower_bounds, xy_upper_bounds, par_magnitudes);
bivariate_confidenceprofiles!(model, 6, confidence_level=0.2, method=AnalyticalEllipseMethod(0.0, 1.0))
bivariate_confidenceprofiles!(model, 6, method=RadialMLEMethod(0.0, 1.0))

using Plots; gr()
using Plots.PlotMeasures

format = (size=(400, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    xlims=(10.0, 100),
    # ylims=(0.1,0.8),
    xlabel="n",
    ylabel="p",
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
savefig(plt, joinpath(output_location, "statmodel.pdf"))