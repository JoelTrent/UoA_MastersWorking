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
xy_true = [100.0, 0.2] #x,y, truth. N, p
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

bivariate_confidenceprofiles!(model, 500, method=RadialRandomMethod(3), use_distributed=false)
true_boundary = model.biv_profiles_dict[1].confidence_boundary
LikelihoodBasedProfileWiseAnalysis.minimum_perimeter_polygon!(true_boundary)
true_boundary = hcat(true_boundary, true_boundary[:,1])

model = initialise_LikelihoodModel(loglhood, data, θnames, xy_initial, xy_lower_bounds, xy_upper_bounds, par_magnitudes);

bivariate_confidenceprofiles!(model, 200, method=IterativeBoundaryMethod(3,1,1, 0.15, 1.0, use_ellipse=true), use_distributed=false)

using Plots; gr()

format = (size=(400, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    xlims=(20.0, 70),
    ylims=(0.2, 0.7),
    yticks=0.2:0.1:0.7,
    xlabel="n",
    ylabel="p",
    title="", 
    # aspect_ratio=:equal, 
    legend_position=:topright, palette=:Paired_7)

###################### PLOT 1 #########################################
plt1 = plot(true_boundary[1,:], true_boundary[2,:]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts_start = hcat(pnts[:, 1:3], pnts[:, 1])
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5);

candidate_pnt = (pnts[:,1] .+ pnts[:,2]) ./ 2
plot!([candidate_pnt[1], pnts[1, 4]], [candidate_pnt[2], pnts[2, 4]], arrow=true, color=:black, linewidth=2, label="");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)

plot!(pnts[1,[1,4,2]], pnts[2,[1,4,2]], label=nothing, linestyle=:dash, color=2, opacity=0.75)
scatter!([pnts[1, 4]], [pnts[2, 4]], label="New point", markershape=:diamond, color=4, msw=0, ms=5)


###################### PLOT 2 #########################################
plt2 = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts_start = pnts[:, [1,4,2,3,1]]
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5)

candidate_pnt = (pnts[:, 1] .+ pnts[:, 3]) ./ 2
plot!([candidate_pnt[1], pnts[1,5]], [candidate_pnt[2], pnts[2,5]], arrow=true, color=:black, linewidth=2, label="");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)

plot!(pnts[1, [3, 5, 1]], pnts[2, [3, 5, 1]], label=nothing, linestyle=:dash, color=2, opacity=0.75)
scatter!([pnts[1, 5]], [pnts[2, 5]], label="New point", markershape=:diamond, color=4, msw=0, ms=5)

###################### PLOT 3 #########################################
plt3 = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts_start = pnts[:, [1, 4, 2, 3, 5, 1]]
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5)

candidate_pnt = (pnts[:, 1] .+ pnts[:, 4]) ./ 2
plot!([candidate_pnt[1], pnts[1, 6]], [candidate_pnt[2], pnts[2, 6]], arrow=true, color=:black, linewidth=2, label="");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)

plot!(pnts[1, [1, 6, 4]], pnts[2, [1, 6, 4]], label=nothing, linestyle=:dash, color=2, opacity=0.75)
scatter!([pnts[1, 6]], [pnts[2, 6]], label="New point", markershape=:diamond, color=4, msw=0, ms=5)

###################### PLOT 4 #########################################
plt4 = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts_start = pnts[:, [1, 6, 4, 2, 3, 5, 1]]
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5)

candidate_pnt = (pnts[:, 4] .+ pnts[:,2]) ./ 2
plot!([candidate_pnt[1], pnts[1, 7]], [candidate_pnt[2], pnts[2, 7]], arrow=true, color=:black, linewidth=2, label="");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)
plot!(pnts[1, [2, 7, 4]], pnts[2, [2, 7, 4]], label=nothing, linestyle=:dash, color=2, opacity=0.75)
scatter!([pnts[1, 7]], [pnts[2, 7]], label="New point", markershape=:diamond, color=4, msw=0, ms=5)

###################### PLOT 5 #########################################
plt5 = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts_start = pnts[:, [1, 6, 4, 7, 2, 3, 5, 1]]
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5)

candidate_pnt = (pnts[:, 1] .+ pnts[:, 5]) ./ 2
plot!([candidate_pnt[1], pnts[1, 8]], [candidate_pnt[2], pnts[2, 8]], arrow=true, color=:black, linewidth=2, label="");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)
plot!(pnts[1, [1, 8, 5]], pnts[2, [1, 8, 5]], label=nothing, linestyle=:dash, color=2, opacity=0.75)
scatter!([pnts[1, 8]], [pnts[2, 8]], label="New point", markershape=:diamond, color=4, msw=0, ms=5)

###################### PLOT 6 #########################################
plt6 = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts_start = pnts[:, [1, 6, 4, 7, 2, 3, 5, 8, 1]]
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5)

candidate_pnt = (pnts[:, 5] .+ pnts[:, 3]) ./ 2
plot!([candidate_pnt[1], pnts[1, 9]], [candidate_pnt[2], pnts[2, 9]], arrow=true, color=:black, linewidth=2, label="");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)
plot!(pnts[1, [3, 9, 5]], pnts[2, [3, 9, 5]], label=nothing, linestyle=:dash, color=2, opacity=0.75)
scatter!([pnts[1, 9]], [pnts[2, 9]], label="New point", markershape=:diamond, color=4, msw=0, ms=5)


output_location=joinpath("Bespoke graphics", "iterativeboundaryupdates", "concave")
for (i, plt) in enumerate((plt1, plt2, plt3, plt4, plt5, plt6))
    if i!=1; plot!(plt, legend_position=nothing) end
    savefig(plt, joinpath(output_location, "update"*string(i)*".pdf"))
end

format = (size=(600, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    # xlims=(20.0, 100),
    title="",
    # aspect_ratio=:equal, 
    legend_position=:topright, palette=:Paired_7)

plt = plot_bivariate_profiles(model; markeralpha=0.5, format...)[1]
savefig(plt, joinpath(output_location, "200point_boundary.pdf"))

output_location = joinpath("Bespoke graphics", "iterativeboundaryupdates", "concave", "gif")
plot_bivariate_profiles_iterativeboundary_gif(model, 0.2, 0.2; markeralpha=0.5, save_folder=output_location, color=2, save_as_separate_plots=false, format...)

