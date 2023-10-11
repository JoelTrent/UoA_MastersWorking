using Revise
using PlaceholderLikelihood
using EllipseSampling

using Distributions

a, b = 2.0, 1.0
α = 0.25 * π

Hw11 = (cos(α)^2 / a^2 + sin(α)^2 / b^2)
Hw22 = (sin(α)^2 / a^2 + cos(α)^2 / b^2)
Hw12 = cos(α) * sin(α) * (1 / a^2 - 1 / b^2)
Hw_norm = [Hw11 Hw12; Hw12 Hw22]

confidence_level = 0.95
Hw = Hw_norm ./ (0.5 ./ (Distributions.quantile(Distributions.Chisq(2), confidence_level) * 0.5))
Γ = convert.(Float64, inv(BigFloat.(Hw, precision=64)))
EllipseSampling.calculate_ellipse_parameters(Γ, 1, 2, 0.95)

true_boundary = generate_N_equally_spaced_points(500, Γ, [0.0, 0.0], 1, 2, confidence_level=0.95, start_point_shift=0.0)
true_boundary = hcat(true_boundary, true_boundary[:,1])

loglhood(θ, data) = PlaceholderLikelihood.ellipse_loglike(θ, (θmle=[0.,0.], Hmle=Hw))

model = initialise_LikelihoodModel(loglhood, (1,), [:θ1, :θ2], [0.0, 0.0], [-10.0, -10.0], [10.0, 10.0], [1.0, 1.0]);
bivariate_confidenceprofiles!(model, 6, profile_type=LogLikelihood(), method=IterativeBoundaryMethod(3,1,1, 0.15, 1.0, use_ellipse=true))

using Plots; gr()

format = (size=(400, 400), dpi=300, xlabel=:θ1, ylabel=:θ2, title="", aspect_ratio=:equal, legend_position=:bottomright, palette=:Paired_7)

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
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5);

candidate_pnt = (pnts[:, 1] .+ pnts[:, 3]) ./ 2
plot!([candidate_pnt[1], pnts[1,5]], [candidate_pnt[2], pnts[2,5]], arrow=true, color=:black, linewidth=2, label="");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)

plot!(pnts[1, [3, 5, 1]], pnts[2, [3, 5, 1]], label=nothing, linestyle=:dash, color=2, opacity=0.75)
scatter!([pnts[1, 5]], [pnts[2, 5]], label="New point", markershape=:diamond, color=4, msw=0, ms=5)

###################### PLOT 3 #########################################
plt3 = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts_start = pnts[:, [1, 4, 2, 3, 5, 1]]
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5);

candidate_pnt = (pnts[:, 2] .+ pnts[:, 3]) ./ 2
plot!([candidate_pnt[1], pnts[1, 6]], [candidate_pnt[2], pnts[2, 6]], arrow=true, color=:black, linewidth=2, label="");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)

plot!(pnts[1, [2, 6, 3]], pnts[2, [2, 6, 3]], label=nothing, linestyle=:dash, color=2, opacity=0.75)
scatter!([pnts[1, 6]], [pnts[2, 6]], label="New point", markershape=:diamond, color=4, msw=0, ms=5)

###################### PLOT 4 #########################################
plt4 = plot(true_boundary[1, :], true_boundary[2, :]; label="True boundary", format...);
pnts = model.biv_profiles_dict[1].confidence_boundary
pnts_start = pnts[:, [1, 4, 2, 6, 3, 5, 1]]
plot!(pnts_start[1, :], pnts_start[2, :], label="Boundary polygon", marker=(:circle), msw=0, ms=5)


output_location=joinpath("Bespoke graphics", "iterativeboundaryupdates", "convex")
for (i, plt) in enumerate((plt1, plt2, plt3, plt4))
    if i!=1; plot!(plt, legend_position=nothing) end
    savefig(plt, joinpath(output_location, "update"*string(i)*".pdf"))
end