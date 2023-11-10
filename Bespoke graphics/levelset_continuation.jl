using Revise
using CSV, DataFrames
using Random, Distributions
using LaTeXStrings

using EllipseSampling
e1 = construct_ellipse(1.0, 2.0, 0.5)
e2 = construct_ellipse(2.0, 4.0, 0.5)
true_boundary = generate_N_equally_spaced_points(200, e1)
true_boundary1 = hcat(true_boundary, true_boundary[:,1])
true_boundary = generate_N_equally_spaced_points(200, e2)
true_boundary2 = hcat(true_boundary, true_boundary[:,1])


using Plots; gr()
using Plots.PlotMeasures

format = (size=(400, 400), dpi=300, #xlabel=:θ1, ylabel=:θ2, 
    # xlims=(10.0, 100),
    # ylims=(0.1,0.8),
    xlabel=L"\theta_1",
    ylabel=L"\theta_2",
    title="",
    rightmargin=3mm,
    # aspect_ratio=:equal, 
    legend_position=:topright, palette=:Paired_7)


pnts = generate_N_equally_spaced_points(3, e1, start_point_shift=0.0)
pnts1 = pnts[:, [1:3..., 1]]

pnts = generate_N_equally_spaced_points(3, e2, start_point_shift=0.0)
pnts2 = pnts[:, [1:3..., 1]]


plt = plot(true_boundary1[1, :], true_boundary1[2, :]; label=L"\ell_{c,lower}"*" boundary", lw=2, format...);
plot!(true_boundary2[1, :], true_boundary2[2, :]; lw=2, label=L"\ell_{c}"*" boundary")


plot!(pnts1[1, :], pnts1[2, :], label=L"\ell_{c,lower}"*" polygon", linestyle=:dash, lw=2, marker=(:diamond), color=3, msw=0, ms=5)
plot!(pnts2[1, :], pnts2[2, :], label=L"\ell_{c}"*" polygon", marker=(:circle), lw=2, color=4, msw=0, ms=5)

scatter!([0], [0], label=L"\hat{\psi}", msw=0, ms=5, color=6)
for i in 1:3
    plot!([pnts1[1, i], pnts2[1, i]], [pnts1[2, i], pnts2[2, i]], arrow=true, color=:black, linewidth=2, label="")
end

output_location = joinpath("Bespoke graphics", "levelset_continuation");
savefig(plt, joinpath(output_location, "ellipse.pdf"))