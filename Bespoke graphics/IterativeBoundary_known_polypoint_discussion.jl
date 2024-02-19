using Plots; gr()
using Plots.PlotMeasures
using EllipseSampling

format = (size=(500, 400)./1.25, dpi=300, xlabel="x", ylabel="y", title="", aspect_ratio=:equal,
    legend_position=:bottomright, palette=:Paired_7, leftmargin=3mm, bottommargin=3mm,
     xticks=-3:10, yticks=-3:5, xlims=(-2, 7), ylims=(-3,4.3))

e1 = construct_ellipse(3., 1., 0.6π, 0., 0.)
e2 = construct_ellipse(1., 1., 0., 5., 3.)

n = 100
pnts1 = generate_N_equally_spaced_points(n, e1)
pnts2 = generate_N_equally_spaced_points(n, e2)
pnts1 = pnts1[:, vcat(collect(1:n), [1])]
pnts2 = pnts2[:, vcat(collect(1:n), [1])]

poly1 = reduce(hcat, generate_perimeter_point.([0.1, 0.3, 0.6], Ref(e1)))
poly2 = reduce(hcat, generate_perimeter_point.([0.6], Ref(e2)))
poly = hcat(poly1, poly2)
poly = poly[:, vcat(collect(1:4), [1])]

plt = plot(pnts1[1, :], pnts1[2, :]; label="True boundaries", color=1, format...)
plot!(pnts2[1, :], pnts2[2, :], label="", color=1)
plot!(poly[1, :], poly[2, :], marker=(:circle), label="Boundary polygon", msw=0, ms=5, color=2)

candidate_pnt = (poly[:, 1] .+ poly[:, 4]) ./ 2
normal = [(poly[2,4] - poly[2,1]), -(poly[1,4] - poly[1,1])]
new_pnt = candidate_pnt .+ (normal./3)

scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="Candidate point", markershape=:diamond, color=3, msw=0, ms=5)
plot!([candidate_pnt[1], new_pnt[1]], [candidate_pnt[2], new_pnt[2]], arrow=true, color=5, linewidth=2, label="Normal direction");
plot!([candidate_pnt[1], poly[1,3]], [candidate_pnt[2], poly[2,3]], arrow=true, color=:black, linewidth=2, label="Adjusted direction");
scatter!([candidate_pnt[1]], [candidate_pnt[2]], label="", markershape=:diamond, color=3, msw=0, ms=5)

scatter!([0,5], [0,3], msw=0, ms=5, label="MLE points", color=6)
display(plt)
# plt.subplots[1].series_list = plt.subplots[1].series_list[[1, 2, 3, 6, 4, 5, 7]]

output_location = joinpath("Bespoke graphics", "iterativeboundary_polypoint")
savefig(plt, joinpath(output_location, "internal_search_direction.pdf"))