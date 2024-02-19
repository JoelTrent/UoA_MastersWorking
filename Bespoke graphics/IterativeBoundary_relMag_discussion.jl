using Plots; gr()
using Plots.PlotMeasures

points = [0 10 10 0 0; 0 0 1 1 0]
tri = points[:,[1,2,3,1]]

format = (size=(800, 400), dpi=300, xlabel="x", ylabel="y", title="", aspect_ratio=:equal,
    legend_position=:bottomright, palette=:Paired_7, leftmargin=3mm, bottommargin=3mm, xticks=0:10)

plt = plot(points[1,:], points[2,:]; marker=(:circle), label="True Boundary", lw=4, opacity=1.0, mw=8, msw=0, format...)
plot!(tri[1,:], tri[2,:]; marker=(:utriangle), label="Current Boundary", linestyle=:dash, lw=2, opacity=1.0, mw=4, msw=0)

plot!([5, 4.95], [0.5, 1], arrow=true, linewidth=3, label="[-1, 10]")
plot!([5, 4.5], [0.5, 1], arrow=true, linewidth=3, label="[-1, 1]")
plot!([5, 0], [0.5, 1], arrow=true, linewidth=3, label="[-10, 1]")

output_location=joinpath("Bespoke graphics", "iterativeboundary_relmag")
savefig(plt, joinpath(output_location, "relativeMagnitude.pdf"))