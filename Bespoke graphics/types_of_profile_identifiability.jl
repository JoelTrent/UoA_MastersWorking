using Plots; gr()
using LaTeXStrings
using Interpolations

llstar = -1.92

format = (size=(400, 400)./1.4, dpi=300, title="", legend_position=:topright, 
    ylims=[llstar + llstar*0.2, 0.1], xlims=(0.4,2.1), xticks=0.5:0.5:2.0,
    palette=:Paired, 
    xlabel=L"\psi", ylabel=L"\hat{\ell_p}", legend=false)

plt1 = plot(LinRange(0.4, 2.1, 19), fill(0, 19);
    label=L"\hat{\ell}_p (\psi \, ; \, y_{1:I}^{\textrm{o}})", color=3, lw=3, format...)
hline!(plt1, [llstar], color=6, lw=3, linestyle=:dash, label=L"\ell_c")


spline = cubic_spline_interpolation(LinRange(0.4, 2.1, 19), 
    [-3.7, -2.6, -1.5, -0.9, -0.5, -0.3, -0.2, -0.14, -0.08, -0.03, 0.0, 
        -0.01, -0.033, -0.06, -0.08, -0.1, -0.1, -0.1, -0.1])

plt2 = plot(LinRange(0.4,2.1, 100), spline(LinRange(0.4,2.1, 100));
    label=L"\hat{\ell}_p (\psi \, ; \, y_{1:I}^{\textrm{o}})", color=3, lw=3, format...)
vline!(plt2, [1.344], color=5, lw=3, linestyle=:dash, label="MLE point")
hline!(plt2, [llstar], color=6, lw=3, linestyle=:dash, label=L"\ell_c")

f(x) = -6(x-1.2)^2 + ifelse(x≤ 1.2, 0, 2(x-1.2)^2)
plt3 = plot(LinRange(0.4, 2.1, 100), f.(LinRange(0.4, 2.1, 100),);
    label=L"\hat{\ell}_p (\psi \, ; \, y_{1:I}^{\textrm{o}})", color=3, lw=3, format..., legend=true)
vline!(plt3, [1.2], color=5, lw=3, linestyle=:dash, label="MLE point")
hline!(plt3, [llstar], color=6, lw=3, linestyle=:dash, label=L"\ell_c")


output_location = joinpath("Bespoke graphics", "profile_identifiability");
savefig(plt1, joinpath(output_location, "struct_nonidentifiable"))
savefig(plt2, joinpath(output_location, "pract_nonidentifiable"))
savefig(plt3, joinpath(output_location, "identifiable"))