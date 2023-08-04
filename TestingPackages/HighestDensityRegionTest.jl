using HighestDensityRegions, KernelDensity
using Random, Distributions
using Plots; gr()

# NORMAL
dist = Normal(0,1)
xgrid = range(quantile(dist, [0.0001, 0.9990])..., length=100000)
ygrid = pdf.(dist, xgrid)
threshold = first(hdr_thresholds([0.95], ygrid))

hdr = [xgrid[findfirst(ygrid .> threshold)], xgrid[findlast(ygrid .> threshold)]]
quants = quantile(dist, [0.025, 0.975])

plot(xgrid, ygrid, label="Normal(0,1)", dpi=300)
vline!(hdr, label="Highest density region (95%)")
vline!(quants, label="Quantile (0.025, 0.975)")
xlims!(0, 10)

using Roots, NLopt
f(x, density_threshold=0.95) = -abs(x[2] - x[1]) - 1000 * abs(density_threshold - (cdf(dist, x[2]) - cdf(dist, x[1])))

# Section 7: Numerical optimisation 
function optimise(fun, θ₀;
    method=:LN_BOBYQA
)
    tomax = (θ, ∂θ) -> fun(θ)

    opt = Opt(method, length(θ₀))
    opt.max_objective = tomax
    opt.maxeval = 6000
    res = optimize(opt, θ₀)
    return res[[2, 1]]
end

optimise(f, [-1.0, 1.0])

# LOG-NORMAL
dist = LogNormal(0,1)
xgrid = range(quantile(dist, [0.0, 0.999])..., length=100000)
ygrid = pdf.(dist, xgrid)
threshold = first(hdr_thresholds([0.95], ygrid))

hdr = [xgrid[findfirst(ygrid .> threshold)], xgrid[findlast(ygrid .> threshold)]]
quants = quantile(dist, [0.025, 0.975])

plot(xgrid, ygrid, label="LogNormal(0,1)", dpi=300)
vline!(hdr, label="Highest density region (95%)")
vline!(quants, label="Quantile (0.025, 0.975)")
xlims!(0,10)
savefig("TestingPackages/highest_density_region_test.png")

optimise(f, [0.0, 10.0])