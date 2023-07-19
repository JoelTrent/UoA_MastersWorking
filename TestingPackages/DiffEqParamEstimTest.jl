using DifferentialEquations, DiffEqParamEstim, Optimization, OptimizationBBO
f1 = function (du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -3.0 * u[2] + u[1] * u[2]
end
p = [1.5, 1.0]
u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
prob1 = ODEProblem(f1, u0, tspan, p)
sol = solve(prob1, Tsit5())

using RecursiveArrayTools # for VectorOfArray
t = collect(range(0, stop=10, length=200))
function generate_data(sol, t)
    randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
    data = convert(Array, randomized)
end
aggregate_data = convert(Array, VectorOfArray([generate_data(sol, t) for i in 1:1]))

using Distributions
distributions = [fit_mle(Normal, aggregate_data[i, j, :]) for i in 1:2, j in 1:200]

# fit_mle(Normal, aggregate_data[1,1,:])

# sol
# distributions = [Normal(sol[i][j], 0.01) for i in axes(sol, 2), j in 1:2] 


obj = build_loss_objective(prob1, Tsit5(), LogLikeLoss(t, distributions), AutoFiniteDiff(),
    maxiters=10000, verbose=false)

using Plots;
plotly();
prange = 0.5:0.1:5.0
heatmap(prange, prange, [obj([j, i]) for i in prange, j in prange],
    yscale=:log10, xlabel="Parameter 1", ylabel="Parameter 2",
    title="Likelihood Landscape")

plot(prange, [obj([1.5, i]) for i in prange], lw=3,
    title="Parameter 2 Likelihood (Parameter 1 = 1.5)",
    xlabel="Parameter 2", ylabel="Objective Function Value")

bound1 = Tuple{Float64,Float64}[(0.5, 5), (0.5, 5)]
optprob = OptimizationProblem(obj, [2.0, 2.0], lb=first.(bound1), ub=last.(bound1))

opt = Opt(:LD_LBFGS, 2)
res = solve(optprob, NLopt.LD_LBFGS)
