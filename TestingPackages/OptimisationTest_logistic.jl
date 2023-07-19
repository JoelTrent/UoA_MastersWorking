# Define the problem to solve
using Optimization, ForwardDiff, Zygote, FiniteDiff
using Distributions, Random

# rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
# x0 = zeros(2)
# _p = [1.0, 100.0]
global count = 0

@inline function solvedmodel(_t, θ)
    return (θ[2] * θ[3]) ./ ((θ[2] - θ[3]) .* (exp.(-θ[1] .* _t)) .+ θ[3])
end

function loglhood(θ, p)
    global count+=1
    yobs, t, dist = p
    return -sum(loglikelihood(dist, yobs .- solvedmodel(t, θ)))
end

@inline function ll_l2norm(θ, p)
    yobs, _t = p
    y = solvedmodel(_t, θ)

    return sum((yobs .- y).^2.0)
end

λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
_θ=[λ, K, C0]

λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50.0, 150.0)
C0min, C0max = (0.00, 50.0)

θG = [λ, K, C0]
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]

ytrue = solvedmodel(t, _θ)
Random.seed!(12348)
yobs = ytrue + σ * randn(length(t))
# _data = (yobs=yobs, σ=σ, t=t, dist=Normal(0, σ))
_data = (yobs, t, Normal(0, σ))

loglhood(θG, _data)
_p = (yobs, t)
ll_l2norm(θG, _p)

fopt = OptimizationFunction(ll_l2norm, AutoForwardDiff())
prob = OptimizationProblem(fopt, θG, _data, lb=lb, ub=ub)

## Optim.jl Solvers

using OptimizationOptimJL

# Start with some derivative-free optimizers

sol = solve(prob, SAMIN())

sol = solve(prob, NelderMead())

# Now a gradient-based optimizer with forward-mode automatic differentiation

sol = solve(prob, LBFGS())


using OptimizationNLopt
optf = OptimizationFunction(loglhood, AutoForwardDiff())
prob = OptimizationProblem(optf, θG, _data, lb=lb, ub=ub)
sol = solve(prob, NLopt.LD_LBFGS, xtol_rel=1e-9)

fixedtheta1(x, p) = loglhood([0.11, x[1], x[2]], p)
optf = OptimizationFunction(fixedtheta1, AutoForwardDiff())
prob = OptimizationProblem(optf, θG[2:3], _data, lb=lb[2:3], ub=ub[2:3])
sol = solve(prob, NLopt.LD_LBFGS())
sol = solve(prob, NLopt.LN_BOBYQA)


using NLopt

function optimise(fun, θ₀, lb, ub;
    dv=false,
    method=dv ? :LD_LBFGS : :LN_BOBYQA
)

    if dv || String(method)[2] == 'D'
        tomax = fun
    else
        tomax = (θ, ∂θ) -> fun(θ)
    end

    opt = Opt(method, length(θ₀))
    opt.max_objective = tomax
    opt.lower_bounds = lb       # Lower bound
    opt.upper_bounds = ub       # Upper bound
    # opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
    # opt.xtol_rel = 1e-9
    # opt.maxeval=4000
    # opt.maxtime = 15
    res = NLopt.optimize(opt, θ₀)
    display(opt.numevals)
    return res[[2, 1]]
end

f(x) = -loglhood(x, _data)
f(x) = -ll_l2norm(x, _p)
optimise(f, θG, lb, ub)
f(x) = -fixedtheta1(x, _data)
optimise(f, θG[2:3], lb[2:3], ub[2:3])