using Random
using ProfileLikelihood
using Optimization
using OrdinaryDiffEq
using CairoMakie
using LaTeXStrings
using OptimizationOptimJL
using OptimizationNLopt
using StableRNGs

######################################################
## Example II: Logistic ODE
######################################################
## Step 1: Generate the data and define the likelihood
λ = 0.01
K = 100.0
u₀ = 10.0
t = 0:100:1000
σ = 10.0
@inline function ode_fnc(u, p, t)
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
# Initial data is obtained by solving the ODE 
tspan = extrema(t)
p = (λ, K)
prob = ODEProblem(ode_fnc, u₀, tspan, p)
sol = solve(prob, Rosenbrock23(), saveat=t)
rng = StableRNG(123)
uᵒ = sol.u + σ * randn(rng, length(t))
function loglik_fnc2(θ, data, integrator)
    λ, K, u₀ = θ
    uᵒ, σ = data
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u₀)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
end

## Step 2: Define the problem 
lb = [0.0, 50.0, 0.0] # λ, K, u₀
ub = [0.05, 150.0, 50.0]
θ₀ = [λ, K, u₀]
syms = [:λ, :K, :u₀]
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t); # Note that u₀ is just a placeholder IC in this case since we are estimating it
    syms=syms,
    data=(uᵒ, σ),
    ode_parameters=[1.0, 1.0], # temp values for [λ, K]
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)

## Step 3: Compute the MLE 
@time sol = mle(prob, NLopt.LD_LBFGS)

prof = profile(prob, sol; alg=NLopt.LN_NELDERMEAD, parallel=false)

using CairoMakie, LaTeXStrings
fig = plot_profiles(prof;
    latex_names=[L"\lambda", L"K", L"u_0"],
    show_mles=true,
    shade_ci=true,
    nrow=1,
    ncol=3,
    true_vals=[λ, K, u₀],
    fig_kwargs=(fontsize=41,),
    axis_kwargs=(width=600, height=300))
resize_to_layout!(fig)
display(fig)