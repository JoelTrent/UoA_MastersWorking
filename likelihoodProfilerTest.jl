using LikelihoodProfiler
using DifferentialEquations, Distributions
using NLopt, Roots


# testing profile function
f(x) = 5.0 + (x[1]-3.0)^2 + (x[1]-x[2]-1.0)^2 + 0*x[3]^2

# Calculate parameters intervals for first parameter component, x[1]
res_1 = get_interval(
  [3., 2., 2.1], # starting point
  1,             # parameter component to analyze
  f,             # profile function
  :LIN_EXTRAPOL; # method
  loss_crit = 9. # critical level of loss function
  )

using Plots
plotly()
plot(res_1)

Random.seed!(12348)
# Logistic model example
function DE!(dC, C, p, t)
    λ,K=p
    dC[1]= λ * C[1] * (1.0 - C[1]/K)
end

function odesolver(t, λ, K, C0)
    p=(λ,K)
    tspan=(0.0, maximum(t))
    prob=ODEProblem(DE!, [C0], tspan, p)
    sol=solve(prob, saveat=t)
    return sol[1,:]
end

function model(t, a::AbstractVector{<:Real})
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

# Section 6: Define loglikelihood function
function loglhood(data, a, σ)
  y=model(t, a)
  e=0
  dist=Normal(0, σ);
  e=loglikelihood(dist, data-y) 
  return sum(e)
end

# Section 7: Numerical optimisation 
function optimise(fun, θ₀, lb, ub;
  dv = false,
  method = dv ? :LD_LBFGS : :LN_BOBYQA,
  )

  if dv || String(method)[2] == 'D'
      tomax = fun
  else
      tomax = (θ,∂θ) -> fun(θ)
  end

  opt = Opt(method,length(θ₀))
  opt.max_objective = tomax
  opt.lower_bounds = lb       # Lower bound
  opt.upper_bounds = ub       # Upper bound
  opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
  res = optimize(opt, θ₀)
  return res[[2,1]]
end

function funmle(a); return loglhood(data, a, σ) end 

λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]

# true data
data0 = model(t, a)

# noisy data
data = data0 + σ*randn(length(t))

# Bounds on model parameters #################################################################
λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50, 150)
C0min, C0max = (0, 50)

θG = [λ, K, C0]
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]

##############################################################################################
# Section 9: Find MLE by numerical optimisation, visually compare data and MLE solution
# Use Nelder-Mead algorithm to estimate maximum likelihood solution for parameters given 
# noisy data
(xopt, fopt) = optimise(funmle, θG, lb, ub)
fmle=fopt
λmle, Kmle, C0mle = xopt

# 1D profile confidence intervals for all parameters
f(x) = -funmle(x)

α = f(xopt) + cquantile(Chisq(3), 0.05)/2

res = [get_interval(xopt, i, f, :CICO_ONE_PASS, loss_crit=α, scale = fill(:log,length(xopt))) for i in 1:3];

for i in 1:3; update_profile_points!(res[i]) end

plot(res[1])
plot(res[2])
plot(res[3])

[res[i].result[j].value for i in 1:3 for j in 1:2]