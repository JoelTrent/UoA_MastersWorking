# Section 1: set up packages and parameter values
# using BenchmarkTools
using Revise
using Distributed
using PlaceholderLikelihood
using Surrogates
# if nprocs() == 1
#     addprocs(10)
# end
# @everywhere using Revise
@everywhere using DifferentialEquations, Random, Distributions, StaticArrays
@everywhere using PlaceholderLikelihood

# Workflow functions ##########################################################################

@everywhere function solvedmodel(t, a)
    return (a[2] * a[3]) ./ ((a[2] - a[3]) .* (exp.(-a[1] .* t)) .+ a[3])
end

# Section 6: Define loglikelihood function
@everywhere function loglhood(a, data)
    # y=ODEmodel(data.t, a)
    y = solvedmodel(data.t, a)
    e = 0
    e = sum(loglikelihood(data.dist, data.yobs .- y))
    return e
end

# Data setup #################################################################################
# true parameters
λ = 0.01;
K = 100.0;
C0 = 10.0;
t = 0:100:1000;
σ = 10.0;
tt = 0:5:1000
a = [λ, K, C0]
θtrue = [λ, K, C0]

# true data
ytrue = solvedmodel(t, a)

Random.seed!(12348)
# noisy data
yobs = ytrue + σ * randn(length(t))
yobs = ytrue .+ rand(Normal(0, σ), length(t))

# Named tuple of all data required within the likelihood function
data = (yobs=yobs, σ=σ, t=t, dist=Normal(0, σ))

# Bounds on model parameters #################################################################
λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50.0, 150.0)
C0min, C0max = (0.0, 50.0)

θG = [λ, K, C0]
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]
par_magnitudes = [0.005, 10, 10]

@everywhere likelihoodFunc = loglhood
θnames = [:λ, :K, :C0]

model = initialiseLikelihoodModel(likelihoodFunc, data, θnames, θG, lb, ub, par_magnitudes);
univariate_confidenceintervals!(model)
get_uni_confidence_interval(model, 1)
get_uni_confidence_interval(model, 2)
get_uni_confidence_interval(model, 3)

f = a -> loglhood(a, data)
x = Surrogates.sample(1000, lb, ub, SobolSample())
y = f.(x)

@everywhere radial_basis = RadialBasis(x, y, lb_sur, ub_sur)

@everywhere function loglhood_surrogate(a, data)
    return radial_basis(a)
end

model_surrogate = initialiseLikelihoodModel(loglhood_surrogate, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model_surrogate)
get_uni_confidence_interval(model_surrogate, 1)
get_uni_confidence_interval(model_surrogate, 2)
get_uni_confidence_interval(model_surrogate, 3)


@everywhere kriging_basis = Kriging(x, y, lb_sur, ub_sur)
@everywhere function loglhood_surrogate_krig(a, data)
    return kriging_basis(a)
end

model_surrogate_krig = initialiseLikelihoodModel(loglhood_surrogate_krig, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model_surrogate_krig)
get_uni_confidence_interval(model_surrogate_krig, 1)
get_uni_confidence_interval(model_surrogate_krig, 2)
get_uni_confidence_interval(model_surrogate_krig, 3)