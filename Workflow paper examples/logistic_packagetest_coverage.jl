# Section 1: set up packages and parameter values
# using BenchmarkTools

# using Distributed
# addprocs(3)
# @everywhere using Revise
# using PlaceholderLikelihood
using DifferentialEquations, Random, Distributions
using PlaceholderLikelihood

# Workflow functions ##########################################################################

# Section 2: Define ODE model
function DE!(dC, C, p, t)
    λ,K=p
    dC[1]= λ * C[1] * (1.0 - C[1]/K)
end

# Section 3: Solve ODE model
function odesolver(t, λ, K, C0)
    p=(λ,K)
    tspan=(0.0, t[end])
    prob=ODEProblem(DE!, [C0], tspan, p)
    sol=solve(prob, saveat=t, verbose=false)
    return sol[1,:]
end

# Section 4: Define function to solve ODE model 
function ODEmodel(t, a)
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

function solvedmodel(t, a)
    return (a[2]*a[3]) ./ ((a[2]-a[3]) .* (exp.(-a[1] .* t)) .+ a[3])
end

# Section 6: Define loglikelihood function
function loglhood(a, data)
    # y=ODEmodel(data.t, a)
    y=solvedmodel(data.t, a)
    e=0
    e=sum(loglikelihood(data.dist, data.yobs .- y))
    return e
end

# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
function funmle(a); return loglhood(a, data) end 

# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]
θtrue=[λ, K, C0]

# true data
ytrue = ODEmodel(t, a)

Random.seed!(12348)
# noisy data
yobs = ytrue + σ*randn(length(t))
yobs = ytrue .+ rand(Normal(0, σ), length(t))

# Named tuple of all data required within the likelihood function
data = (yobs=yobs, σ=σ, t=t, dist=Normal(0, σ))

# Bounds on model parameters #################################################################
λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50., 150.)
C0min, C0max = (0.0, 50.)

θG = [λ, K, C0]
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]
par_magnitudes = [0.005, 10, 10]
# par_magnitudes = [1, 1, 1]

##############################################################################################
# Section 9: Find MLE by numerical optimisation, visually compare data and MLE solution
# Use Nelder-Mead algorithm to estimate maximum likelihood solution for parameters given 
# noisy data
(xopt, fopt) = optimise(funmle, θG, lb, ub)
fmle=fopt
λmle, Kmle, C0mle = xopt .* 1.0
θmle = [λmle, Kmle, C0mle]
ymle(t) = Kmle*C0mle/((Kmle-C0mle)*exp(-λmle*t)+C0mle) # full solution


##############################################################################################
# Section 12: Prediction interval from the univariate quadratic approximation of log-likelihood for parameter λ
# Compute and propogate uncertainty forward from the univariate likelihood for parameter λ

likelihoodFunc = loglhood
# REQUIRED TO SPECIFY WITH DEFAULT OF 3RD ARG DEPENDING ON DATA SO THAT YMLE IS FOUND
# @everywhere function predictFunc(θ, data, t=data.t); ODEmodel(t, θ) end
function predictFunc(θ, data, t=data.t); solvedmodel(t, θ) end
θnames = [:λ, :K, :C0]
confLevel = 0.95

# initialisation. model is a mutable struct that is currently intended to hold all model information
model = initialiseLikelihoodModel(likelihoodFunc, predictFunc, data, θnames, θG, lb, ub, par_magnitudes);

function data_generator(θtrue, generator_args::NamedTuple)
    yobs = generator_args.ytrue .+ rand(generator_args.dist, length(generator_args.t))
    data = (yobs=yobs, generator_args...)
    return data
end

gen_args = (ytrue=ytrue, σ=σ, t=t, dist=Normal(0, σ))

coverage_df = check_univariate_parameter_coverage(data_generator, gen_args, model, θtrue, 2000, collect(1:3))