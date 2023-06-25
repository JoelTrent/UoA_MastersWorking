# Section 1: set up packages and parameter values
# using BenchmarkTools

using Distributed
using PlaceholderLikelihood
if nprocs()==1; addprocs(10) end
# @everywhere using Revise
@everywhere using DifferentialEquations, Random, Distributions
@everywhere using PlaceholderLikelihood

# Workflow functions ##########################################################################

@everywhere function solvedmodel(t, a)
    return (a[2]*a[3]) ./ ((a[2]-a[3]) .* (exp.(-a[1] .* t)) .+ a[3])
end

# Section 6: Define loglikelihood function
@everywhere function loglhood(a, data)
    # y=ODEmodel(data.t, a)
    y=solvedmodel(data.t, a)
    e=0
    e=sum(loglikelihood(data.dist, data.yobs .- y))
    return e
end

# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
@everywhere function funmle(a)
    return loglhood(a, data)
end

# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]
θtrue=[λ, K, C0]

# true data
ytrue = solvedmodel(t, a)

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

@everywhere likelihoodFunc = loglhood
θnames = [:λ, :K, :C0]

model = initialiseLikelihoodModel(likelihoodFunc, data, θnames, θG, lb, ub, par_magnitudes);

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    yobs = generator_args.ytrue .+ rand(generator_args.dist, length(generator_args.t))
    data = (yobs=yobs, generator_args...)
    return data
end
gen_args = (ytrue=ytrue, σ=σ, t=t, dist=Normal(0, σ))

# PARAMETER COVERAGE CHECKS
uni_coverage_df = check_univariate_parameter_coverage(data_generator, gen_args, model, 100, θtrue, collect(1:3), show_progress=true)
println(uni_coverage_df)

biv_coverage_df = check_bivariate_parameter_coverage(data_generator, gen_args, model, 500, 100, θtrue, [[1, 2], [1, 3], [2, 3]], show_progress=true, distributed_over_parameters=true)
println(biv_coverage_df)
# rmprocs(workers())