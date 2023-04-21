# Section 1: set up packages and parameter values
# using Plots
using DifferentialEquations
# using .Threads 
using Interpolations, Random, Distributions
using BenchmarkTools
# gr()

fileDirectory = joinpath("Workflow paper examples", "Logistic Model Num Opt", "Plots")
include(joinpath("..", "..", "JuLikelihood.jl"))

using Distributed
# addprocs(3)
@everywhere using DifferentialEquations, Random, Distributions
@everywhere include(joinpath("..", "..", "JuLikelihood.jl"))

# Workflow functions ##########################################################################

# Section 2: Define ODE model
@everywhere function DE!(dC, C, p, t)
    λ,K=p
    dC[1]= λ * C[1] * (1.0 - C[1]/K)
end

# Section 3: Solve ODE model
@everywhere function odesolver(t, λ, K, C0)
    p=(λ,K)
    tspan=(0.0, t[end])
    prob=ODEProblem(DE!, [C0], tspan, p)
    sol=solve(prob, saveat=t, verbose=false)
    return sol[1,:]
end

# Section 4: Define function to solve ODE model 
@everywhere function ODEmodel(t, a)
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

# Section 6: Define loglikelihood function
@everywhere function loglhood(a, data)
    y=ODEmodel(data.t, a)
    e=0
    e=sum(loglikelihood(data.dist, data.yobs-y))
    return e
end


# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
@everywhere function funmle(a); return loglhood(a, data) end 

# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]

# true data
ytrue = ODEmodel(t, a)

Random.seed!(12348)
# noisy data
yobs = ytrue + σ*randn(length(t))

# Named tuple of all data required within the likelihood function
data = (yobs=yobs, σ=σ, t=t, dist=Normal(0, σ))

# Bounds on model parameters #################################################################
λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50., 150.)
C0min, C0max = (0.0, 50.)

θG = [λ, K, C0]
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]

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

@everywhere likelihoodFunc = loglhood
@everywhere function predictFunc(θ, data); ODEmodel(data.t, θ) end
θnames = [:λ, :K, :C0]
confLevel = 0.95

# initialisation. model is a mutable struct that is currently intended to hold all model information
model = initialiseLikelihoodModel(likelihoodFunc, predictFunc, data, θnames, θG, lb, ub, uni_row_prealloaction_size=3)

# not strictly required - functions that rely on these being computed will check if 
# they're missing and call this function on model if so.
getMLE_ellipse_approximation!(model)

# @time univariate_confidenceintervals!(model, confidence_level=0.95, profile_type=EllipseApproxAnalytical())
# @time univariate_confidenceintervals!(model, profile_type=EllipseApprox())
@time univariate_confidenceintervals!(model, profile_type=LogLikelihood())

# univariate_confidenceintervals!(model, [1], profile_type=EllipseApproxAnalytical())
# univariate_confidenceintervals!(model, [:K], profile_type=EllipseApprox())
# univariate_confidenceintervals!(model, [1,2,3], confidence_level=0.7, use_existing_profiles=true, use_distributed=true)
# univariate_confidenceintervals!(model, 2, profile_type=EllipseApprox())


# @time bivariate_confidenceprofiles!(model, 100, confidence_level=0.1, profile_type=EllipseApproxAnalytical(), method=BracketingMethodFix1Axis())
# @time bivariate_confidenceprofiles!(model, 100, confidence_level=0.1, profile_type=EllipseApproxAnalytical(), method=BracketingMethodRadial(5))
# @time bivariate_confidenceprofiles!(model, 100, confidence_level=0.1, profile_type=EllipseApproxAnalytical(), method=BracketingMethodSimultaneous())
# @time bivariate_confidenceprofiles!(model, 100, confidence_level=0.3, profile_type=EllipseApprox(), method=ContinuationMethod(0.1, 5, 0.3))
# @time bivariate_confidenceprofiles!(model, 100, profile_type=LogLikelihood(), method=BracketingMethodRadial(5))
# @time bivariate_confidenceprofiles!(model, 100, profile_type=LogLikelihood(), method=ContinuationMethod(0.1, 2, 0.95))


@time bivariate_confidenceprofiles!(model, 100, confidence_level=0.9, method=AnalyticalEllipseMethod())


get_points_in_interval!(model, 50, additional_width=0.3)
generate_predictions_univariate!(model, 1.0, use_distributed=true)
generate_predictions_bivariate!(model, 0.1, use_distributed=false)

# model.core.θmle .= θG

using Plots
gr()
plots = plot_univariate_profiles(model, 0.2, 0.2)
for i in eachindex(plots); display(plots[i]) end

plots = plot_bivariate_profiles(model, 0.2, 0.2)
for i in eachindex(plots); display(plots[i]) end


println()



# Combination profiles
# confIntsRel, pRel = univariateprofile_providedrelationship(AMinusB(:K, :C0, 0.0, 150.0, :KMinusC0), likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)
# println(confIntsRel)

# confIntsRel_ellipse, pRel_ellipse = univariateprofile_ellipse_providedrelationship(AMinusB(:K, :C0, 0.0, 150.0, :KMinusC0), θnames, θmle, lb, ub, H, Γ; confLevel=0.95)
# println(confIntsRel_ellipse)


# confIntsRel, pRel = univariateprofile_providedrelationship(APlusB(:K, :C0, 50.0, 200.0, :KPlusC0), likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)
# println(confIntsRel)
# confIntsRel, pRel = univariateprofile_providedrelationship(ADivB(:K, :C0, 1.0, 200.0, :KDivC0), likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)
# println(confIntsRel)
# confIntsRel, pRel = univariateprofile_providedrelationship(ATimesB(:K, :C0, 10.0, 7500.0, :KTimesC0), likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)
# println(confIntsRel)

# confIntsBivariate, pBivariate = bivariateprofiles(likelihoodFunc, fmle, data, θnames, θmle, lb, ub, 50; confLevel=0.95, method=(:Brent, :fix1axis))

# confIntsBivariate, pBivariate = bivariateprofiles(likelihoodFunc, fmle, data, θnames, θmle, lb, ub, 50; confLevel=0.95, method=(:Brent, :vectorsearch))

# function likelihoodFuncLog(data, θ); return loglhood(data, exp.(θ)) end

# λmin, λmax = (0.001, 0.05)
# Kmin, Kmax = (50., 150.)
# C0min, C0max = (0.01, 50.)

# θG = [λ, K, C0]
# lb = [λmin, Kmin, C0min]
# ub = [λmax, Kmax, C0max]

# confIntsLog, pLog = univariateprofiles(likelihoodFuncLog, fmle, data, θnames, log.(θmle), log.(lb), log.(ub); confLevel=0.95)

# function funmleLog(a); likelihoodFuncLog(data,a) end
# H, Γ = getMLE_hessian_and_covariance(funmleLog, log.(θmle))
# confInts_ellipseLog, p_ellipseLog = univariateprofiles_ellipse(θnames, log.(θmle), log.(lb), log.(ub), H, Γ; confLevel=0.95)




# # complementary rearrangement to obtain K from, θ[2] (K-C0) and θ[3] (C0)
# function likelihoodFunctionKMinusC0!(data, θ); 
#     # θnew = zeros(length(θ))
#     # θnew[1] = θ[1]
#     # θnew[2] = θ[2] + θ[3]
#     # θnew[3] = θ[3]
#     θ[2] = θ[2]+θ[3]

#     loglhood(data, θ) 
# end

# lb[2] = lb[2]-ub[3]
# ub[2] = ub[2]-lb[3]
# θmle[2] = θmle[2]-θmle[3]
# θnames[2] = :KMinusC0

# confIntsKMinusC0, pKMinusC0 = univariateprofiles(likelihoodFunctionKMinusC0!, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)

function forward_parameter_transformLog(θ)
    return log.(θ)
end


newlb, newub = transformbounds(forward_parameter_transformLog, lb, ub, collect(1:3), Int[])
exp.(newlb)
exp.(newub)

λmin, λmax = (0.001, 0.05)
Kmin, Kmax = (50., 150.)
C0min, C0max = (0.01, 50.)

θG = [λ, K, C0]
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]

function forward_parameter_transformKminusC0(θ)
    Θ=zeros(length(θ))

    Θ .= θ

    Θ[2] = θ[2]-θ[3]

    return Θ
end


newlb, newub = transformbounds(forward_parameter_transformKminusC0, lb, ub, Int[1,3], Int[2])
newθmle = forward_parameter_transformKminusC0(θmle)


# REMOVE WORKER PROCESSORS WHEN FINISHED #######################################
rmprocs(workers())
################################################################################
