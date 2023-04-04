# Section 1: set up packages and parameter values
using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using BenchmarkTools
gr()

Random.seed!(12348)
fileDirectory = joinpath("Workflow paper examples", "Logistic Model Num Opt", "Plots")
include(joinpath("..", "plottingFunctions.jl"))
include(joinpath("..", "..", "JuLikelihood.jl"))

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
    sol=solve(prob, saveat=t)
    return sol[1,:]
end

# Section 4: Define function to solve ODE model 
function ODEmodel(t, a)
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

# Section 6: Define loglikelihood function
function loglhood(a, data)
    y=ODEmodel(data.t, a)
    return sum(loglikelihood(data.dist, data.yobs-y))
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

# true data
ytrue = ODEmodel(t, a)

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

# 3D approximation of the likelihood around the MLE solution
H, Γ = getMLE_hessian_and_covariance(funmle, θmle)
θmle
Γ
inv(Γ[[2,3], [2,3]])

# using LinearAlgebra
# sqrt.(1.0 ./ eigvals(inv(Γ[[2,3], [2,3]]) .* 0.5 ./ (quantile(Chisq(2), 0.95)*0.5)))

# eigs = eigvecs(inv(Γ[[2,3], [2,3]]) .* 0.5 ./ (quantile(Chisq(2), 0.95)*0.5))

# atan(eigs[2,1], eigs[1,1]) + pi/2


# using EllipseSampling
# EllipseSampling.calculate_ellipse_parameters(Γ, 2, 3, 0.95)


likelihoodFunc = loglhood
θnames = [:λ, :K, :C0]
confLevel = 0.95


# initialisation. model is a mutable struct that is currently intended to hold all model information
model = initialiseLikelihoodModel(likelihoodFunc, data, θnames, θG, lb, ub)

# not strictly required - functions that rely on these being computed will check if 
# they're missing and call this function on model if so.
getMLE_ellipse_approximation!(model)

analytic_ellipse_loglike_1D_soln(3, (θmle=model.core.θmle, Γmle=model.ellipse_MLE_approx.Γmle), -quantile(Chisq(1), 0.95)/2 )

univariate_confidenceintervals(model, profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals(model, profile_type=EllipseApprox())
univariate_confidenceintervals(model, profile_type=LogLikelihood())

univariate_confidenceintervals(model, [1], profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals(model, [:K], profile_type=EllipseApprox())
univariate_confidenceintervals(model, [1,2,3])
univariate_confidenceintervals(model, 2, profile_type=EllipseApprox())


@time bivariate_confidenceprofiles(model, 100, profile_type=EllipseApproxAnalytical())
@time bivariate_confidenceprofiles(model, 10, profile_type=EllipseApprox())
Random.seed!(12348)
@time bivariate_confidenceprofiles(model, 30, profile_type=LogLikelihood(), method=BracketingMethodFix1Axis())
Random.seed!(12348)
@time bivariate_confidenceprofiles(model, 10, profile_type=LogLikelihood(), method=BracketingMethodSimultaneous())


bivariate_confidenceprofiles(model, [[:K, :C0], [:λ, :K]],  10, profile_type=LogLikelihood(), method=BracketingMethodSimultaneous())
bivariate_confidenceprofiles(model, [(:K, :C0)],  10, profile_type=LogLikelihood(), method=BracketingMethodSimultaneous())
# @profview bivariate_confidenceprofiles(model, 2,  10, profile_type=LogLikelihood(), method=BracketingMethodSimultaneous())
# @profview bivariate_confidenceprofiles(model, 2,  10, profile_type=LogLikelihood(), method=BracketingMethodFix1Axis())

Random.seed!(12348)
@time bivariate_confidenceprofiles(model, 100, profile_type=LogLikelihood(), method=BracketingMethodFix1Axis())
Random.seed!(12348)
@time bivariate_confidenceprofiles(model, 100, profile_type=LogLikelihood(), method=BracketingMethodSimultaneous())
Random.seed!(12348)
@time bivariate_confidenceprofiles(model, 100, profile_type=LogLikelihood(), method=BracketingMethodRadial(8))


test = bivariate_confidenceprofiles(model, 10, profile_type=EllipseApproxAnalytical(), method=BracketingMethodFix1Axis())

test = bivariate_confidenceprofiles(model, 10, method=AnalyticalEllipseMethod())

test[(:K, :C0)].confidence_boundary_all_pars

a=10

# # 1D profiles
# confInts, p = univariateprofiles(likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)

# # 1D profile using elliptical approximation
# confInts_ellipse_analyt, p_ellipse_analyt = univariateprofiles_ellipse_analytical(θnames, θmle, lb, ub, H, Γ; confLevel=0.95)
# confInts_ellipse, p_ellipse = univariateprofiles_ellipse(θnames, θmle, lb, ub, H, Γ; confLevel=0.95)


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
