# Section 1: set up packages and parameter values
using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using Roots, NLopt
using ForwardDiff
gr()

Random.seed!(12348)
fileDirectory = joinpath("Workflow paper examples", "Logistic Model Num Opt", "Plots")
include(joinpath("..", "plottingFunctions.jl"))
include(joinpath("..", "..", "ellipseLikelihood.jl"))
include(joinpath("..", "..", "profileLikelihood.jl"))

# Workflow functions ##########################################################################

# Section 2: Define ODE model
function DE!(dC, C, p, t)
    λ,K=p
    dC[1]= λ * C[1] * (1.0 - C[1]/K)
end

# Section 3: Solve ODE model
function odesolver(t, λ, K, C0)
    p=(λ,K)
    tspan=(0.0, maximum(t))
    prob=ODEProblem(DE!, [C0], tspan, p)
    sol=solve(prob, saveat=t)
    return sol[1,:]
end

# Section 4: Define function to solve ODE model 
function ODEmodel(t, a, σ)
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

# Section 6: Define loglikelihood function
function loglhood(data, a)
    y=ODEmodel(t, a, data[2])
    e=0
    dist=Normal(0, data[2]);
    e=loglikelihood(dist, data[1]-y) 
    return sum(e)
end


# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
function funmle(a); return loglhood(data, a) end 

# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]

# true data
data0 = ODEmodel(t, a, σ)

# noisy data
data = (data0 + σ*randn(length(t)), σ)

# Bounds on model parameters #################################################################
λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50, 150)
C0min, C0max = (0.1, 50)

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

likelihoodFunc = loglhood
θnames = [:λ, :K, :C0]
confLevel = 0.95

# 1D profiles
confInts, p = univariateprofiles(likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)

# 1D profile using elliptical approximation
confInts_ellipse_analyt, p_ellipse_analyt = univariateprofiles_ellipse_analytical(θnames, θmle, lb, ub, H, Γ; confLevel=0.95)
confInts_ellipse, p_ellipse = univariateprofiles_ellipse(θnames, θmle, lb, ub, H, Γ; confLevel=0.95)


# Combination profiles
confIntsRel, pRel = univariateprofile_providedrelationship(AMinusB(:K, :C0, 0.0, 150.0, :KMinusC0), likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)
println(confIntsRel)

confIntsRel_ellipse, pRel_ellipse = univariateprofile_ellipse_providedrelationship(AMinusB(:K, :C0, 0.0, 150.0, :KMinusC0), θnames, θmle, lb, ub, H, Γ; confLevel=0.95)
println(confIntsRel_ellipse)


# confIntsRel, pRel = univariateprofile_providedrelationship(APlusB(:K, :C0, 50.0, 200.0, :KPlusC0), likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)
# println(confIntsRel)
# confIntsRel, pRel = univariateprofile_providedrelationship(ADivB(:K, :C0, 1.0, 200.0, :KDivC0), likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)
# println(confIntsRel)
# confIntsRel, pRel = univariateprofile_providedrelationship(ATimesB(:K, :C0, 10.0, 7500.0, :KTimesC0), likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)
# println(confIntsRel)

confIntsBivariate, pBivariate = bivariateprofiles(likelihoodFunc, fmle, data, θnames, θmle, lb, ub, 50; confLevel=0.95)
