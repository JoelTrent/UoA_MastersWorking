using LinearAlgebra: norm, dot

# Section 1: set up packages and parameter values
using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using BenchmarkTools
gr()

Random.seed!(12348)
fileDirectory = joinpath("Workflow paper examples", "Logistic Model Num Opt", "Plots")
# include(joinpath("..", "plottingFunctions.jl"))
include("JuLikelihood.jl")

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
    prob=ODEProblem{true, SciMLBase.FullSpecialize}(DE!, [C0], tspan, p)
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

# initialisation. model is a mutable struct that is currently intended to hold all model information
likelihoodFunc = loglhood
θnames = [:λ, :K, :C0]
confLevel = 0.95
model = initialiseLikelihoodModel(likelihoodFunc, data, θnames, θG, lb, ub)
getMLE_ellipse_approximation!(model)

ind1, ind2 = 2, 3

newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

pointa = generate_N_equally_spaced_points(1, model.ellipse_MLE_approx.Γmle, model.core.θmle, ind1, ind2, confidence_level=0.1, start_point_shift=0.0)

uhat = [0.0, 0.0]

p = (consistent=get_consistent_tuple(model, 0.95, LogLikelihood(), 2), initGuess=initGuess, newLb=newLb, newUb=newUb, λ_opt=zeros(1), θranges=θranges, λranges=λranges, ind1=ind1, ind2=ind2, uhat=uhat, pointa=pointa[:,1])
# g(x) = bivariateΨ_gradient!(x, p)

function bivariateΨ_gradient!(Ψ::Vector, p)
    θs=zeros(eltype(Ψ), p.consistent.num_pars)

    θs[p.ind1], θs[p.ind2] = Ψ
    variablemapping2d!(θs, p.λ_opt, p.θranges, p.λranges)
    return p.consistent.loglikefunction(θs, p.consistent.data)
end

# given a point, choose search direction
function gradient_all(x); likelihoodFunc(x, data) end

function gradient_2d(x); bivariateΨ_gradient!(x, p) end

p.uhat .= [0.0,0.0]
# once found a given point, find optimised nuisance parameters and store in p.λ_opt for use in the 2d (and also 1d) gradient calculation
bivariateΨ_vectorsearch!(0.0, p) 

fullgrad = ForwardDiff.gradient(gradient_all, [p.pointa..., p.λ_opt...])
searchdir = -fullgrad[[ind1, ind2]] / norm(fullgrad[[ind1,ind2]],2)

twodgrad = ForwardDiff.gradient(gradient_2d, p.pointa)
searchdir = -twodgrad / norm(twodgrad,2)

# p.uhat .= -twodgrad#searchdir
p.uhat .= searchdir


function mappedLikeFunction(Ψ, p)
    θs=zeros(eltype(Ψ), p.consistent.num_pars)
    Ψxy=zeros(eltype(Ψ), 2)
    Ψxy .= p.pointa .+ Ψ[1]*p.uhat

    θs[p.ind1], θs[p.ind2] = Ψxy
    variablemapping2d!(θs, p.λ_opt, p.θranges, p.λranges)
    return p.consistent.loglikefunction(θs, p.consistent.data)-p.consistent.targetll
end

bivariateΨ_vectorsearch!(0.00, p)
p.pointa

# g!(gvec, x)
gvec=[0.0]
function grad_Ψ_ForwardDiff!(gvec, x)
    bivariateΨ_vectorsearch!(x[1], p)
    gvec .= ForwardDiff.gradient(grad_Ψ!, x)
    gvec 
end

function grad_Ψ_ForwardDiff_fj!(gvec, x)
    gvec .= ForwardDiff.gradient(grad_Ψ!, x)
    gvec 
end

# f(x)
function grad_Ψ!(Ψ); mappedLikeFunction(Ψ, p) end

grad_Ψ_ForwardDiff!(gvec, [0.0])

mappedLikeFunction(0.00, p)

ϵ=1e-8

println("find_zero methods:")
@btime find_zero(bivariateΨ_vectorsearch!, 0.0, atol=ϵ, Roots.Order0(); p=p)
@btime find_zero(bivariateΨ_vectorsearch!, 0.0, atol=ϵ, Roots.Order8(); p=p)

psi = find_zero(bivariateΨ_vectorsearch!, 0.0, atol=ϵ, Roots.Order8(); p=p)
p.pointa .+ psi*p.uhat
mappedLikeFunction(psi, p)


using LineSearches
using NLsolve

println("nlsolve methods:")

function NL_bivariateΨ_vectorsearch!(F, x)
    F[1] = bivariateΨ_vectorsearch!(x[1], p)
end 

fp = nlsolve(NL_bivariateΨ_vectorsearch!, [0.0], method=:trust_region)
p.pointa .+ fp.zero.*p.uhat


c = zeros(1)
NL_bivariateΨ_vectorsearch!(c, fp.zero)


function myfun!(F, J, x)
    if !(F == nothing) && !(J == nothing)
        NL_bivariateΨ_vectorsearch!(F, x)
        grad_Ψ_ForwardDiff_fj!(J, x)
        return nothing
    end
    if !(F == nothing)
        NL_bivariateΨ_vectorsearch!(F, x)
        return nothing
    end
    if !(J == nothing)
        grad_Ψ_ForwardDiff!(J, x)
        return nothing
    end
end

# may not be able to guarantee global convergence but should be ok in general 
# doesn't work well when encounters singular matrices. Is fast otherwise
# @btime nlsolve(NL_bivariateΨ_vectorsearch!, [0.0], method=:anderson);
# fp = nlsolve(NL_bivariateΨ_vectorsearch!, [0.0], method=:anderson);
# p.pointa .+ fp.zero.*p.uhat

@btime nlsolve(NL_bivariateΨ_vectorsearch!, [0.0], method=:trust_region);

@btime nlsolve(NL_bivariateΨ_vectorsearch!, grad_Ψ_ForwardDiff!, [0.0], method=:trust_region);
@btime nlsolve(NL_bivariateΨ_vectorsearch!, grad_Ψ_ForwardDiff!, [0.0], method=:newton, linesearch=MoreThuente());

# best versions - any are quite fast
@btime nlsolve(only_fj!(myfun!), [0.0], method=:trust_region);
@btime nlsolve(only_fj!(myfun!), [0.0], method=:newton, linesearch=MoreThuente());
@btime nlsolve(only_fj!(myfun!), [0.0], method=:newton, linesearch=Static());