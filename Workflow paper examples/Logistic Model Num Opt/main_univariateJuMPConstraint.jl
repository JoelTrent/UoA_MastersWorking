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
include(joinpath("..", "ellipseLikelihood.jl"))

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

# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
function funmle(a); return loglhood((data,σ), a) end 

# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]

# true data
data0 = ODEmodel(t, a, σ)

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
λmle, Kmle, C0mle = xopt .* 1.0
θmle = [λmle, Kmle, C0mle]
ymle(t) = Kmle*C0mle/((Kmle-C0mle)*exp(-λmle*t)+C0mle) # full solution

# p1 = plot(ymle, 0, 1000, color=:turquoise1, xlabel="t", ylabel="C(t)",
#             legend=false, lw=4, xlims=(0,1100), ylims=(0,120),
#             xticks=[0,500,1000], yticks=[0,50,100])

# p1 = scatter!(t, data, legend=false, msw=0, ms=7,
#             color=:darkorange, msa=:darkorange)
# display(p1)
# savefig(p1, joinpath(fileDirectory,"mle.pdf"))

# 3D approximation of the likelihood around the MLE solution
H, Γ = getMLE_hessian_and_covariance(funmle, θmle)


##############################################################################################
# Section 12: Prediction interval from the univariate quadratic approximation of log-likelihood for parameter λ
# Compute and propogate uncertainty forward from the univariate likelihood for parameter λ
df = 1
llstar = -quantile(Chisq(df), 0.95)/2

# nonlinear optimisation version with constraint - minimum working example
using JuMP
import Ipopt
# model = Model(Ipopt.Optimizer)
# set_silent(model)
# θG = [λ, K, C0]
# lb = [λmin, Kmin, C0min]
# ub = [λmax, Kmax, C0max]

# function b(θ...); loglhood(data, θ, σ)-fmle end

# function myConstraint(θ...); θ[1] end
# myConstraintRHS = 0.02

# register(model, :my_obj, 3, b; autodiff = true)
# register(model, :myConstraint, 3, myConstraint; autodiff = true)

# @variable(model, θ[i=1:3], lower_bound=lb[i], upper_bound=ub[i], start=θmle[i])
# @NLobjective(model, Max, my_obj(θ...))
# @NLconstraint(model, myConstraint(θ...)==myConstraintRHS)

# JuMP.optimize!(model)

# objective_value(model)
# JuMP.value.(θ)


struct ConfidenceStruct
    mle::Float64
    confidence_interval::Vector{<:Float64}
    bounds::Vector{<:Float64}
end

likelihoodFunc = loglhood
data = (data, σ)
θnames = [:a, :b, :c]
confLevel=0.95

loglhood(data, θG)

# function univariateprofiles_constrained(likelihoodFunc, data, fmle, θnames, θmle, lb, ub; confLevel=0.95)

df = 1
llstar = -quantile(Chisq(df), confLevel)/2
num_vars = length(θnames)

m = Model(Ipopt.Optimizer)
set_silent(m)

function my_obj(θ...); likelihoodFunc(data, θ)-fmle-llstar end
register(m, :my_obj, num_vars, my_obj; autodiff = true)
@variable(m, θ[i=1:num_vars], lower_bound=lb[i], upper_bound=ub[i], start=θmle[i])
@NLobjective(m, Max, my_obj(θ...))

confidenceDict = Dict{Symbol, ConfidenceStruct}()

# for (j, θname) in enumerate(θnames)
#     println(θname)
j = 1
θname=:a

if j>1
    delete(m, con)
    unregister(m, :myConstraint)
end

function myConstraint(θ...); θ[j] end
register(m, :myConstraint, num_vars, myConstraint; autodiff = true)
@constraint(m, con, myConstraint(θ...)==0.0)

set_normalized_rhs(m.obj_dict[:con], 1.0)

univariateΨ

function univariateΨ(Ψ)
    set_normalized_rhs(con, Ψ)
    JuMP.optimize!(m)
    return objective_value(m)            
end

univariateΨ(lb[1])

set_normalized_rhs(con, 0.01-ϵ)
JuMP.optimize!(m)

ϵ=(ub[j]-lb[j])/10^6

interval=zeros(2)

interval[1] = find_zero(univariateΨ, (lb[j], θmle[j]), atol=ϵ, Roots.Brent())
println("Made it to here")
interval[2] = find_zero(univariateΨ, (θmle[j], ub[j]), atol=ϵ, Roots.Brent())

confidenceDict[θname] = ConfidenceStruct(θmle[j], interval, [lb[j], ub[j]])
# end

#     return confidenceDict
# end


# univariateprofiles_constrained(loglhood, (data, σ), fmle, [:λ, :K, :C0], θmle, lb, ub)