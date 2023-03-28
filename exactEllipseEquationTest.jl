

# Section 1: set up packages and parameter values
using Plots, DifferentialEquations

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

H, Γ = getMLE_hessian_and_covariance(funmle, θmle)
θmle
Γ
Hw = inv(Γ[[2,3], [2,3]])



using JuMP
import Ipopt
    
m = Model(Ipopt.Optimizer)
set_silent(m)

# variables will be binary integer automatically due to how the obj function is setup
# IF the transformation function applied to each θ[i] is monotonic between lb[i] and ub[i]
@variable(m, α, lower_bound=-pi, upper_bound=pi, start=0.0)
@variable(m, a, lower_bound=1e-10, start=1.0)
@variable(m, b, lower_bound=1e-10, start=1.0)


@NLconstraint(m, Hw[1,1] == (cos(α)^2)/a^2 + (sin(α)^2)/b^2)
@NLconstraint(m, Hw[2,2] == (cos(α)^2)/b^2 + (sin(α)^2)/a^2)
@NLconstraint(m, 2*Hw[1,2] == sin(2*α)*(1/a^2 - 1/b^2))

@NLobjective(m, Max, 1.0)
JuMP.optimize!(m)
solution_summary(m)

a=value(a)
b=value(b)
α=value(α)

(cospi(α)^2)/a^2 + (sinpi(α)^2)/b^2
cospi(α)*sinpi(α)*(1/a^2 - 1/b^2)
(cospi(α)^2)/b^2 + (sinpi(α)^2)/a^2

# analytical formula for exact rotation of ellipse
ellipse_rotation = atan(2*Hw[1,2]/(Hw[1,1]-Hw[2,2]))/2

using JuMP
import Ipopt
    
m = Model(Ipopt.Optimizer)
set_silent(m)

# variables will be binary integer automatically due to how the obj function is setup
# IF the transformation function applied to each θ[i] is monotonic between lb[i] and ub[i]
@variable(m, a, lower_bound=1e-10, start=1.0)
@variable(m, b, lower_bound=1e-10, start=1.0)

@NLconstraint(m, Hw[1,1] == (cos(ellipse_rotation)^2)/a^2 + (sin(ellipse_rotation)^2)/b^2)
@NLconstraint(m, Hw[2,2] == (cos(ellipse_rotation)^2)/b^2 + (sin(ellipse_rotation)^2)/a^2)

@NLobjective(m, Max, 1.0)
JuMP.optimize!(m)
solution_summary(m)

a=value(a)
b=value(b)

