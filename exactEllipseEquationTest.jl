# Section 1: set up packages and parameter values
using Plots, DifferentialEquations
using Distributions

include("JuLikelihood.jl")
Random.seed!(12348)

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


H, Γ = getMLE_hessian_and_covariance(funmle, θmle)
θmle
Γ
indexes=[2,3]
Hw = inv(Γ[indexes, indexes]) .* 0.5 ./ (quantile(Chisq(2), 0.95)/2) # normalise Hw so that the RHS of the ellipse equation == 1



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

a1=value(a)
b1=value(b)
α1=value(α)

(cos(α1)^2)/a1^2 + (sin(α1)^2)/b1^2
cos(α1)*sin(α1)*(1/a1^2 - 1/b1^2)
(cos(α1)^2)/b1^2 + (sin(α1)^2)/a1^2

# analytical formula for exact rotation of ellipse
ellipse_rotation = atan(2*Hw[1,2]/(Hw[1,1]-Hw[2,2]))/2
    
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

a2=value(a)
b2=value(b)

# ANALYTICAL VALUES
H, Γ = getMLE_hessian_and_covariance(funmle, θmle)
indexes=[2,3]
Hw = inv(Γ[indexes, indexes]) .* 0.5 ./ (quantile(Chisq(2), 0.01)/2) # normalise Hw so that the RHS of the ellipse equation == 1

ellipse_rotation = atan(2*Hw[1,2]/(Hw[1,1]-Hw[2,2]))/2
y_radius = sqrt( (cos(ellipse_rotation)^4 - sin(ellipse_rotation)^4) / (Hw[2,2]*(cos(ellipse_rotation)^2) - Hw[1,1]*(sin(ellipse_rotation)^2))  )
x_radius = sqrt( (cos(ellipse_rotation)^2) / (Hw[1,1] - (sin(ellipse_rotation)^2)/y_radius^2))

a_analyt = max(x_radius, y_radius)
b_analyt = min(x_radius, y_radius)

m = 1 - (b_analyt/a_analyt)^2

using Elliptic
using Roots

# full perimeter
E(m) * 4 * a_analyt
E(0.5*pi, m) * 4 * a_analyt

# functions from https://www.johndcook.com/blog/2022/11/02/ellipse-rng/
function E_inverse(z, m)
    em = E(m)
    t = (z/em)*(pi/2)
    f(y) = E(y, m) - z
    r = find_zero(f, t, Order0())
    return r
end

function t_from_length(length, a, b)
    m = 1 - (b/a)^2
    T = 0.5*pi - E_inverse(E(m) - length/a, m)
    return T
end

function t_from_length_robust(length, a, b, x_radius, y_radius)
    if x_radius < y_radius
        return t_from_length(length, a, b) + 0.5*pi
    else
        return t_from_length(length, a, b) 
    end
end

function param_ellipse_x(angle::T, x_radius::T, y_radius::T, α::T, xmle::T) where T<:Float64
    return x_radius*cos(angle)*cos(α) - y_radius*sin(angle)*sin(α) + xmle
end

function param_ellipse_y(angle::T, x_radius::T, y_radius::T, α::T, ymle::T) where T<:Float64
    return x_radius*cos(angle)*sin(α) + y_radius*sin(angle)*cos(α) + ymle
end

N = 100
perimeter_len = E(m) * 4 * a_analyt
lengths = collect(LinRange(0, perimeter_len, N+1))[1:end-1]

# angle from major axis
@time angles = t_from_length_robust.(lengths, a_analyt, b_analyt, x_radius, y_radius)

x = param_ellipse_x.(angles, x_radius, y_radius, ellipse_rotation, θmle[indexes[1]])
y = param_ellipse_y.(angles, x_radius, y_radius, ellipse_rotation, θmle[indexes[2]])

using Plots
gr()

boundaryPlot=scatter(x, y,
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=2,
            aspect_ratio = :equal,
            # xlimits=(85,115),
            # ylimits=(-5, 25)
            )
display(boundaryPlot)

# check all x,y points are on the boundary

[analytic_ellipse_loglike([x[i],y[i]], indexes, (θmle=θmle, Γmle = Γ)) for i in 1:N] 