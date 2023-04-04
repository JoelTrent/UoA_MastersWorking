# Section 1: set up packages and parameter values
using Plots, DifferentialEquations
using Distributions

include(joinpath("..", "JuLikelihood.jl"))
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
indexes=[1,3]
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
function E_inverse(em::T, z::T, m::T) where T<:Float64
    t = (z/em)*(pi/2)
    f(y) = E(y, m) - z
    r = find_zero(f, t, Order0())
    return r
end

function t_from_arclength(arc_len::T, a::T, b::T) where T<:Float64
    m = 1 - (b/a)^2
    em = E(m)
    t = 0.5*pi - E_inverse(em, em - arc_len/a, m)
    return t
end

function t_from_arclength_robust(arc_len::T, a::T, b::T, x_radius::T, y_radius::T) where T<:Float64
    if x_radius < y_radius
        return t_from_arclength(arc_len, a, b) + 0.5*pi
    else
        return t_from_arclength(arc_len, a, b) 
    end
end

function param_ellipse_x(angle::T, x_radius::T, y_radius::T, α::T, xmle::T) where T<:Float64
    return x_radius*(cos(angle)*cos(α)) - y_radius*(sin(angle)*sin(α)) + xmle
end

function param_ellipse_y(angle::T, x_radius::T, y_radius::T, α::T, ymle::T) where T<:Float64
    return x_radius*(cos(angle)*sin(α)) + y_radius*(sin(angle)*cos(α)) + ymle
end

N = 100
perimeter_len = E(m) * 4 * a_analyt
lengths = collect(LinRange(0, perimeter_len, N+1))[1:end-1]

# angle from major axis
@time angles = t_from_arclength_robust.(lengths, a_analyt, b_analyt, x_radius, y_radius)

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


function calculate_ellipse_parameters(Γ::Matrix{Float64}, ind1::Int, ind2::Int, confidence_level::Float64)
    Hw = inv(Γ[[ind1, ind2], [ind1, ind2]]) .* 0.5 ./ (quantile(Chisq(2), confidence_level)*0.5) # normalise Hw so that the RHS of the ellipse equation == 1

    ellipse_rotation = atan(2*Hw[1,2]/(Hw[1,1]-Hw[2,2]))/2
    y_radius = sqrt( (cos(ellipse_rotation)^4 - sin(ellipse_rotation)^4) / (Hw[2,2]*(cos(ellipse_rotation)^2) - Hw[1,1]*(sin(ellipse_rotation)^2))  )
    x_radius = sqrt( (cos(ellipse_rotation)^2) / (Hw[1,1] - (sin(ellipse_rotation)^2)/y_radius^2))

    a_analyt = max(x_radius, y_radius)
    b_analyt = min(x_radius, y_radius)

    return a_analyt, b_analyt, x_radius, y_radius, ellipse_rotation 
end

# start_point_shift ∈ [0,1] (random by default)
function generateNpoints_on_ellipse(Γ::Matrix{Float64}, θmle::Vector{Float64}, ind1::Int, ind2::Int, num_points::Int; confidence_level::Float64=0.01,             
    start_point_shift::Float64=rand())

    points = zeros(2,num_points)

    a_analyt, b_analyt, x_radius, y_radius, ellipse_rotation = calculate_ellipse_parameters(Γ, ind1, ind2, confidence_level)
    
    m = 1 - (b_analyt/a_analyt)^2
    perimeter_len = E(m) * 4 * a_analyt

    if !(0.0 ≤ start_point_shift && start_point_shift ≤ 1.0)
        start_point_shift = abs(rem(start_point_shift,1))
    end

    shift = start_point_shift/num_points

    lengths = collect(LinRange((shift)*perimeter_len, 
                                (1+shift)*perimeter_len, num_points+1)
                        )[1:end-1]
    angles = t_from_arclength_robust.(lengths, a_analyt, b_analyt, x_radius, y_radius)
    
    for i in 1:num_points
        points[:,i] .= param_ellipse_x(angles[i], x_radius, y_radius, ellipse_rotation, θmle[ind1]), 
                param_ellipse_y(angles[i], x_radius, y_radius, ellipse_rotation, θmle[ind2])
    end

    return points
end

# points1 = generateNpoints_on_ellipse(Γ, θmle, 1, 3, 100, confidence_level=0.01)

points2 = generateNpoints_on_ellipse(Γ, θmle, 2, 3, 50, confidence_level=0.01, start_point_shift=0.9)

# boundaryPlot=scatter(points1[1,:], points1[2,:],
#             markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=2,
#             # aspect_ratio = :equal,
#             # xlimits=(85,115),
#             # ylimits=(-5, 25)
#             )
scatter(points2[1,:], points2[2,:],
            markersize=3, markershape=:circle, markercolor=:blue, msw=0, ms=2,
            aspect_ratio = :equal,
            # xlimits=(85,115),
            # ylimits=(-5, 25)
            )
display(boundaryPlot)


import EllipseSampling

e = EllipseSampling.construct_ellipse(1.0, 2.0)
points = EllipseSampling.generateN_equally_spaced_points(100, e)
boundaryPlot=scatter(points[1,:], points[2,:],
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=2,
            aspect_ratio = :equal,
            # xlimits=(85,115),
            # ylimits=(-5, 25)
            )
