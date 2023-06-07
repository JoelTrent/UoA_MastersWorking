
using EllipseSampling
using LinearAlgebra
using Plots
using DifferentialEquations, Random, Distributions
gr()

include("JuLikelihood.jl")

e = construct_ellipse(1., 1.0)
num_points=15
ind_lambda = 1
ind1, ind2 = 2,3
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
θnames = ["λ", "K", "C0"]

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
# Section 12: 
# 3D approximation of the likelihood around the MLE solution
H, Γ = getMLE_hessian_and_covariance(funmle, θmle)

##############################################################################################
##############################################################################################
##############################################################################################

# place N equally spaced points on boundary of ellipse approximation at some small confidence level.
# points = generate_N_equally_spaced_points(num_points, e, start_point_shift=0.0)
points = generate_N_equally_spaced_points(num_points, Γ, θmle, ind1, ind2, 
                confidence_level=0.1, start_point_shift=0.0)

# determine true loglikelihood function values at all N points
# ll_vals = .....
ll_vals = zeros(num_points) .- 0.03
# θmle = zeros(model.core.num_pars)

gradient_i = zeros(num_points)
# for i in 1:num_points
    # estimate gradient of normal as function value change between mle and ellipse point ÷ euclidean distance
    

# end

# calculate normal at each point
# Also implement via forward diff - gives both magnitude and 
f(x) = analytic_ellipse_loglike(x, [ind1, ind2], (θmle=θmle, Γmle=Γ))
ForwardDiff.gradient(f, [1.0,0.0])


function f_full(x::Vector) 
    θs=zeros(eltype(x), 3)
    θs[ind1], θs[ind2] = x[1:2]
    θs[ind_lambda] = x[3]
    ellipse_loglike(θs[:], (θmle=θmle, Hmle=H))
end

function bivariateΨ_ellipse!(x)
    θs=zeros(3)
    
    function fun(λ)
        θs[ind1], θs[ind2] = x
        θs[ind_lambda] = λ[1]
        return ellipse_loglike(θs, (θmle=θmle, Hmle=H))
    end
    (xopt,fopt)=optimise(fun, [θmle[ind_lambda]], [lb[ind_lambda]], [ub[ind_lambda]])
    # return fun(0.01)
    return xopt[1]
end

ellipse_loglike(θmle, (θmle=θmle, Hmle=H))

f([100, 10])
lambda=0.01
f_full([100.0, 10.0, lambda])

bivariateΨ_ellipse!([100, 10])

ForwardDiff.gradient(f, [1.0,0.0])

normal_vectors = zeros(2, num_points)

# NOTE: METHOD REQUIRES THERE TO BE AT LEAST 3 POINTS
function normal_vector!(normal_vectors, index, point1, point2)
    normal_vectors[:, index] .= [(point2[2]-point1[2]), -(point2[1]-point1[1])]
    normal_vectors[:, index] .= @view(normal_vectors[:, index]) / norm(@view(normal_vectors[:,index])) 
    return nothing
end

normal_vector!(normal_vectors, 1, @view(points[:, end]), @view(points[:, 2]))
normal_vector!(normal_vectors, num_points, @view(points[:, end-1]), @view(points[:, 1]))

for i in 2:num_points-1
   normal_vector!(normal_vectors, i, @view(points[:, i-1]), @view(points[:, i+1]))
end

boundary = scatter(points[1,:], points[2,:], label="Ellipse", markersize=6)
xlabel!(θnames[ind1])
ylabel!(θnames[ind2])
# quiver!(points[1,:], points[2,:], lw=2, quiver=(normal_vectors[1,:], normal_vectors[2,:]))

normal_vectors_true = zeros(2, num_points)
for i in 1:num_points
    normal_vectors_true[:,i] .= -ForwardDiff.gradient(f, points[:,i])
    # normal_vectors_true[:, i] .= @view(normal_vectors_true[:, i]) / norm(@view(normal_vectors_true[:,i])) 
end

quiver!(points[1,:], points[2,:], lw=4, linecolor=:maroon, quiver=(normal_vectors_true[1,:], normal_vectors_true[2,:]))

normal_vectors_full = zeros(2, num_points)
for i in 1:num_points
    global lambda = bivariateΨ_ellipse!(points[:,i])
    normal_vectors_full[:,i] .= -ForwardDiff.gradient(f_full, [points[:,i]..., lambda])[1:2]
    # normal_vectors_full[:, i] .= @view(normal_vectors_full[:, i]) / norm(@view(normal_vectors_full[:,i])) 
end

quiver!(points[1,:], points[2,:], lw=1, linecolor=:lightgrey, quiver=(normal_vectors_full[1,:], normal_vectors_full[2,:]))