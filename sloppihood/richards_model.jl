using Plots
using Distributions
using NLopt
using ForwardDiff
using LinearAlgebra
using DifferentialEquations

# ---------------------------------------------
# ---------- Load 'Sloppihood' tools ----------
# ---------------------------------------------
include("SloppihoodTools.jl")
using .SloppihoodTools

# Define ODE model
function DE!(dC,C,xy,t)
    λ,β=xy
    dC[1]=λ*C[1]*(1.0-(C[1]/100)^β);
end

# ODE model solver
function solve_ode(t,xy)
    tspan=(0.0,maximum(t));
    prob=ODEProblem(DE!,[1.0],tspan,xy);
    sol=solve(prob,saveat=t,abstol=1e-6,reltol=1e-6,maxiters=Int(1e6));
    return sol[1,:]
end

# ---------------------------------------------
# ---- User inputs in original 'x,y' param ----
# ---------------------------------------------
# time measurements
T = 10
NT = 10
t_data=LinRange(0,T,NT)
t_model=LinRange(0,T,10*NT)
# parameter -> data dist (forward) mapping
σ = 2
distrib_xy(xy) = MultivariateNormal(solve_ode(t_data,xy),σ^2*I(length(t_data)))
#distrib_xy(xy) = Normal.(solve_ode(t_data,xy),σ) # note actually vector of dist at each t
# variables
varnames = Dict("x"=>"λ", "y"=>"β")
# parameter bounds
λmin=0.1
λmax=3.0
βmin=0.01
βmax=1.5
xy_lower_bounds = [λmin,βmin]
xy_upper_bounds = [λmax,βmax]
# initial guess for optimisation
xy_initial =  0.5*(xy_lower_bounds + xy_upper_bounds) # [1.5, 1.5]# x (i.e. n) and y (i.e. p), starting guesses
# true parameter
λ_true=1.0; β_true=0.40;
xy_true = [λ_true,β_true] #x,y, truth. N, p
# generate data
Nrep = 100
data = rand(distrib_xy(xy_true),Nrep)

#data = [6.3, 3.0, 9.9, 4.9, 22.0, 27.7, 38.8, 61.3, 78.3, 88.8]
# 6.813809329808995
# 0.38050281243017636
# 11.363990646497129
# 4.543088881681962
# 19.556248506103422
# 20.316871853712257
# 48.58119902475581
# 61.995374077995656
# 78.70631351406446
# 80.36593625377138
# ---- use above to construct log likelihood in original parameterisation given (iid) data
lnlike_xy = SloppihoodTools.construct_lnlike_xy(distrib_xy,data;dist_type=:multi)
# ----

# ---------------------------------------------
# --- Analysis in original parameterisation ---
# ---------------------------------------------
include("richards_model_analysis_xy.jl")

# ---------------------------------------------
# ----- Analysis in log parameterisation -----
# --------------------------------------------- 
include("richards_model_analysis_log.jl")

# ----------------------------------------------------
# - Analysis in sloppihood-informed parameterisation -
# ----------------------------------------------------
include("richards_model_analysis_sip.jl")


