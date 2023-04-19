using Plots
using Distributions
using NLopt
using ForwardDiff
using LinearAlgebra

# ---------------------------------------------
# ---------- Load 'Sloppihood' tools ----------
# ---------------------------------------------
include("SloppihoodTools.jl")
using .SloppihoodTools

# ---------------------------------------------
# ---- User inputs in original 'x,y' param ----
# ---------------------------------------------
# parameter -> data dist (forward) mapping
distrib_xy(xy) = Normal(xy[1]*xy[2],sqrt(xy[1]*xy[2]*(1-xy[2]))) # 
# variables
varnames = Dict("x"=>"n", "y"=>"p")
# initial guess for optimisation
xy_initial =  [50, 0.3]# x (i.e. n) and y (i.e. p), starting guesses
# parameter bounds
xy_lower_bounds = [0.0001,0.0001]
xy_upper_bounds = [500,1.0]
# true parameter
xy_true = [100,0.2] #x,y, truth. N, p
N_samples = 10 # measurements of model
# generate data
#data = rand(distrib_xy(xy_true),N_samples)
data = [21.9,22.3,12.8,16.4,16.4,20.3,16.2,20.0,19.7,24.4]
#data = [21.9 22.3 12.8 16.4 16.4 20.3 16.2 20.0 19.7 24.4]
#data = [16,18,22,25,27]

# ---- use above to construct log likelihood in original parameterisation given (iid) data
lnlike_xy = SloppihoodTools.construct_lnlike_xy(distrib_xy,data)
# ----

# ---------------------------------------------
# --- Analysis in original parameterisation ---
# ---------------------------------------------
include("stat_model_analysis_xy.jl")

# ---------------------------------------------
# ----- Analysis in log parameterisation -----
# --------------------------------------------- 
include("stat_model_analysis_log.jl")

# ----------------------------------------------------
# - Analysis in sloppihood-informed parameterisation -
# ----------------------------------------------------
include("stat_model_analysis_sip.jl")
