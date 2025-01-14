# Section 1: set up packages and parameter values
# using BenchmarkTools

using Distributed
using Revise
# if nprocs()==1; addprocs(10) end
using LikelihoodBasedProfileWiseAnalysis
@everywhere using Revise
@everywhere using DifferentialEquations, Random, Distributions
@everywhere using LikelihoodBasedProfileWiseAnalysis

# Workflow functions ##########################################################################

# Section 2: Define ODE model
@everywhere function DE!(dC, C, p, t)
    λ,K=p
    dC[1]= λ * C[1] * (1.0 - C[1]/K)
end

# Section 3: Solve ODE model
@everywhere function odesolver(t, λ, K, C0)
    p=(λ,K)
    tspan=(0.0, t[end])
    prob=ODEProblem(DE!, [C0], tspan, p)
    sol=solve(prob, saveat=t, verbose=false)
    return sol[1,:]
end

# Section 4: Define function to solve ODE model 
@everywhere function ODEmodel(t, a)
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

@everywhere function solvedmodel(t, a)
    return (a[2]*a[3]) ./ ((a[2]-a[3]) .* (exp.(-a[1] .* t)) .+ a[3])
end

# Section 6: Define loglikelihood function
@everywhere function loglhood(a, data)
    # y=ODEmodel(data.t, a)
    y=solvedmodel(data.t, a)
    e=0
    e=sum(loglikelihood(data.dist, data.yobs .- y))
    return e
end

# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
@everywhere function funmle(a,_); return loglhood(a, data) end 

# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]

# true data
ytrue = ODEmodel(t, a)

Random.seed!(12348)
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
par_magnitudes = [0.005, 10, 10]
# par_magnitudes = [1, 1, 1]

##############################################################################################
# Section 9: Find MLE by numerical optimisation, visually compare data and MLE solution
# Use Nelder-Mead algorithm to estimate maximum likelihood solution for parameters given 
# noisy data
function funmle_max(θ,_); -funmle(θ,0); end
(xopt, fopt) = optimise(funmle_max, default_OptimizationSettings(), θG, lb, ub)
fmle=fopt
λmle, Kmle, C0mle = xopt .* 1.0
θmle = [λmle, Kmle, C0mle]
ymle(t) = Kmle*C0mle/((Kmle-C0mle)*exp(-λmle*t)+C0mle) # full solution


##############################################################################################
# Section 12: Prediction interval from the univariate quadratic approximation of log-likelihood for parameter λ
# Compute and propogate uncertainty forward from the univariate likelihood for parameter λ

@everywhere likelihoodFunc = loglhood
# REQUIRED TO SPECIFY WITH DEFAULT OF 3RD ARG DEPENDING ON DATA SO THAT YMLE IS FOUND
# @everywhere function predictFunc(θ, data, t=data.t); ODEmodel(t, θ) end
@everywhere function predictFunc(θ, data, t=data.t); solvedmodel(t, θ) end
θnames = [:λ, :K, :C0]
confLevel = 0.95

# initialisation. model is a mutable struct that is currently intended to hold all model information
# using FiniteDiff
# using ReverseDiff
# using Tracker
# optim_settings=OptimizationSettings(AutoTracker(), NLopt.LD_LBFGS(), (xtol_rel=1e-9,))
optim_settings=default_OptimizationSettings()
# optim_settings=OptimizationSettings(SciMLBase.NoAD(), NLopt.LN_BOBYQA(), NamedTuple())
model = initialise_LikelihoodModel(likelihoodFunc, predictFunc, data, θnames, θG, lb, ub, par_magnitudes, find_zero_atol=0.001, optimizationsettings=optim_settings);

icdf = -quantile(Chisq(1),0.95)/2
# prediction for t=200
# find values of f(θ,t=200) that make PPL approximately == model.core.maximisedmle + icdf
function VPL_to_max(θ, data, z::Real, t::Real, SD::Real)
    return loglhood(θ, data) - ((z-solvedmodel(t, θ))/SD)^2
end

function VPL(z, t_interest, SD)
    function f(θ) return VPL_to_max(θ, data, zi, t_interest, SD) end

    zi=0.0
    vpls = zeros(length(z))

    for i in eachindex(z)
        zi = z[i]
        (_, vpls[i]) = optimise(f, θG, lb, ub)
    end
    return vpls
end
function VPL_to_PPL(z, t_interest, SD)
    function f(θ) return VPL_to_max(θ, data, zi, t_interest, SD) end

    zi=0.0
    ppls = zeros(length(z))

    for i in eachindex(z)
        zi = z[i]
        (xopt, fopt) = optimise(f, θG, lb, ub)
        ppls[i] = fopt - 0.5 * (solvedmodel(t_interest, xopt) - zi)^2 / SD^2
    end
    return ppls
end
z = collect(10:0.5:80) 
vpls = VPL(z, 200, σ)
plot(z, vpls, label="VPL")

ppls = VPL_to_PPL(z, 200, σ)
plot!(z, ppls, label="PPL")
hline!([model.core.maximisedmle + icdf], label="confidence threshold")

plot(z, ppls .- (model.core.maximisedmle + icdf), label="Rescaled PPL")

full_likelihood_sample!(model, 1000000, sample_type=LatinHypercubeSamples(), use_distributed=false)
full_likelihood_sample!(model, 100, sample_type=UniformGridSamples(), use_distributed=false)

# not strictly required - functions that rely on these being computed will check if 
# they're missing and call this function on model if so.
# getMLE_ellipse_approximation!(model)

@time univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
@time univariate_confidenceintervals!(model, profile_type=EllipseApprox(), use_distributed=false)
LikelihoodBasedProfileWiseAnalysis.TimerOutputs.enable_debug_timings(LikelihoodBasedProfileWiseAnalysis)
LikelihoodBasedProfileWiseAnalysis.TimerOutputs.disable_debug_timings(LikelihoodBasedProfileWiseAnalysis)
LikelihoodBasedProfileWiseAnalysis.TimerOutputs.reset_timer!(LikelihoodBasedProfileWiseAnalysis.timer)
univariate_confidenceintervals!(model, profile_type=LogLikelihood(), existing_profiles=:overwrite)
display(LikelihoodBasedProfileWiseAnalysis.timer)
@time get_points_in_intervals!(model, 101, additional_width=0.2, use_threads=true)

@time bivariate_confidenceprofiles!(model, 50, profile_type=LogLikelihood(), method=RadialMLEMethod(0.0), existing_profiles=:overwrite, use_distributed=false)

@time bivariate_confidenceprofiles!(model, 500, profile_type=EllipseApproxAnalytical(), method=Fix1AxisMethod(), existing_profiles=:overwrite, save_internal_points=true, use_distributed=false)
@time bivariate_confidenceprofiles!(model, 60, profile_type=LogLikelihood(), method=IterativeBoundaryMethod(30, 5, 5, 0.0), existing_profiles=:overwrite, save_internal_points=true, use_distributed=false)
bivariate_confidenceprofiles!(model, 60, profile_type=LogLikelihood(), method=ContinuationMethod(2,0.1), existing_profiles=:overwrite, save_internal_points=true)
@time sample_bivariate_internal_points!(model, 200, hullmethod=ConvexHullMethod(), sample_type=LatinHypercubeSamples(), use_threads=false)
sample_bivariate_internal_points!(model, 100, hullmethod=MPPHullMethod(), sample_type=UniformRandomSamples())

# bivariate_confidenceprofiles!(model, 10, profile_type=LogLikelihood(), method=RadialRandomMethod(3))
# bivariate_confidenceprofiles!(model, 10, profile_type=LogLikelihood(), method=SimultaneousMethod(1))
# bivariate_confidenceprofiles!(model, 10, profile_type=EllipseApprox(), method=Fix1AxisMethod())

# combine_bivariate_boundaries!(model, not_evaluated_predictions=false)

# bivariate_confidenceprofiles!(model, 10, profile_type=LogLikelihood(), method=Fix1AxisMethod())
# println()

# @time bivariate_confidenceprofiles!(model, 100, profile_type=LogLikelihood(), method=IterativeBoundaryMethod(10, 10, 10), confidence_level=0.95, existing_profiles=:overwrite, save_internal_points=true)
# @time bivariate_confidenceprofiles!(model, 200, confidence_level=0.95, profile_type=EllipseApprox(), method=ContinuationMethod(2, 0.1, 0.0), existing_profiles=:overwrite, use_distributed=true)
# bivariate_confidenceprofiles!(model, 50, confidence_level=0.95, profile_type=LogLikelihood(), method=RadialMLEMethod(0.0, 0.1), save_internal_points=true, existing_profiles=:overwrite)

bivariate_confidenceprofiles!(model, 52, profile_type=EllipseApproxAnalytical(), method=AnalyticalEllipseMethod(0.0, 0.1), existing_profiles=:overwrite)

# bivariate_confidenceprofiles!(model, 100, confidence_level=0.95, profile_type=LogLikelihood(), method=ContinuationMethod(5, 0.1, 0.0), save_internal_points=true, existing_profiles=:overwrite)

# @time bivariate_confidenceprofiles!(model, 100, confidence_level=0.95, method=AnalyticalEllipseMethod())

dimensional_likelihood_samples!(model, 1, 300, sample_type=UniformGridSamples(), use_threads=true)
dimensional_likelihood_samples!(model, 2, 3000, sample_type=UniformRandomSamples(), use_threads=true)
dimensional_likelihood_samples!(model, 2, 2000, use_threads=true)

# dimensional_likelihood_samples!(model, 3, 20000, use_threads=false)

prediction_locations = collect(LinRange(t[1], t[end], 30))
generate_predictions_univariate!(model, prediction_locations, 1.0, profile_types=[EllipseApprox(), LogLikelihood()], use_distributed=false)
generate_predictions_bivariate!(model, prediction_locations, 0.1)
generate_predictions_dim_samples!(model, prediction_locations, 0.1)

using Plots
gr()

# ellipse_points = generate_N_clustered_points(100, model.ellipse_MLE_approx.Γmle,
#                                                         model.core.θmle, 1, 2,
#                                                         confidence_level=0.1, 
#                                                         start_point_shift=0.0,
#                                                         sqrt_distortion=1.0)

# scatter(ellipse_points[1,:], ellipse_points[2,:], legend=false,  markeropacity=0.4)


# Profiles ################################################################
plots = plot_univariate_profiles(model, 0.5, 0.6, palette_to_use=:Spectral_8)
for i in eachindex(plots); display(plots[i]) end

# plots = plot_univariate_profiles_comparison(model, 0.2, 0.2, palette_to_use=:Spectral_8)
# for i in eachindex(plots); display(plots[i]) end

plots = plot_bivariate_profiles(model, 0.2, 0.2, include_internal_points=true, markeralpha=0.9)
for i in eachindex(plots); display(plots[i]) end

# plots = plot_bivariate_profiles(model, 0.2, 0.2, for_dim_samples=true, include_internal_points=true, markeralpha=0.9)
# for i in eachindex(plots); display(plots[i]) end

# plots = plot_bivariate_profiles_comparison(model, 0.2, 0.2, compare_within_methods=false, include_dim_samples=true)
# for i in eachindex(plots); display(plots[i]) end

# plots = plot_bivariate_profiles_comparison(model, 0.2, 0.2, compare_within_methods=true)
# for i in eachindex(plots); display(plots[i]) end

# # Predictions ############################################################
plots = plot_predictions_individual(model, prediction_locations, 2, profile_types=[EllipseApproxAnalytical()])
for i in eachindex(plots); display(plots[i]) end

plots = plot_predictions_individual(model, prediction_locations, 3, ylims=[0,120]; for_dim_samples=true)
for i in eachindex(plots); display(plots[i]) end
vline!([200])

# plots = plot_predictions_individual(model, prediction_locations, 2, ylims=[0,120], profile_types=[LogLikelihood()])
# for i in eachindex(plots); display(plots[i]) end

# union_plot = plot_predictions_union(model, prediction_locations, 1, ylims=[0,120])

# union_plot = plot_predictions_union(model, prediction_locations, 2, ylims=[0,120], for_dim_samples=true, include_lower_confidence_levels=true, compare_to_full_sample_type=LatinHypercubeSamples())
# display(union_plot)



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
newlb, newub = transformbounds_NLopt(forward_parameter_transformLog, lb, ub)

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

newlb, newub = transformbounds_NLopt(forward_parameter_transformKminusC0, lb, ub)

# REMOVE WORKER PROCESSORS WHEN FINISHED #######################################
rmprocs(workers())
################################################################################
