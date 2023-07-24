##############################################################################
######################### TWO SPECIES LOGISTIC MODEL #########################
######### FROM https://github.com/ProfMJSimpson/profile_predictions ##########
##############################################################################

using Distributed
using PlaceholderLikelihood
# if nprocs()==1; addprocs(8, exeflags=`--threads 1`) end
@everywhere using Revise
@everywhere using DifferentialEquations, Random, Distributions
@everywhere using PlaceholderLikelihood

using Plots
gr()

# DEFINE DATA #########################################################################
t=[0, 769, 1140, 1488, 1876, 2233, 2602, 2889, 3213, 3621, 4028]

data11=[0.748717949, 0.97235023, 5.490243902, 17.89100529, 35, 56.38256703, 64.55087666, 66.61940299, 71.67362453, 80.47179487, 79.88291457]

data12=[1.927065527, 0.782795699, 1.080487805, 2.113227513, 3.6, 2.74790376, 2.38089652, 1.8, 0.604574153, 1.305128205, 1.700502513]

# Named tuple of all data required within the likelihood function
data = (data11=data11, data12=data12, t=t)

# DEFINE MODEL AND LOG-LIKELIHOOD FUNCTION ############################################
@everywhere function DE!(dC, C, p, t) # function to define process model
    λ1, λ2, δ, KK = p # parameter vector for the process model
    S = C[1] + C[2]
    dC[1] = λ1 * C[1] * (1.0 - S/KK) # define differential equations
    dC[2] = λ2 * C[2] * (1.0 - S/KK) - δ*C[2]*C[1]/KK; # define differential equations
end

@everywhere function odesolver(t, λ1, λ2, δ, KK, C01, C02) #function to solve the process model
    p=(λ1, λ2, δ, KK) # parameter vector for the process model
    C0=[C01, C02]
    tspan=(0.0, maximum(t)) # time horizon
    prob=ODEProblem(DE!, C0, tspan, p) # define ODE model
    sol=solve(prob, saveat=t)  # solve and save solutions
    return sol[1,:], sol[2,:]
end

@everywhere function ODEmodel(t, θ) # function to solve the process model
    (y1, y2) = odesolver(t, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6]) #solve the process model
    return y1, y2
end
     
@everywhere function loglhood(θ, data) # function to evaluate the loglikelihood for the data given parameters θ
    (y1, y2) = ODEmodel(data.t, θ)
    dist = Normal(0.0, θ[7])
    e = loglikelihood(dist, data.data11 - y1) + loglikelihood(dist, data.data12 - y2)
    return e
end

@everywhere function predictFunc(θ, data, t=data.t)
    y1, y2 = ODEmodel(t, θ) 
    y = hcat(y1,y2)
    return y
end

# initial parameter estimates 
λ1g=0.002; λ2g=0.002; δg=0.0; KKg=80.0; C0g=[1.0, 1.0]; σg=1.0; 

θG = [λ1g, λ2g, δg, KKg, C0g[1], C0g[2], σg] #parameter estimates
lb = [0.0001, 0.0001, 0.0, 60.0, 0.0001, 0.0001, 0.0001]; #lower bound
ub = [0.01, 0.01, 0.01, 90.0, 1.0, 1.0, 3.0]; #upper bound
par_magnitudes = [0.01, 0.01, 0.01, 10, 1, 1, 1]
θnames = [:λ1, :λ2, :δ, :KK, :C01, :CO2, :σ]

using ForwardDiff
using FiniteDiff
optim_settings=OptimizationSettings(AutoFiniteDiff(), NLopt.LN_BOBYQA(), NamedTuple())
model = initialise_LikelihoodModel(loglhood, predictFunc, data, θnames, θG, lb, ub, par_magnitudes)

@everywhere function loglhood2(θ, data) # function to evaluate the loglikelihood for the data given parameters θ
    (y1, y2) = ODEmodel(data.t, θ)
    dist = Normal(0.0, data.σ)
    e = loglikelihood(dist, data.data11 - y1) + loglikelihood(dist, data.data12 - y2)
    return e
end

data2 = (data..., σ=model.core.θmle[7])
model = initialise_LikelihoodModel(loglhood2, predictFunc, data2, θnames[1:6], θG[1:6], lb[1:6], ub[1:6], par_magnitudes[1:6]);

# full_likelihood_sample!(model, 5000000, sample_type=LatinHypercubeSamples(), use_distributed=false)

univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals!(model, profile_type=EllipseApprox())
univariate_confidenceintervals!(model, profile_type=LogLikelihood())
get_points_in_intervals!(model, 100, additional_width=0.5)

# bivariate_confidenceprofiles!(model, 5, 20, profile_type=LogLikelihood(), method=IterativeBoundaryMethod(10, 0, 5, 0.0, use_ellipse=false))

bivariate_confidenceprofiles!(model, [[2,6]], 20, profile_type=EllipseApprox(), method=RadialRandomMethod(3))
@time sample_bivariate_internal_points!(model, 200, hullmethod=ConvexHullMethod(), sample_type=LatinHypercubeSamples(), use_threads=true)
# bivariate_confidenceprofiles!(model, [[1,2]], 100, profile_type=LogLikelihood(), method=SimultaneousMethod(0.01))
# bivariate_confidenceprofiles!(model, [[1,2]], 5, profile_type=LogLikelihood(), method=ContinuationMethod(2, 0.1), existing_profiles=:overwrite)

# dimensional_likelihood_sample!(model, 2, 500)
println()

plots = plot_univariate_profiles(model, 0.5, 0.6, palette_to_use=:Spectral_8)
for i in eachindex(plots)
    display(plots[i])
end

plots = plot_univariate_profiles_comparison(model, 0.2, 0.2, profile_types=[EllipseApproxAnalytical(), EllipseApprox(), LogLikelihood()], palette_to_use=:Spectral_8)
for i in eachindex(plots)
    display(plots[i])
end

plots = plot_bivariate_profiles(model, 0.2, 0.2, include_internal_points=true, markeralpha=0.9)
for i in eachindex(plots)
    display(plots[i])
end

# plots = plot_bivariate_profiles(model, 0.2, 0.2, for_dim_samples=true, include_internal_points=true, markeralpha=0.9)
# for i in eachindex(plots)
#     display(plots[i])
# end


# prediction_locations = collect(LinRange(t[1], t[end], 50))
# generate_predictions_univariate!(model, prediction_locations, 0.2, profile_types=[EllipseApprox(), LogLikelihood()])
# generate_predictions_dim_samples!(model, prediction_locations, 0.2)

# union_plot = plot_predictions_union(model, prediction_locations, 1, for_dim_samples=false, include_lower_confidence_levels=true, compare_to_full_sample_type=LatinHypercubeSamples())
# display(union_plot)

# union_plot = plot_predictions_union(model, prediction_locations, 2, for_dim_samples=true, include_lower_confidence_levels=true, compare_to_full_sample_type=LatinHypercubeSamples())
# display(union_plot)

# union_plot = plot_predictions_union(model, prediction_locations, 6, for_dim_samples=true, include_lower_confidence_levels=true)
# display(union_plot)