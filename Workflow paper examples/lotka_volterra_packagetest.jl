using Distributed
# addprocs(6)
@everywhere using DifferentialEquations, Random, Distributions
@everywhere include(joinpath("..", "JuLikelihood.jl"))

# Workflow functions ##########################################################################
# Section 2: Define ODE model
@everywhere function DE!(dC,C,p,t)
    α,β=p
    dC[1]=α*C[1]-C[1]*C[2];
    dC[2]=β*C[1]*C[2]-C[2];
end

# Section 3: Solve ODE model
@everywhere function odesolver(t,α,β,C01,C02)
    p=(α,β)
    C0=[C01,C02]
    tspan=(0.0,maximum(t))
    prob=ODEProblem(DE!,C0,tspan,p)
    sol=solve(prob,saveat=t,verbose=false);
    cc1=sol[1,:]
    cc2=sol[2,:]
    tt=sol.t[:]
    return cc1,cc2
end

# Section 4: Define function to solve ODE model 
@everywhere function ODEmodel(t,a)
    (x,y)=odesolver(t,a[1],a[2],a[3],a[4])
    return x,y
end

# Section 6: Define loglikelihood function
@everywhere function likelihoodFunc(a,data)
    (x,y)=ODEmodel(data.t,a)
    e=0.0
    f=0.0
    e+=loglikelihood(data.dist, data.xobs[data.like_range] .- x[data.like_range])  
    f+=loglikelihood(data.dist, data.yobs[data.like_range] .- y[data.like_range])
    return e+f
end


# Data setup #################################################################################
# true parameters
α = 0.9; β=1.1; x0=0.8; y0=0.3; 
t=LinRange(0,10,21);
tt=LinRange(0,10,2001)
σ=0.2

# true data
(xtrue, ytrue) = ODEmodel(t,[α,β,x0,y0]);

Random.seed!(12348)
# noisy data
xobs = xtrue+σ*randn(length(t));
yobs = ytrue+σ*randn(length(t));

# Named tuple of all data required within the likelihood function
data = (xobs=xobs, yobs=yobs, t=t, dist=Normal(0, σ), like_range=1:15)

# Bounds on model parameters #################################################################
αmin, αmax   = (0.7, 1.2)
βmin, βmax   = (0.7, 1.4)
x0min, x0max = (0.5, 1.2)
y0min, y0max = (0.1, 0.5)

θG = [α,β,x0,y0]
lb = [αmin,βmin,x0min,y0min]
ub = [αmax,βmax,x0max,y0max]
par_magnitudes = [1,1,1,1]

@everywhere function predictFunc(θ, data, t=data.t); hcat(ODEmodel(t, θ)...) end
θnames = [:α, :β, :x0, :y0]


model = initialiseLikelihoodModel(likelihoodFunc, predictFunc, data, θnames, θG, lb, ub, par_magnitudes);

full_likelihood_sample!(model, 100000, sample_type=LatinHypercubeSamples())
full_likelihood_sample!(model, 100000, sample_type=UniformRandomSamples())
univariate_confidenceintervals!(model, profile_type=LogLikelihood())
univariate_confidenceintervals!(model, profile_type=EllipseApprox())
univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
get_points_in_interval!(model, 100, additional_width=0.3)

bivariate_confidenceprofiles!(model, 200, method=AnalyticalEllipseMethod())
bivariate_confidenceprofiles!(model, 200, profile_type=EllipseApprox(), method=ContinuationMethod(0.1, 1, 0.0), save_internal_points=true)
bivariate_confidenceprofiles!(model, 200, profile_type=LogLikelihood(), method=BracketingMethodFix1Axis(), save_internal_points=true, existing_profiles=:overwrite)

prediction_locations = collect(LinRange(t[1], t[end], 50));
generate_predictions_univariate!(model, prediction_locations, 1.0, profile_types=[LogLikelihood()])
generate_predictions_bivariate!(model, prediction_locations, 1.0, profile_types=[LogLikelihood()])
generate_predictions_dim_samples!(model, prediction_locations, 0.1)

using Plots
gr()

# p1 = plot(prediction_locations, predictFunc(model.core.θmle, model.core.data, prediction_locations), color=:turquoise1, xlabel="t", ylabel="C(t)",
#             legend=false, lw=4, xlims=(0,1100), ylims=(0,120),
#             xticks=[0,500,1000], yticks=[0,50,100])

# p1 = scatter!(data.t, data.yobs, legend=false, msw=0, ms=7,
#             color=:darkorange, msa=:darkorange)
# display(p1)

plots = plot_univariate_profiles_comparison(model, 0.2, 0.2, profile_types=[EllipseApproxAnalytical(), EllipseApprox(), LogLikelihood()], palette_to_use=:Spectral_8)
for i in eachindex(plots); display(plots[i]) end

plots = plot_predictions_individual(model, prediction_locations, 1, ylims=[0,2.5], profile_types=[LogLikelihood()])
for i in eachindex(plots); display(plots[i]) end

union_plot = plot_predictions_union(model, prediction_locations, 1, ylims=[0,2.5], compare_to_full_sample_type=LatinHypercubeSamples(), ylabel=["a(t)", "b(t)"])
display(union_plot)

plots = plot_bivariate_profiles_comparison(model, 0.2, 0.2, compare_within_methods=false)
for i in eachindex(plots); display(plots[i]) end

plots = plot_predictions_individual(model, prediction_locations, 2, ylims=[0,2.5], profile_types=[LogLikelihood()])
for i in eachindex(plots); display(plots[i]) end

union_plot = plot_predictions_union(model, prediction_locations, 2, ylims=[0,2.5], compare_to_full_sample_type=LatinHypercubeSamples())
display(union_plot)

scatter!(union_plot[1], data.t, data.xobs, label="", legend=false) 
scatter!(union_plot[2], data.t, data.yobs, label="", legend=false) 


plots = plot_predictions_sampled(model, prediction_locations)
for i in eachindex(plots); display(plots[i]) end
println()

# REMOVE WORKER PROCESSORS WHEN FINISHED #######################################
rmprocs(workers())
################################################################################