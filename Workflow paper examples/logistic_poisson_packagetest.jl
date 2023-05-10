using Distributed
# addprocs(3)
@everywhere using DifferentialEquations, Random, Distributions
@everywhere include(joinpath("..", "JuLikelihood.jl"))


# Workflow functions ##########################################################################
# Section 2: Define ODE model
@everywhere function DE!(dC,C,p,t)
    λ,K=p
    dC[1]=λ*C[1]*(1.0-C[1]/K);
end

# Section 3: Solve ODE model
@everywhere function odesolver(t,λ,K,C0)
    p=(λ,K)
    tspan=(0.0,maximum(t));
    prob=ODEProblem(DE!,[C0],tspan,p);
    sol=solve(prob,saveat=t, verbose=false);
    return sol[1,:];
end

@everywhere function solvedmodel(t, a)
    return (a[2]*a[3]) ./ ((a[2]-a[3]) .* (exp.(-a[1] .* t)) .+ a[3])
end

# Section 4: Define function to solve ODE model 
@everywhere function ODEmodel(t,a)
    y=odesolver(t,a[1],a[2],a[3])
    return y
end

# Section 6: Define loglikelihood function
@everywhere function likelihoodFunc(a, data)
    # y=ODEmodel(data.t,a)
    y=solvedmodel(data.t, a)
    e=0
    data_dists=Poisson.(y)
    e+=sum(loglikelihood.(data_dists, data.yobs))
    return e
end

@everywhere function predictFunc(θ, data, t=data.t); solvedmodel(t, θ) end


# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000;
tt=0:5:1000;

# true data
ytrue=ODEmodel(t,[λ,K,C0])

Random.seed!(12348)
# noisy data
data_dists=Poisson.(ytrue)
yobs=rand.(data_dists)

# Named tuple of all data required within the likelihood function
data = (yobs=yobs, t=t)

# Bounds on model parameters #################################################################
λmin, λmax = (0.001, 0.05)
Kmin, Kmax = (50., 150.)
C0min, C0max = (0.01, 50.)

θG = [λ,K,C0]
lb=[λmin,Kmin,C0min]
ub=[λmax,Kmax,C0max]
par_magnitudes = [0.01, 20, 10]

θnames = [:λ, :K, :C0]

model = initialiseLikelihoodModel(likelihoodFunc, predictFunc, data, θnames, θG, lb, ub, par_magnitudes);

full_likelihood_sample!(model, 1000000, sample_type=LatinHypercubeSamples())
full_likelihood_sample!(model, 1000000, sample_type=UniformRandomSamples())
univariate_confidenceintervals!(model, profile_type=LogLikelihood())
univariate_confidenceintervals!(model, profile_type=EllipseApprox())
univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
get_points_in_interval!(model, 100, additional_width=0.3)

bivariate_confidenceprofiles!(model, 200, method=AnalyticalEllipseMethod())
bivariate_confidenceprofiles!(model, 200, profile_type=EllipseApprox(), method=ContinuationMethod(0.1, 1, 0.0), save_internal_points=true)
bivariate_confidenceprofiles!(model, 400, profile_type=LogLikelihood(), method=BracketingMethodRadial(3), save_internal_points=true)

prediction_locations = collect(LinRange(t[1], t[end], 50));
generate_predictions_univariate!(model, prediction_locations, 1.0, profile_types=[LogLikelihood()])
generate_predictions_bivariate!(model, prediction_locations, 1.0, profile_types=[LogLikelihood()])
generate_predictions_dim_samples!(model, prediction_locations, 0.1)

using Plots
gr()

p1 = plot(prediction_locations, predictFunc(model.core.θmle, model.core.data, prediction_locations), color=:turquoise1, xlabel="t", ylabel="C(t)",
            legend=false, lw=4, xlims=(0,1100), ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100])

p1 = scatter!(data.t, data.yobs, legend=false, msw=0, ms=7,
            color=:darkorange, msa=:darkorange)
display(p1)

plots = plot_univariate_profiles_comparison(model, 0.2, 0.2, profile_types=[EllipseApproxAnalytical(), EllipseApprox(), LogLikelihood()], palette_to_use=:Spectral_8)
for i in eachindex(plots); display(plots[i]) end

plots = plot_predictions_individual(model, prediction_locations, 1, ylims=[0,120], profile_types=[LogLikelihood()])
for i in eachindex(plots); display(plots[i]) end

union_plot = plot_predictions_union(model, prediction_locations, 1, ylims=[0,120], compare_to_full_sample_type=LatinHypercubeSamples())
display(union_plot)

plots = plot_bivariate_profiles_comparison(model, 0.2, 0.2, compare_within_methods=false)
for i in eachindex(plots); display(plots[i]) end

plots = plot_predictions_individual(model, prediction_locations, 2, ylims=[0,120], profile_types=[LogLikelihood()])
for i in eachindex(plots); display(plots[i]) end

union_plot = plot_predictions_union(model, prediction_locations, 2, ylims=[0,120], compare_to_full_sample_type=LatinHypercubeSamples())
display(union_plot)


# REMOVE WORKER PROCESSORS WHEN FINISHED #######################################
rmprocs(workers())
################################################################################