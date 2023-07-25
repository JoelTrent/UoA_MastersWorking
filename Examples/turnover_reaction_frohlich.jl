##############################################################################
######################### TURNOVER REACTION MODEL ############################
### FROM Frohlich et. al. 2014, Uncertainty Analysis for Non-identifiable ####
####### Dynamical Systems: Profile Likelihoods, Bootstrapping and More #######
##############################################################################

using Distributed
# if nprocs()==1; addprocs(10) end
using PlaceholderLikelihood
@everywhere using Revise
@everywhere using Random, Distributions
@everywhere using PlaceholderLikelihood

@everywhere function solvedmodel(t, θ)
    k1, k2, s = θ
    return ((s/k1)*k2) .* (1 .- exp.(-k2 .* t))
end

# Section 6: Define loglikelihood function
@everywhere function loglhood(θ, data)
    y = solvedmodel(data.t, θ)
    e = sum(loglikelihood(data.dist, data.yobs .- y))
    return e
end

# Data setup #################################################################################
# true parameters
k1=0.75; k2=0.25; s=1
# σ=0.2 #1.0
σ=1.0
t=LinRange(0,30,30)
θ=[k1, k2, s]

# true data
ytrue = solvedmodel(t, θ)

Random.seed!(12344)
Random.seed!(12350)
Random.seed!(13) # mle is pretty close to true parameter values
# Random.seed!(25)
# Random.seed!(45)
# Random.seed!(51)
# noisy data
yobs = ytrue + rand(Normal(0,σ), length(t))

data = (yobs=yobs, σ=σ, t=t, dist=Normal(0, σ))

# Bounds on model parameters #################################################################
θG = log10.([k1, k2, s])
lb = fill(-3, 3)
ub = fill(3, 3)
par_magnitudes = [1, 1, 1]

@everywhere function logged_predictFunc(Θ, data, t=data.t); solvedmodel(t, exp10.(Θ)) end
θnames = [:log10_k1, :log10_k2, :log10_s]

@everywhere function logged_loglhood(Θ, data)
    return loglhood(exp10.(Θ), data)
end

optim_settings = create_OptimizationSettings(solve_kwargs=(maxtime=15, ))

model = initialise_LikelihoodModel(logged_loglhood, logged_predictFunc, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals!(model, profile_type=EllipseApprox())
univariate_confidenceintervals!(model, profile_type=LogLikelihood())
get_points_in_intervals!(model, 200, additional_width=0.5)

bivariate_confidenceprofiles!(model, [[1,2], [2,3]], 30, method=AnalyticalEllipseMethod(0.0, 0.1))
bivariate_confidenceprofiles!(model, [[1,2], [2,3]], 30, method=IterativeBoundaryMethod(20, 0, 10, 0.0))
bivariate_confidenceprofiles!(model, [[1,2], [2,3]], 30, method=RadialRandomMethod(30), profile_type=EllipseApprox(), find_zero_atol=0.0)
bivariate_confidenceprofiles!(model, [[1,2], [2,3]], 30, method=RadialRandomMethod(30), profile_type=EllipseApproxAnalytical(), find_zero_atol=0.0)

dimensional_likelihood_samples!(model, 2, 10000)

# generate_predictions_bivariate!(model, collect(t), 0.1)

using Plots
gr()

plots = plot_univariate_profiles(model, 0.1, 0.1, θs_to_plot=[1,2,3])
for i in eachindex(plots)
    display(plots[i])
end

# plots = plot_bivariate_profiles(model, 0.05, 0.05, markeralpha=0.9, for_dim_samples=true)
# for i in eachindex(plots)
#     display(plots[i])
# end

plots = plot_bivariate_profiles_comparison(model, 0.05, 0.05, markeralpha=0.9, include_dim_samples=true)
for i in eachindex(plots)
    display(plots[i])
end

display(model.ellipse_MLE_approx.Hmle)
display(model.ellipse_MLE_approx.Γmle)

# inv(model.ellipse_MLE_approx.Hmle)
# pinv(model.ellipse_MLE_approx.Hmle)


xytoXY_sip(xy) = [(xy[1]/xy[2])*xy[3]; xy[3]/(xy[1]*xy[2]); xy[1]*xy[3]]
XYtoxy_sip(XY) = [(XY[1]/XY[2])^0.5; XY[3]/XY[1]; XY[3]/((XY[1]/XY[2])^0.5)]


# xytoXY_sip(xy) = [xy[3] / (xy[1] * xy[2]); xy[2]; xy[3]]
# XYtoxy_sip(XY) = [XY[2] / (XY[1]*XY[3]); XY[2]; XY[3]]

function loglhood_XYtoxy_sip(Θ,data); loglhood(XYtoxy_sip(Θ), data) end

function XYtoxy_sip_predictFunc(Θ, data, t=data.t); solvedmodel(t, XYtoxy_sip(Θ)) end


# θnames=[:k1s_over_k2, :s_over_k1k2, :k1s]
θnames=[:s_over_k1k2, :k2, :s]
θG = xytoXY_sip(exp10.([-0.5, -0.5, -0.5]))
lb, ub = fill(1e-3, 3), fill(1e2, 3)

model = initialise_LikelihoodModel(loglhood_XYtoxy_sip, XYtoxy_sip_predictFunc, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals!(model, profile_type=EllipseApprox())
univariate_confidenceintervals!(model, profile_type=LogLikelihood())
get_points_in_intervals!(model, 200, additional_width=0.5)


plots = plot_univariate_profiles_comparison(model, 0.1, 0.1, θs_to_plot=[1,2,3], palette_to_use=:Spectral_8)
for i in eachindex(plots)
    display(plots[i])
end

bivariate_confidenceprofiles!(model, 30, method=AnalyticalEllipseMethod(0.0, 0.1))
# bivariate_confidenceprofiles!(model, 30, method=SimultaneousMethod(), profile_type=EllipseApproxAnalytical())
bivariate_confidenceprofiles!(model, 30, method=IterativeBoundaryMethod(20, 0, 10, 0.0))
# bivariate_confidenceprofiles!(model, 30, method=RadialMLEMethod(0.0, 0.1), profile_type=EllipseApprox())

dimensional_likelihood_samples!(model, 2, 10000)

plots = plot_bivariate_profiles(model, 0.05, 0.05, markeralpha=0.9, for_dim_samples=true)
for i in eachindex(plots)
    display(plots[i])
end

plots = plot_bivariate_profiles_comparison(model, 0.05, 0.05, markeralpha=0.9, include_dim_samples=true)
for i in eachindex(plots)
    display(plots[i])
end
