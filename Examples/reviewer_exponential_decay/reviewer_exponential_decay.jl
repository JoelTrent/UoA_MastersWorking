using Revise
using LikelihoodBasedProfileWiseAnalysis
using DifferentialEquations, Random, Distributions

using Plots
gr()

fileDirectory = joinpath("Examples", "reviewer_exponential_decay", "Plots")

# Workflow functions ##########################################################################
function DE!(dC, C, p, t)
    θ_1, θ_2 = p
    dC[1]= -(θ_1+θ_2) * C[1]
end

function odesolver(t, θ)
    tspan=(0.0, t[end])
    prob=ODEProblem(DE!, [C0], tspan, θ)
    sol=solve(prob, saveat=t, verbose=false)
    return sol[1,:]
end

function solvedmodel(t, θ)
    return exp.(-(θ[1] + θ[2]) .* t)
end

function loglhood(θ, data)
    y=odesolver(data.t, θ)
    # y=solvedmodel(data.t, θ)
    e=sum(loglikelihood(data.dist, data.yobs .- y))
    return e
end

# Data setup #################################################################################
# true parameters
C0=1.0
θ=[0.005, 0.005]; σ=0.05;
t=0:50:500
tt=0:5:500

# true data
ytrue = odesolver(t, θ)
solvedmodel(t, θ)
Random.seed!(12348)
# noisy data
yobs = ytrue + σ*randn(length(t))

# Named tuple of all data required within the likelihood function
data = (yobs=yobs, σ=σ, t=t, dist=Normal(0, σ))

# Bounds on model parameters #################################################################
θG = θ .* 1.0
lb = [-0.1, -0.1]
ub = [0.1, 0.1]
par_magnitudes = [1, 1]
θnames = [:θ1, :θ2]


optimsettings = create_OptimizationSettings(solve_kwargs=NamedTuple())
model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=optimsettings);

x = LinRange(lb[1], ub[1], 500)
y = LinRange(lb[2], ub[2], 500)
p_contour = contourf(x, y, (x,y)->loglhood([x,y], data) - model.core.maximisedmle, levels=collect(LinRange(-120, 0, 30)), fill=true, c=:dense, lw=0, colorbar_ticks=collect(LinRange(-150, 0, 10)), cbar=false, dpi=300)
xlabel!(p_contour, "θ1")
ylabel!(p_contour, "θ2")
title!(p_contour, "Lp_hat for full likelihood between -120 and 0. Darker is closer to 0 (MLE)", titlefontsize=10)
savefig(p_contour, joinpath(fileDirectory, "full_likelihood_contour.pdf"))
savefig(p_contour, joinpath(fileDirectory, "full_likelihood_contour.png"))

univariate_confidenceintervals!(model)
get_points_in_intervals!(model, 200, additional_width=1.0)

plots = plot_univariate_profiles(model, 0.1, 0.1)
for plot in plots
    plot!(plot, legend=:right)
    display(plot) 
end

savefig(plots[1], joinpath(fileDirectory, "theta1_widerbounds.pdf"))
savefig(plots[2], joinpath(fileDirectory, "theta2_widerbounds.pdf"))

lb = [-0.03, -0.03]
ub = [0.03, 0.03]
model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=optimsettings);

univariate_confidenceintervals!(model)
get_points_in_intervals!(model, 200)

plots = plot_univariate_profiles(model, 0.1, 0.1)
for plot in plots
    plot!(plot, legend=:right)
    display(plot)
end

savefig(plots[1], joinpath(fileDirectory, "theta1_smallerbounds.pdf"))
savefig(plots[2], joinpath(fileDirectory, "theta2_smallerbounds.pdf"))

plots = plot_univariate_profiles(model, 0.1, 0.1)
for plot in plots
    xlims!(plot, -0.01, 0.02)
    plot!(plot, legend=:right)
    display(plot)
end

savefig(plots[1], joinpath(fileDirectory, "theta1_smallerbounds_with_xlims.pdf"))
savefig(plots[2], joinpath(fileDirectory, "theta2_smallerbounds_with_xlims.pdf"))


p1 = plot(tt, solvedmodel(tt, θ), color=:turquoise1, xlabel="t", ylabel="x(t)", legend=false, lw=4, xlims=(0, 500))
scatter!(p1, t, yobs, legend=false, msw=0, ms=7, color=:darkorange, msa=:darkorange)
display(p1)
savefig(p1, joinpath(fileDirectory,"true_solution_and_observed_data.pdf"))

# IF MODEL ONLY HAD ONE PARAMETER
function solvedmodel(t, θ)
    return exp.(-(θ[1]) .* t)
end

function loglhood(θ, data)
    y=solvedmodel(data.t, θ)
    e = sum(loglikelihood(data.dist, data.yobs .- y))
    return e
end

model = initialise_LikelihoodModel(loglhood, data, [:θ3], [0.01], [0.0], [0.02], [1], optimizationsettings=optimsettings);
univariate_confidenceintervals!(model)
get_points_in_intervals!(model, 200, additional_width=0.2)

plots = plot_univariate_profiles(model, 0.1, 0.1)
for plot in plots
    plot!(plot, legend=:right)
    display(plot)
end

savefig(plots[1], joinpath(fileDirectory, "theta3.pdf"))