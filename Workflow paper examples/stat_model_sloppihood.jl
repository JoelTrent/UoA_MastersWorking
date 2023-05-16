using DifferentialEquations, Random, Distributions, StaticArrays
include(joinpath("..", "JuLikelihood.jl"))

# ---------------------------------------------
# ---- User inputs in original 'x,y' param ----
# ---------------------------------------------
# parameter -> data dist (forward) mapping
distrib_xy(xy) = Normal(xy[1]*xy[2],sqrt(xy[1]*xy[2]*(1-xy[2]))) # 
# variables
varnames = Dict("x"=>"n", "y"=>"p")
θnames = [:n, :p]


# initial guess for optimisation
xy_initial =  [50, 0.3]# x (i.e. n) and y (i.e. p), starting guesses
# parameter bounds
xy_lower_bounds = [0.001,0.001]
xy_upper_bounds = [500,0.999]
# true parameter
xy_true = [100,0.2] #x,y, truth. N, p
N_samples = 10 # measurements of model
# generate data
#data = rand(distrib_xy(xy_true),N_samples)
data = (;samples=SA[21.9,22.3,12.8,16.4,16.4,20.3,16.2,20.0,19.7,24.4])

par_magnitudes = [100, 1]


# ---- use above to construct log likelihood in original parameterisation given (iid) data
function lnlike_xy(xy, data)
    return sum(logpdf.(distrib_xy(xy), data.samples))
end

function predictFunc_xy(xy, data, t=["n*p"]); [prod(xy)] end

model = initialiseLikelihoodModel(lnlike_xy, predictFunc_xy, data, θnames, xy_initial, xy_lower_bounds, xy_upper_bounds, par_magnitudes);

full_likelihood_sample!(model, 1000000, sample_type=LatinHypercubeSamples())

univariate_confidenceintervals!(model, [2], profile_type=LogLikelihood(), existing_profiles=:overwrite, num_points_in_interval=100)

bivariate_confidenceprofiles!(model, 200, profile_type=LogLikelihood(), method=BracketingMethodSimultaneous(), existing_profiles=:overwrite, save_internal_points=true)

using Plots
gr()

plots = plot_univariate_profiles(model, 0.05, 0.3, palette_to_use=:Spectral_8)
for i in eachindex(plots); display(plots[i]) end

plots = plot_bivariate_profiles(model, 0.2, 0.2, include_internal_points=true, markeralpha=0.9)
for i in eachindex(plots); display(plots[i]) end

df=2
llstar95=exp(-quantile(Chisq(df),0.95)/2)
llstar50=exp(-quantile(Chisq(df),0.50)/2)
llstar05=exp(-quantile(Chisq(df),0.05)/2)

using ConcaveHull

points = model.dim_samples_dict[1].points
ll = exp.(model.dim_samples_dict[1].ll)
inds = sample(1:length(ll), min(length(ll), 2000), replace=false, ordered=true)
boundary_plot = scatter(points[1,inds], points[2,inds], label=nothing, mw=0, ms=1, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4))

hull95 = concave_hull([eachcol(points)...], 300)
plot!(hull95, label="0.95")

hull50 = concave_hull([eachcol(points[:, ll .> llstar50])...], 300)
plot!(hull50, label="0.50")

hull05 = concave_hull([eachcol(points[:, ll .> llstar05])...], 30)
plot!(hull05, label="0.05")

display(boundary_plot)


n = length(hull95.vertices)
edges = zeros(Int, n, 2)
for i in 1:n-1; edges[i,:] .= i, i+1 end
edges[end,:] .= n, 1

nodes = transpose(reduce(hcat, hull95.vertices))
# nodes[:,1] .= nodes[:,1] ./ sqrt(prod(par_magnitudes))
# nodes[:,2] .= nodes[:,2] .* sqrt(prod(par_magnitudes))

xy, dist = polylabel(nodes, edges)

centroid_cell = get_centroid_cell(nodes, edges)

scatter!([xy[1]], [xy[2]], label="polylabel")
scatter!([centroid_cell.x], [centroid_cell.y], label="centroid")
###### LOG SPACE #####################################################################################
######################################################################################################
function lnlike_XY(XY, data)
    return sum(logpdf.(distrib_xy(exp.(XY)), data.samples))
end

function forward_parameter_transformLog(θ)
    return log.(θ)
end

function predictFunc_XY(xy, data, t=["n*p"]); [prod(exp.(xy))] end

newlb, newub = transformbounds(forward_parameter_transformLog, xy_lower_bounds, xy_upper_bounds, collect(1:2), Int[])
θnames = [:ln_n, :ln_p]
par_magnitudes = [2, 1]

model = initialiseLikelihoodModel(lnlike_XY, predictFunc_XY, data, θnames, log.(xy_initial), newlb, newub, par_magnitudes);

full_likelihood_sample!(model, 1000000, sample_type=LatinHypercubeSamples())

univariate_confidenceintervals!(model, profile_type=LogLikelihood(), existing_profiles=:overwrite, num_points_in_interval=300)

bivariate_confidenceprofiles!(model, 200, profile_type=LogLikelihood(), method=BracketingMethodSimultaneous(), existing_profiles=:overwrite, save_internal_points=true)

plots = plot_univariate_profiles(model, 0.05, 0.3, palette_to_use=:Spectral_8)
for i in eachindex(plots); display(plots[i]) end

plots = plot_bivariate_profiles(model, 0.2, 0.2, include_internal_points=true, markeralpha=0.9)
for i in eachindex(plots); display(plots[i]) end


points = model.dim_samples_dict[1].points
ll = exp.(model.dim_samples_dict[1].ll)
inds = sample(1:length(ll), min(length(ll), 2000), replace=false, ordered=true)
boundary_plot = scatter(points[1,inds], points[2,inds], label=nothing, mw=0, ms=1, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4))

hull95 = concave_hull([eachcol(points)...], 300)
plot!(hull95, label="0.95")

hull50 = concave_hull([eachcol(points[:, ll .> llstar50])...], 300)
plot!(hull50, label="0.50")

hull05 = concave_hull([eachcol(points[:, ll .> llstar05])...], 30)
plot!(hull05, label="0.05")
display(boundary_plot)









prediction_locations = ["n*p"]
generate_predictions_univariate!(model, prediction_locations, 1.0, profile_types=[LogLikelihood()])
generate_predictions_bivariate!(model, prediction_locations, 1.0, profile_types=[LogLikelihood()])
generate_predictions_dim_samples!(model, prediction_locations, 0.1)

union_plot = plot_predictions_union(model, prediction_locations, 2)
display(union_plot)

histogram(transpose(model.biv_predictions_dict[1].predictions), legend=false, normalize=:true, bins=30)
vline!(model.biv_predictions_dict[1].extrema, label="")

using StatsPlots
density(transpose(model.biv_predictions_dict[1].predictions), legend=false, bandwidth=0.1, trim=true)