using Revise
using DifferentialEquations, Random, Distributions, StaticArrays
using PlaceholderLikelihood

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
xy_lower_bounds = [0.0001,0.0001]
xy_upper_bounds = [500.0, 1.0]
# true parameter
xy_true = [100.0,0.2] #x,y, truth. N, p
N_samples = 10 # measurements of model
# generate data
#data = rand(distrib_xy(xy_true),N_samples)
data = (;samples=SA[21.9,22.3,12.8,16.4,16.4,20.3,16.2,20.0,19.7,24.4])

par_magnitudes = [1000, 1]


# ---- use above to construct log likelihood in original parameterisation given (iid) data
function lnlike_xy(xy, data)
    return sum(logpdf.(distrib_xy(xy), data.samples))
end

function predictFunc_xy(xy, data, t=["n*p"]); [prod(xy)] end

model = initialiseLikelihoodModel(lnlike_xy, predictFunc_xy, data, θnames, xy_initial, xy_lower_bounds, xy_upper_bounds, par_magnitudes);

full_likelihood_sample!(model, 1000000, sample_type=LatinHypercubeSamples())

univariate_confidenceintervals!(model, profile_type=LogLikelihood(), existing_profiles=:overwrite, num_points_in_interval=50)

bivariate_confidenceprofiles!(model, 100, profile_type=EllipseApproxAnalytical(), method=RadialRandomMethod(3), existing_profiles=:overwrite, save_internal_points=true)

bivariate_confidenceprofiles!(model, 100, profile_type=LogLikelihood(), confidence_level=0.95, method=ContinuationMethod(1, 0.1, 0.0), existing_profiles=:overwrite, save_internal_points=true)

bivariate_confidenceprofiles!(model, 10, profile_type=LogLikelihood(), method=RadialMLEMethod(0.5, 1.0), confidence_level=0.95, existing_profiles=:overwrite, save_internal_points=true)

bivariate_confidenceprofiles!(model, 500, profile_type=LogLikelihood(), method=IterativeBoundaryMethod(20, 20, 20, 0.5, 0.01, use_ellipse=true), confidence_level=0.95, existing_profiles=:overwrite, save_internal_points=true)

# bivariate_confidenceprofiles!(model, 500, profile_type=EllipseApprox(), method=IterativeBoundaryMethod(20, 20, 20, 0.5, 0.01, use_ellipse=true), confidence_level=0.95, existing_profiles=:overwrite, save_internal_points=true)

bivariate_confidenceprofiles!(model, 500, profile_type=LogLikelihood(), method=RadialRandomMethod(5), confidence_level=0.95, existing_profiles=:overwrite, save_internal_points=true)

using Plots
gr()

plots = plot_univariate_profiles(model, 0.05, 0.3, palette_to_use=:Spectral_8)
for i in eachindex(plots); display(plots[i]) end

plots = plot_bivariate_profiles(model, 0.2, 0.2, include_internal_points=true, markeralpha=0.9)
for i in eachindex(plots); display(plots[i]) end

plot_bivariate_profiles_iterativeboundary_gif(model, 0.2, 0.2, markeralpha=0.5, save_folder="Workflow paper examples")

plots = plot_bivariate_profiles_comparison(model, 0.2, 0.2, compare_within_methods=true)
for i in eachindex(plots); display(plots[i]) end

df=2
llstar95=exp(-quantile(Chisq(df),0.95)/2)
llstar50=exp(-quantile(Chisq(df),0.50)/2)
llstar05=exp(-quantile(Chisq(df),0.05)/2)

using ConcaveHull

points = model.dim_samples_dict[1].points
ll = exp.(model.dim_samples_dict[1].ll)
inds = sample(1:length(ll), min(length(ll), 2000), replace=false, ordered=true)
boundary_plot = scatter(points[1,inds], points[2,inds], label=nothing, mw=0, ms=1, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(800,800))

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

# xy, dist = polylabel(log.(nodes), edges, 1.0)
# xy=exp.(xy)
# centroid_cell = get_centroid_cell(log.(nodes), edges)

# scatter!(hull95, label="hull points", opacity=0.1)
# scatter!([xy[1]], [xy[2]], label="polylabel")
# scatter!([centroid_cell.x], [centroid_cell.y], label="centroid")



using Meshes
points = model.biv_profiles_dict[3].confidence_boundary
minimum_perimeter_polygon!(points)
n = size(points,2)
mesh = SimpleMesh([(points[1,i], points[2,i]) for i in 1:n], [connect(tuple(1:n...))])

using Clustering
n_points=1000
internal_points = reduce(hcat,[point.coords for point in collect(sample(mesh, HomogeneousSampling(n_points)))])
R = kmeans(points, 10)
nclusters(R)
counts(R)

scatter(internal_points[1,:], internal_points[2,:], label=nothing, mw=0, ms=4, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(800,800))
scatter!(points[1,:], points[2,:], label=nothing, mw=0, ms=4, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(800,800))
scatter!(R.centers[1,:], R.centers[2,:], label=nothing, mw=0, ms=4, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(800,800))

# closeness of each kmean point to centre of shape, by the number of points above and below and to each side of it
obj = [ sum(abs.([sum(internal_points[1,:] .< R.centers[1,i]) , sum(internal_points[2,:] .< R.centers[2,i]) ]./ n_points .- 0.5))   for i in axes(R.centers, 2)]

# closeness of point to being a star point
function segment_plot(segment1, segment2; kwargs...)
    plt=scatter([segment1.vertices[1].coords[1], segment1.vertices[2].coords[1]], [segment1.vertices[1].coords[2], segment1.vertices[2].coords[2]], label="internal", ms=4, markerstrokewidth=0.2, opacity=1.0; kwargs...)
    scatter!([segment2.vertices[1].coords[1], segment2.vertices[2].coords[1]], [segment2.vertices[1].coords[2], segment2.vertices[2].coords[2]], label="edge", ms=4, markerstrokewidth=0.2, opacity=1.0)

    display(plt)
end

function star_obj(centers, points)
    n = size(points,2)
    obj = zeros(size(centers,2))
    for ci in axes(centers,2)
        c_point = centers[:,ci]
        for vi in 1:n
            intersects_polygon=false
            
            internal_segment = Segment(Point(c_point...), Point(points[:,vi]...))
            # println()
            # println("Vi=", vi)
            # println(internal_segment)

            # all vertex to vertex edges in the polygon that don't include vertex vi
            v1s = vcat(collect(1:vi-2), vcat(collect(vi+1:n-1), vi != 1 && vi != n ? [n] : Int[]))
            v2s = vcat(collect(2:vi-1), vcat(collect(vi+2:n), vi != 1 && vi != n ? [1] : Int[]))
            # println("v1s:", v1s)
            # println("v2s:", v2s)

            for ei in eachindex(v1s)
                edge_segment = Segment(Point(points[:,v1s[ei]]...), Point(points[:,v2s[ei]]...)) 
                # println(edge_segment)
                # segment_plot(internal_segment, edge_segment, xlim=[-0.1,4.1], ylim=[-0.1, 6.1])
                if intersection(internal_segment, edge_segment).type != IntersectionType(0) 
                    intersects_polygon=true
                    continue
                end
            end
            if !intersects_polygon
                obj[ci] +=1
            end
        end
    end
    return obj
end

objs = star_obj(R.centers, points)
star_centers = R.centers[:, findall(maximum(objs) .== objs)]

if size(star_centers, 2) == 1
    new_center = star_centers
else
    new_center = star_centers[:, argmin([ sum(abs.([sum(internal_points[1,:] .< star_centers[1,i]) , sum(internal_points[2,:] .< star_centers[2,i]) ]./ n_points .- 0.5))   for i in axes(star_centers, 2)])]
end

scatter!([new_center[1,1]], [new_center[2,1]], label=nothing, mw=0, ms=4, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(800,800))



smesh = mesh |> LambdaMuSmoothing(30,0.5,1.0)
smesh = mesh |> LaplaceSmoothing(4)
smesh = mesh |> TaubinSmoothing(30)


scatter(nodes[:,1], nodes[:,2], label=nothing, mw=0, ms=4, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(600,500))

smeshvertices = reduce(hcat,[point.coords for point in smesh.vertices])
scatter!(smeshvertices[1,:], smeshvertices[2,:], label=nothing, mw=0, ms=4, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(600,500))


using Distances, TravelingSalesmanHeuristics
Dij = pairwise(Euclidean(), points, dims=2)
path, cost = solve_tsp(Dij)

reordered_points = points[:,path]

n=size(reordered_points,2)-1
plot(reordered_points[1,:], reordered_points[2,:])
scatter(reordered_points[1,:], reordered_points[2,:], label=nothing, mw=0, ms=4, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(600,500))

mesh = SimpleMesh([(reordered_points[1,i], reordered_points[2,i]) for i in 1:n], [connect(tuple(1:n...))])
smesh = mesh |> LaplaceSmoothing(20)

smeshvertices = reduce(hcat,[point.coords for point in smesh.vertices])
scatter(smeshvertices[1,:], smeshvertices[2,:], label=nothing, mw=0, ms=4, markerstrokewidth=0.0, opacity=1.0,palette=palette(:Reds_4), size=(600,500))






###### LOG SPACE #####################################################################################
######################################################################################################
function lnlike_XY(XY, data)
    return sum(logpdf.(distrib_xy(exp.(XY)), data.samples))
end

function forward_parameter_transformLog(θ)
    return log.(θ)
end

function predictFunc_XY(xy, data, t=["n*p"]); [prod(exp.(xy))] end

transformbounds_NLopt(forward_parameter_transformLog, xy_lower_bounds, xy_upper_bounds)
newlb, newub = transformbounds(forward_parameter_transformLog, xy_lower_bounds, xy_upper_bounds, collect(1:2), Int[])
θnames = [:ln_n, :ln_p]
par_magnitudes = [2, 1]

model = initialiseLikelihoodModel(lnlike_XY, predictFunc_XY, data, θnames, forward_parameter_transformLog(xy_initial), newlb, newub, par_magnitudes);

full_likelihood_sample!(model, 1000000, sample_type=LatinHypercubeSamples())

univariate_confidenceintervals!(model, profile_type=LogLikelihood(), existing_profiles=:overwrite, num_points_in_interval=300)

bivariate_confidenceprofiles!(model, 200, profile_type=LogLikelihood(), method=RadialMLEMethod(), existing_profiles=:overwrite, save_internal_points=true)

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