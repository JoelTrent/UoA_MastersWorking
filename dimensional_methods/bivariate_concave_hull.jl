function bivariate_concave_hull(points::AbstractArray{Float64}, ll::Vector{<:Float64}, min_proportion_to_keep::Real, min_scaling_from_desired_ll::Real, target_ll::Float64,
    sample_type::AbstractSampleType)

    (0.0 < min_proportion_to_keep && min_proportion_to_keep ≤ 1.0) || throw(DomainError("min_proportion_to_keep must be in the interval (0.0,1.0]"))

    0.0 < min_scaling_from_desired_ll || throw(DomainError("min_percent_from_desired_ll must be strictly greater than zero"))

    num_points = length(ll)
    num_points < 3 && return points

    min_num_to_keep = min_proportion_to_keep == 1.0 ? num_points : convert(Int, round(num_points*min_proportion_to_keep, RoundDown))
    sortvec = sortperm(ll)
    sorted_ll = ll[sortvec]

    allowed_ll = target_ll - target_ll*min_scaling_from_desired_ll

    allowed_ind = findlast(sorted_ll .<= allowed_ll)
    allowed_ind = isnothing(allowed_ind) ? 0 : allowed_ind

    num_to_keep = max(3, min(min_num_to_keep, allowed_ind))
    points_kept = points[:, sortvec[1:num_to_keep]]

    if sample_type isa UniformGridSamples
        k = max(3, convert(Int, round(num_to_keep^(1.0/2.0), RoundDown)))
    else
        k = max(3, convert(Int, round(num_to_keep^(1.0/1.6), RoundDown)))
    end
    hull = concave_hull([eachcol(points_kept)...], k)

    hull_points = reduce(hcat, hull.vertices)

    # TSP the points to make sure they're in the right order
    minimum_perimeter_polygon!(hull_points)

    # additional concave hull iter to get rid of any accidental non boundary points
    hull = concave_hull([eachcol(hull_points)...])
    hull_points = reduce(hcat, hull.vertices)

    # TSP the points again
    minimum_perimeter_polygon!(hull_points)

    return hull_points
end

function bivariate_concave_hull(sampled_struct::AbstractSampledConfidenceStruct, θindices::Vector{Int},
    min_proportion_to_keep::Real, min_scaling_from_desired_ll::Real, target_ll::Float64, sample_type::AbstractSampleType)

    length(θindices) == 2 || throw(ArgumentError("θindices must have length 2; this function estimates the concave hull of a 2D point cloud"))

    return bivariate_concave_hull(sampled_struct.points[θindices, :], sampled_struct.ll, min_proportion_to_keep, min_scaling_from_desired_ll, target_ll, sample_type)
end