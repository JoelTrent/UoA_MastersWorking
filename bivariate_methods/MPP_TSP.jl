function minimum_perimeter_polygon!(points::Array{<:Real,2})
    Dij = pairwise(Euclidean(), points, dims=2)
    path, _ = solve_tsp(Dij)
    points .= points[:,path[1:end-1]]
    return path
end