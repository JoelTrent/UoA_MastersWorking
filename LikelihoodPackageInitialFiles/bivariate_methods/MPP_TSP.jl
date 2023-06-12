"""
    minimum_perimeter_polygon!(points::Array{<:Real,2})

Given a set of N points that define a boundary polygon in a 2 row, N column array, solve a minimum perimeter polygon TSP problem, reorder these points in place and return the path used (vertices in order of visitation). Uses [TravelingSalesmanHeuristics.jl](https://github.com/evanfields/TravelingSalesmanHeuristics.jl).
"""
function minimum_perimeter_polygon!(points::Array{<:Real,2})
    Dij = pairwise(Euclidean(), points, dims=2)
    path, _ = solve_tsp(Dij)
    points .= points[:,path[1:end-1]]
    return path
end