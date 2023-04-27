function valid_points(model::LikelihoodModel, 
                        grid::Base.Iterators.ProductIterator,
                        grid_size::Int,
                        confidence_level::Float64, 
                        num_dims::Int,
                        use_threads::Bool)
    valid_point = falses(grid_size)
    ll_values = zeros(grid_size)
    target_ll = get_target_loglikelihood(model, confidence_level,
                                         LogLikelihood(), num_dims)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=grid_size) 
    @floop ex for (i, point) in enumerate(grid)
        ll_values[i] = model.core.loglikefunction(point, model.core.data)-target_ll
        if ll_values[i] >= 0
            valid_point[i] = true
        end
    end

    points = zeros(model.core.num_pars, sum(valid_point))
    j=1
    for (i, point) in enumerate(grid)
        if (ll_values[i]) > 0 
            points[:,j] .= point
            j+=1
        end
    end

    return points, ll_values[valid_point]
end

function valid_points(model::LikelihoodModel, 
                        grid::Matrix{Float64},
                        grid_size::Int,
                        confidence_level::Float64, 
                        num_dims::Int,
                        use_threads::Bool)
    valid_point = falses(grid_size)
    ll_values = zeros(grid_size)
    target_ll = get_target_loglikelihood(model, confidence_level, LogLikelihood(), num_dims)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=grid_size) 
    @floop ex for i in 1:grid_size
        ll_values[i] = model.core.loglikefunction(grid[:,i], model.core.data)-target_ll
        if ll_values[i] >= 0
            valid_point[i] = true
        end
    end

    return grid[:,valid_point], ll_values[valid_point]
end

function check_if_bounds_supplied(model::LikelihoodModel,
                                    lb::Vector,
                                    ub::Vector)
    if isempty(lb)
        lb = model.core.θlb
    else
        length(lb) == model.core.num_pars || throw(ArgumentError(string("lb must be of length ", num_dims)))
    end
    if isempty(ub)
        ub = model.core.θub
    else
        length(ub) == model.core.num_pars  || throw(ArgumentError(string("ub must be of length ", num_dims)))
    end
    return lb, ub
end

# Uniform grids
function uniform_grid(model::LikelihoodModel,
                        confidence_level::Float64,
                        points_per_dimension::Union{Int, Vector{Int}},
                        lb::Vector=[],
                        ub::Vector=[];
                        use_threads=true
                        )

    num_dims = model.core.num_pars
    if points_per_dimension isa Vector{Int}
        num_dims == length(points_per_dimension) || throw(ArgumentError(string("points_per_dimension must be of length ", num_dims)))
    else
        points_per_dimension = fill(points_per_dimension, num_dims)
    end

    lb, ub = check_if_bounds_supplied(model, lb, ub)

    ranges = LinRange.(lb, ub, points_per_dimension)
    grid = Iterators.product(ranges...)
    grid_size = prod(points_per_dimension)

    return valid_points(model, grid, grid_size, confidence_level, num_dims, use_threads)
end

function uniform_random(model::LikelihoodModel,
                        confidence_level::Float64,
                        num_points::Int,
                        lb::Vector=[],
                        ub::Vector=[];
                        use_threads::Bool=true
                        )

    num_dims = model.core.num_pars
    num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))

    lb, ub = check_if_bounds_supplied(model, lb, ub)

    grid = zeros(num_dims, num_points)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_dims) 
    @floop ex for dim in 1:num_dims
        grid[dim, :] .= rand(Uniform(lb[dim], ub[dim]), num_points)
    end

    return valid_points(model, grid, num_points, confidence_level, num_dims, use_threads)
end


# LatinHypercubeSampling
function LHS(model::LikelihoodModel,
    confidence_level::Float64,
    num_points::Int,
    lb::Vector=[],
    ub::Vector=[];
    use_threads::Bool=true,
    kwargs...)
    
    num_dims = model.core.num_pars
    num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
    
    lb, ub = check_if_bounds_supplied(model, lb, ub)
    
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    # grid = permutedims(scaleLHC(LHCoptim(num_points, num_dims, num_gens; kwargs...)[1], scale_range))
    
    return valid_points(model, grid, num_points, confidence_level, num_dims, use_threads)
end

@time uniform_grid(model, 0.95, 30; use_threads=true)
@time uniform_random(model, 0.95, 27000; use_threads=true)
@time LHS(model, 0.95, 27000)