function dimensional_optimiser!(θs, p, targetll)
    
    function fun(λ)
        θs[p.λindexes] = λ
        return p.consistent.loglikefunction(θs, p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-targetll
    θs[p.λindexes] .= xopt
    return llb
end

function init_dimensional_parameters(model::LikelihoodModel, 
                                        θindexes::Vector{Int},
                                        num_dims::Int)

    λindexes = setdiff(1:model.core.num_pars, θindexes)
    newLb     = zeros(model.core.num_pars-num_dims) 
    newUb     = zeros(model.core.num_pars-num_dims)
    initGuess = zeros(model.core.num_pars-num_dims)

    newLb .= model.core.θlb[λindexes]
    newUb .= model.core.θub[λindexes]
    initGuess .= model.core.θmle[λindexes]

    return newLb, newUb, initGuess, λindexes
end

function valid_points(model::LikelihoodModel,
                        p::NamedTuple, 
                        grid::Base.Iterators.ProductIterator,
                        grid_size::Int,
                        confidence_level::Float64, 
                        num_dims::Int,
                        use_threads::Bool)
    valid_point = falses(grid_size)
    ll_values = zeros(grid_size)
    targetll = get_target_loglikelihood(model, confidence_level,
                                         LogLikelihood(), num_dims)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=grid_size) 
    @floop ex for (i, point) in enumerate(grid)
        ll_values[i] = dimensional_optimiser!(point, p, targetll)
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

    valid_ll_values = ll_values[valid_point]
    valid_ll_values .= valid_ll_values .+ get_target_loglikelihood(model, confidence_level,
                                                        EllipseApproxAnalytical(), num_dims)

    return SampledConfidenceStruct(points, valid_ll_values)
end

function valid_points(model::LikelihoodModel, 
                        p::NamedTuple, 
                        grid::Matrix{Float64},
                        grid_size::Int,
                        confidence_level::Float64, 
                        num_dims::Int,
                        use_threads::Bool)

    valid_point = falses(grid_size)
    ll_values = zeros(grid_size)
    targetll = get_target_loglikelihood(model, confidence_level, LogLikelihood(), num_dims)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=grid_size) 
    @floop ex for i in axes(grid,2)
        ll_values[i] = dimensional_optimiser!(@view(grid[:,i]), p, targetll)
        if ll_values[i] >= 0
            valid_point[i] = true
        end
    end

    valid_ll_values = ll_values[valid_point]
    valid_ll_values .= valid_ll_values .+ get_target_loglikelihood(model, confidence_level,
                                                        EllipseApproxAnalytical(), num_dims)

    return SampledConfidenceStruct(grid[:,valid_point], valid_ll_values)
end

function check_if_bounds_supplied(model::LikelihoodModel,
                                    θindexes::Vector{Int},
                                    lb::Vector,
                                    ub::Vector)
    if isempty(lb)
        lb = model.core.θlb[θindexes]
    else
        length(lb) == length(θindexes) || throw(ArgumentError(string("lb must be of length ", length(θindexes))))
    end
    if isempty(ub)
        ub = model.core.θub[θindexes]
    else
        length(ub) == length(θindexes)  || throw(ArgumentError(string("ub must be of length ", length(θindexes))))
    end
    return lb, ub
end

# Uniform grids
function uniform_grid(model::LikelihoodModel,
                        θindexes::Vector{Int},
                        confidence_level::Float64,
                        points_per_dimension::Union{Int, Vector{Int}},
                        lb::Vector=[],
                        ub::Vector=[];
                        use_threads=true,
                        arguments_checked::Bool=false)

    num_dims = length(θindexes)

    if points_per_dimension isa Vector{Int}
        num_dims == length(points_per_dimension) || throw(ArgumentError(string("points_per_dimension must be of length ", num_dims)))
        all(points_per_dimension .> 0) || throw(DomainError("points_per_dimension must be a vector of strictly positive integers"))
    else
        points_per_dimension > 0 || throw(DomainError("points_per_dimension must be a strictly positive integer"))
        points_per_dimension = fill(points_per_dimension, num_dims)
    end
    newLb, newUb, initGuess, λindexes = init_dimensional_parameters(model, θindexes, num_dims)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), num_dims)
    p=(θindexes=θindexes, newLb=newLb, newUb=newUb, initGuess=initGuess,
        λindexes=λindexes, consistent=consistent)

    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, θindexes, lb, ub)

    ranges = LinRange.(lb, ub, points_per_dimension)
    grid_iterator = Iterators.product(ranges...)
    grid_size = prod(points_per_dimension)

    grid = zeros(model.core.num_pars, grid_size)
    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=grid_size) 
    @floop ex for (i, point) in enumerate(grid_iterator)
        grid[θindexes, i] .= point
    end

    return valid_points(model, p, grid, grid_size, confidence_level, num_dims, use_threads)
end

function uniform_random(model::LikelihoodModel,
                        θindexes::Vector{Int},
                        confidence_level::Float64,
                        num_points::Int,
                        lb::Vector=[],
                        ub::Vector=[];
                        use_threads::Bool=true,
                        arguments_checked::Bool=false)

    num_dims = length(θindexes)
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
    end
    newLb, newUb, initGuess, λindexes = init_dimensional_parameters(model, θindexes, num_dims)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), num_dims)
    p=(θindexes=θindexes, newLb=newLb, newUb=newUb, initGuess=initGuess,
        λindexes=λindexes, consistent=consistent)
    
    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, θindexes, lb, ub)

    grid = zeros(model.core.num_pars, num_points)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_dims) 
    @floop ex for dim in 1:num_dims
        grid[θindexes[dim], :] .= rand(Uniform(lb[dim], ub[dim]), num_points)
    end

    return valid_points(model, p, grid, num_points, confidence_level, num_dims, use_threads)
end

# LatinHypercubeSampling
function LHS(model::LikelihoodModel,
            θindexes::Vector{Int},
            confidence_level::Float64,
            num_points::Int,
            lb::Vector=[],
            ub::Vector=[];
            use_threads::Bool=true,
            arguments_checked::Bool=false)
    
    num_dims = length(θindexes)
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
        lb, ub = check_if_bounds_supplied(model, θindexes, lb, ub)
    end
    newLb, newUb, initGuess, λindexes = init_dimensional_parameters(model, θindexes, num_dims)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), num_dims)
    p=(θindexes=θindexes, newLb=newLb, newUb=newUb, initGuess=initGuess,
        λindexes=λindexes, consistent=consistent)

    grid = zeros(model.core.num_pars, num_points)
    
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid[θindexes, :] = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    # grid = permutedims(scaleLHC(LHCoptim(num_points, num_dims, num_gens; kwargs...)[1], scale_range))
    
    return valid_points(model, p, grid, num_points, confidence_level, num_dims, use_threads)
end

# function full_likelihood_sample(model::LikelihoodModel,
#                                     confidence_level::Float64,
#                                     num_points::Union{Int, Vector{Int}},
#                                     sample_type::AbstractSampleType,
#                                     lb::Vector,
#                                     ub::Vector,
#                                     use_threads::Bool)

#     if sample_type isa UniformGridSamples
#         sample_struct = uniform_grid(model, confidence_level, num_points, lb, ub;
#                                         use_threads=use_threads, arguments_checked=true)
#     elseif sample_type isa UniformRandomSamples
#         sample_struct = uniform_random(model, confidence_level, num_points, lb, ub;             
#                                         use_threads=use_threads, arguments_checked=true)
#     elseif sample_type isa LatinHypercubeSamples
#         sample_struct = LHS(model, confidence_level, num_points, lb, ub;
#                             use_threads=use_threads, arguments_checked=true)
#     end
#     return sample_struct
# end

# function full_likelihood_sample!(model::LikelihoodModel,
#                                     num_points_to_sample::Union{Int, Vector{Int}};
#                                     confidence_level::Float64=0.95,
#                                     sample_type::AbstractSampleType=LatinHypercubeSamples(),
#                                     lb::Vector=[],
#                                     ub::Vector=[],
#                                     use_threads::Bool=true,
#                                     existing_profiles::Symbol=:overwrite)

#     if num_points_to_sample isa Int
#         num_points_to_sample > 0 || throw(DomainError("num_points_to_sample must be a strictly positive integer"))
#     end
#     existing_profiles ∈ [:ignore, :overwrite] || throw(ArgumentError("existing_profiles can only take value :ignore or :overwrite"))
#     lb, ub = check_if_bounds_supplied(model, lb, ub)

#     init_full_samples_row_exists!(model, sample_type)
#     # check if sample has already been evaluated
#     requires_overwrite = model.full_samples_row_exists[sample_type][confidence_level] != 0
#     if existing_profiles == :ignore && requires_overwrite; return nothing end

#     sample_struct = full_likelihood_sample(model, confidence_level, num_points_to_sample, sample_type, lb, ub, use_threads)
#     num_points_kept = length(sample_struct.ll)
    
#     if num_points_kept == 0
#         @warn "no sampled points were in the confidence region of the full likelihood within the supplied bounds: try increasing num_points_to_sample or changing the bounds"
#         return nothing
#     end

#     if requires_overwrite
#         row_ind = model.full_samples_row_exists[sample_type][confidence_level]
#     else
#         model.num_full_samples += 1
#         row_ind = model.num_full_samples * 1
#         if (model.num_full_samples - nrow(model.full_samples_df)) > 0
#             add_full_samples_rows!(model, 1)
#         end
#         model.full_samples_row_exists[sample_type][confidence_level] = row_ind
#     end

#     model.full_samples_dict[row_ind] = sample_struct
#     set_full_samples_row!(model, row_ind, true, confidence_level, sample_type, num_points_kept)

#     return nothing
# end