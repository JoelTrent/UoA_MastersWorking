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
                                        θindices::Vector{Int},
                                        num_dims::Int)

    λindexes = setdiff(1:model.core.num_pars, θindices)
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
    end
    valid_point .= ll_values .≥ 0.0

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

    return points, valid_ll_values
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
    end
    valid_point .= ll_values .≥ 0.0

    valid_ll_values = ll_values[valid_point]
    valid_ll_values .= valid_ll_values .+ get_target_loglikelihood(model, confidence_level,
                                                        EllipseApproxAnalytical(), num_dims)

    return grid[:,valid_point], valid_ll_values
end

function check_if_bounds_supplied(model::LikelihoodModel,
                                    θindices::Vector{Int},
                                    lb::AbstractVector{<:Real},
                                    ub::AbstractVector{<:Real})
    if isempty(lb)
        lb = model.core.θlb[θindices]
    else
        length(lb) == length(θindices) || throw(ArgumentError(string("lb must be of length ", length(θindices))))
    end
    if isempty(ub)
        ub = model.core.θub[θindices]
    else
        length(ub) == length(θindices)  || throw(ArgumentError(string("ub must be of length ", length(θindices))))
    end
    return lb, ub
end

# Uniform grids
function uniform_grid(model::LikelihoodModel,
                        θindices::Vector{Int},
                        points_per_dimension::Union{Int, Vector{Int}},
                        confidence_level::Float64,
                        lb::AbstractVector{<:Real}=Float64[],
                        ub::AbstractVector{<:Real}=Float64[];
                        use_threads=true,
                        arguments_checked::Bool=false)

    num_dims = length(θindices); num_dims > 0 || throw(ArgumentError("θindices must not be empty"))
    if points_per_dimension isa Vector{Int}
        num_dims == length(points_per_dimension) || throw(ArgumentError(string("points_per_dimension must be of length ", num_dims)))
        all(points_per_dimension .> 0) || throw(DomainError("points_per_dimension must be a vector of strictly positive integers"))
    else
        points_per_dimension > 0 || throw(DomainError("points_per_dimension must be a strictly positive integer"))
        points_per_dimension = fill(points_per_dimension, num_dims)
    end

    newLb, newUb, initGuess, λindexes = init_dimensional_parameters(model, θindices, num_dims)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), num_dims)
    p=(θindices=θindices, newLb=newLb, newUb=newUb, initGuess=initGuess,
        λindexes=λindexes, consistent=consistent)

    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, θindices, lb, ub)

    ranges = LinRange.(lb, ub, points_per_dimension)
    grid_iterator = Iterators.product(ranges...)
    grid_size = prod(points_per_dimension)

    grid = zeros(model.core.num_pars, grid_size)
    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=grid_size) 
    @floop ex for (i, point) in enumerate(grid_iterator)
        grid[θindices, i] .= point
    end

    pnts, lls = valid_points(model, p, grid, grid_size, confidence_level, num_dims, use_threads)
    return SampledConfidenceStruct(pnts, lls)
end

function uniform_random_blocks(model::LikelihoodModel,
                        θindices::Vector{Int},
                        num_points::Int,
                        confidence_level::Float64,
                        lb::AbstractVector{<:Real}=Float64[],
                        ub::AbstractVector{<:Real}=Float64[];
                        block_size=20000,
                        use_threads::Bool=true,
                        arguments_checked::Bool=false)

    num_dims = length(θindices); num_dims > 0 || throw(ArgumentError("θindices must not be empty"))
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
    end
    newLb, newUb, initGuess, λindexes = init_dimensional_parameters(model, θindices, num_dims)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), num_dims)
    p=(θindices=θindices, newLb=newLb, newUb=newUb, initGuess=initGuess,
        λindexes=λindexes, consistent=consistent)
    
    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, θindices, lb, ub)

    num_blocks = div(num_points, block_size)
    remainder = mod(num_points, block_size)

    valid_grid = zeros(model.core.num_pars, 0)
    valid_ll_values = zeros(0)

    block_sizes = [block_size for _ in 1:num_blocks]
    if remainder > 0; append!(block_sizes, remainder) end
    cumulative_sizes = cumsum(block_sizes)
    preallocate_j = findfirst((cumulative_sizes ./ num_points) .≥ 0.05)

    current_len = 0
    estimated_size = num_points
    grid = zeros(model.core.num_pars, block_size)
    for (i, block) in enumerate(block_sizes)

        if block < block_size; grid = zeros(model.core.num_pars, block) end

        ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_dims) 
        @floop ex for dim in 1:num_dims
            grid[θindices[dim], :] .= rand(Uniform(lb[dim], ub[dim]), block)
        end

        pnts, lls = valid_points(model, p, grid, block, confidence_level, num_dims, use_threads)

        if i ≤ preallocate_j
            valid_grid = hcat(valid_grid, @view(pnts[:,:]))
            valid_ll_values = vcat(valid_ll_values, @view(lls[:]))
            continue
        end
        if (preallocate_j + 1) == i
            # estimate preallocation size
            current_len = length(valid_ll_values)
            accept_rate = current_len/cumulative_sizes[preallocate_j]
            safety_factor = 1.25
            estimated_size = convert(Int, round(accept_rate * num_points * safety_factor, RoundDown))

            valid_grid = hcat(valid_grid, zeros(model.core.num_pars, estimated_size-current_len))
            resize!(valid_ll_values, estimated_size)
        end

        num_to_add = length(lls)
        if (current_len + num_to_add) ≤ estimated_size
            rnge = current_len+1:current_len+num_to_add
            valid_grid[:, rnge] .= pnts
            valid_ll_values[rnge] .= lls
            current_len += num_to_add
        else
            # do something here
            continue
        end
    end

    return SampledConfidenceStruct(valid_grid[:,1:current_len], valid_ll_values[1:current_len])
end

function uniform_random(model::LikelihoodModel,
                        θindices::Vector{Int},
                        num_points::Int,
                        confidence_level::Float64,
                        lb::AbstractVector{<:Real}=Float64[],
                        ub::AbstractVector{<:Real}=Float64[];
                        use_threads::Bool=true,
                        arguments_checked::Bool=false)

    num_dims = length(θindices); num_dims > 0 || throw(ArgumentError("θindices must not be empty"))
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
    end
    newLb, newUb, initGuess, λindexes = init_dimensional_parameters(model, θindices, num_dims)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), num_dims)
    p=(θindices=θindices, newLb=newLb, newUb=newUb, initGuess=initGuess,
        λindexes=λindexes, consistent=consistent)
    
    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, θindices, lb, ub)

    grid = zeros(model.core.num_pars, num_points)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_dims) 
    @floop ex for dim in 1:num_dims
        grid[θindices[dim], :] .= rand(Uniform(lb[dim], ub[dim]), num_points)
    end

    pnts, lls = valid_points(model, p, grid, num_points, confidence_level, num_dims, use_threads)
    return SampledConfidenceStruct(pnts, lls)
end

# LatinHypercubeSampling
function LHS(model::LikelihoodModel,
            θindices::Vector{Int},
            num_points::Int,
            confidence_level::Float64,
            lb::AbstractVector{<:Real}=Float64[],
            ub::AbstractVector{<:Real}=Float64[];
            use_threads::Bool=true,
            arguments_checked::Bool=false)
    
    num_dims = length(θindices); num_dims > 0 || throw(ArgumentError("θindices must not be empty"))
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
        lb, ub = check_if_bounds_supplied(model, θindices, lb, ub)
    end

    newLb, newUb, initGuess, λindexes = init_dimensional_parameters(model, θindices, num_dims)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), num_dims)
    p=(θindices=θindices, newLb=newLb, newUb=newUb, initGuess=initGuess,
        λindexes=λindexes, consistent=consistent)

    grid = zeros(model.core.num_pars, num_points)
    
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid[θindices, :] = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    # grid = permutedims(scaleLHC(LHCoptim(num_points, num_dims, num_gens; kwargs...)[1], scale_range))
    
    pnts, lls = valid_points(model, p, grid, num_points, confidence_level, num_dims, use_threads)
    return SampledConfidenceStruct(pnts, lls)
end

function dimensional_likelihood_sample(model::LikelihoodModel,
                                    θindices::Vector{Int},
                                    num_points::Union{Int, Vector{Int}},
                                    confidence_level::Float64,
                                    sample_type::AbstractSampleType,
                                    lb::AbstractVector{<:Real},
                                    ub::AbstractVector{<:Real},
                                    use_threads::Bool)

    if sample_type isa UniformGridSamples
        sample_struct = uniform_grid(model, θindices, num_points, confidence_level, lb, ub;
                                        use_threads=use_threads, arguments_checked=true)
    elseif sample_type isa UniformRandomSamples
        sample_struct = uniform_random(model, θindices, num_points, confidence_level, lb, ub;             
                                        use_threads=use_threads, arguments_checked=true)
    elseif sample_type isa LatinHypercubeSamples
        sample_struct = LHS(model, θindices, num_points, confidence_level, lb, ub;
                            use_threads=use_threads, arguments_checked=true)
    end
    return sample_struct
end

function dimensional_likelihood_sample!(model::LikelihoodModel,
                                        θindices::Vector{Vector{Int}},
                                        num_points_to_sample::Union{Int, Vector{Int}};
                                        confidence_level::Float64=0.95,
                                        sample_type::AbstractSampleType=LatinHypercubeSamples(),
                                        lb::AbstractVector{<:Real}=Float64[],
                                        ub::AbstractVector{<:Real}=Float64[],
                                        θs_is_unique::Bool=false,
                                        use_threads::Bool=true,
                                        existing_profiles::Symbol=:overwrite,
                                        show_progress::Bool=model.show_progress)

    if num_points_to_sample isa Int
        num_points_to_sample > 0 || throw(DomainError("num_points_to_sample must be a strictly positive integer"))
    end
    existing_profiles ∈ [:ignore, :overwrite] || throw(ArgumentError("existing_profiles can only take value :ignore or :overwrite"))
    lb, ub = check_if_bounds_supplied(model, lb, ub)
    
    if !θs_is_unique 
        θindices = θindices[.!isempty.(θindices)]
        sort!.(θindices); unique!.(θindices)
        sort!(θindices); unique!(θindices)
    end
    # check if any of θindices is for the full likelihood - do this outside main for loop
    for (i, θs) in enumerate(θindices)
        if length(θs) == model.core.num_pars
            full_likelihood_sample!(model::LikelihoodModel, num_points_to_sample;
                                    confidence_level=confidence_level,
                                    sample_type=sample_type,
                                    lb=lb, ub=ub, use_threads=use_threads,
                                    existing_profiles=existing_profiles)
            θindices = θindices[setdiff(1:length(θindices), i)]
            break
        end
    end

    init_dim_samples_row_exists!(model, θindices, sample_type)

    θs_to_keep = trues(length(θindices))
    θs_to_overwrite = falses(length(θindices))
    num_to_overwrite = 0
    # check if profile has already been evaluated
    # in this case we only have :ignore and :overwrite
    for (i, θs) in enumerate(θindices)
        if model.dim_samples_row_exists[(θs, sample_type)][confidence_level] != 0
            θs_to_keep[i] = false
            θs_to_overwrite[i] = true
            num_to_overwrite += 1
        end
    end
    if existing_profiles == :ignore
        θindices = θindices[θs_to_keep]
        θs_to_overwrite = θs_to_overwrite[θs_to_keep]
        num_to_overwrite = 0
    end
    
    len_θindices = length(θindices)
    len_θindices > 0 || return nothing

    num_rows_required = ((len_θindices-num_to_overwrite) + model.num_dim_samples) - nrow(model.dim_samples_df)

    if num_rows_required > 0
        add_dim_samples_rows!(model, num_rows_required)
    end

    # p = Progress(length(θindices); desc="Computing dimensional profile samples: ",
    #             dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)
    profiles_to_add = @showprogress PROGRESS__METER__DT "Computing dimensional profile samples: " @distributed (vcat) for θs in θindices
        [(θs, dimensional_likelihood_sample(model, θs, num_points_to_sample,
                                            confidence_level, sample_type,
                                            lb[θs], ub[θs], use_threads))]
        # next!(p)
        # out
    end
    # finish!(p)

    for (i, (θs, sample_struct)) in enumerate(profiles_to_add)
        
        num_points_kept = length(sample_struct.ll)
        if num_points_kept == 0
            @warn string("no sampled points for θindices, ", θs, ", were in the confidence region of the profile likelihood within the supplied bounds: try increasing num_points_to_sample or changing the bounds")
            continue
        end

        if θs_to_overwrite[i]
            row_ind = model.dim_samples_row_exists[(θs, sample_type)][confidence_level]
        else
            model.num_dim_samples += 1
            row_ind = model.num_dim_samples * 1
            model.dim_samples_row_exists[(θs, sample_type)][confidence_level] = row_ind
        end

        model.dim_samples_dict[row_ind] = sample_struct
        
        set_dim_samples_row!(model, row_ind, θs, true, confidence_level, sample_type,
                                num_points_kept)
    end

    return nothing
end

function dimensional_likelihood_sample!(model::LikelihoodModel,
    θnames::Vector{Vector{Symbol}},
    num_points_to_sample::Union{Int, Vector{Int}};
    confidence_level::Float64=0.95,
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    lb::AbstractVector{<:Real}=Float64[],
    ub::AbstractVector{<:Real}=Float64[],
    θs_is_unique::Bool=false,
    use_threads::Bool=true,
    existing_profiles::Symbol=:overwrite,
    show_progress::Bool=model.show_progress)

    θindices = convertθnames_toindices(model, θnames)

    dimensional_likelihood_sample!(model, θindices, num_points_to_sample,
                                    confidence_level=confidence_level, sample_type=sample_type,
                                    lb=lb, ub=ub, θs_is_unique=θs_is_unique,
                                    use_threads=use_threads,
                                    existing_profiles=existing_profiles,
                                    show_progress=show_progress)
    return nothing
end

function dimensional_likelihood_sample!(model::LikelihoodModel,
    sample_dimension::Int,
    sample_m_random_combinations::Int,
    num_points_to_sample::Union{Int, Vector{Int}};
    confidence_level::Float64=0.95,
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    lb::AbstractVector{<:Real}=Float64[],
    ub::AbstractVector{<:Real}=Float64[],
    use_threads::Bool=true,
    use_distributed::Bool=false,
    existing_profiles::Symbol=:overwrite,
    show_progress::Bool=model.show_progress)

    sample_m_random_combinations = max(0, min(sample_m_random_combinations, binomial(model.core.num_pars, sample_dimension)))
    sample_m_random_combinations > 0 || throw(DomainError("sample_m_random_combinations must be a strictly positive integer"))

    θcombinations = sample(collect(combinations(1:model.core.num_pars, sample_dimension)),
                            sample_m_random_combinations, replace=false, ordered=true)

    dimensional_likelihood_sample!(model, θcombinations, num_points_to_sample,
                                    confidence_level=confidence_level, sample_type=sample_type,
                                    lb=lb, ub=ub, θs_is_unique=true,
                                    use_threads=use_threads,
                                    existing_profiles=existing_profiles,
                                    show_progress=show_progress)
    return nothing
end

function dimensional_likelihood_sample!(model::LikelihoodModel,
    sample_dimension::Int,
    num_points_to_sample::Union{Int, Vector{Int}};
    confidence_level::Float64=0.95,
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    lb::AbstractVector{<:Real}=Float64[],
    ub::AbstractVector{<:Real}=Float64[],
    use_threads::Bool=true,
    existing_profiles::Symbol=:overwrite,
    show_progress::Bool=model.show_progress)

    θcombinations = collect(combinations(1:model.core.num_pars, sample_dimension))

    dimensional_likelihood_sample!(model, θcombinations, num_points_to_sample,
                                    confidence_level=confidence_level, sample_type=sample_type,
                                    lb=lb, ub=ub, θs_is_unique=true,
                                    use_threads=use_threads,
                                    existing_profiles=existing_profiles,
                                    show_progress=show_progress)
    return nothing
end