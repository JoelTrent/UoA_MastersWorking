"""
    add_biv_profiles_rows!(model::LikelihoodModel, num_rows_to_add::Int)

Adds `num_rows_to_add` rows to `model.biv_profiles_df`. 
"""
function add_biv_profiles_rows!(model::LikelihoodModel, num_rows_to_add::Int)
    new_rows = init_biv_profiles_df(num_rows_to_add, 
                                    existing_largest_row=nrow(model.biv_profiles_df))

    model.biv_profiles_df = vcat(model.biv_profiles_df, new_rows)
    return nothing
end

"""
    set_biv_profiles_row!(model::LikelihoodModel, row_ind::Int, θcombination::Tuple{Int, Int},
                            not_evaluated_internal_points::Bool, not_evaluated_predictions::Bool,
                            confidence_level::Float64, profile_type::AbstractProfileType,
                            method::AbstractBivariateMethod, num_points::Int)

Sets the columns of row `row_ind` of `model.biv_profiles_df` to contain the relevant info about a just conducted profile. `model.biv_profiles_dict` contains the profile for row `row_ind` at key `row_ind`.  
"""
function set_biv_profiles_row!(model::LikelihoodModel,
                                    row_ind::Int,
                                    θcombination::Tuple{Int, Int},
                                    not_evaluated_internal_points::Bool,
                                    not_evaluated_predictions::Bool,
                                    confidence_level::Float64,
                                    profile_type::AbstractProfileType,
                                    method::AbstractBivariateMethod,
                                    num_points::Int)
    model.biv_profiles_df[row_ind, 2:end] .= θcombination, 
                                                not_evaluated_internal_points,
                                                not_evaluated_predictions,
                                                confidence_level,
                                                profile_type,
                                                method,
                                                num_points
    return nothing
end

"""
    get_bivariate_opt_func(profile_type::AbstractProfileType, method::AbstractBivariateMethod)

Returns the correct bivariate optimisation function for finding the optimal nuisance parameters at a set of interest parameters for the `profile_type` loglikelihood function. The optimisation function returns the value of the `profile_type` loglikelihood function as well as finding the optimal nuisance parameters and saving these in one of it's inputs.
"""
function get_bivariate_opt_func(profile_type::AbstractProfileType, method::AbstractBivariateMethod)
    if method isa AnalyticalEllipseMethod
        return bivariateΨ_ellipse_analytical
    elseif method isa Fix1AxisMethod
        if profile_type isa EllipseApproxAnalytical
            return bivariateΨ_ellipse_analytical
        elseif profile_type isa LogLikelihood || profile_type isa EllipseApprox
            return bivariateΨ!
        end

    elseif method isa AbstractBivariateVectorMethod
        if profile_type isa EllipseApproxAnalytical
            return bivariateΨ_ellipse_analytical_vectorsearch
        elseif profile_type isa LogLikelihood || profile_type isa EllipseApprox
            return bivariateΨ_vectorsearch!
        end
    elseif method isa ContinuationMethod
        if profile_type isa EllipseApproxAnalytical
            return bivariateΨ_ellipse_analytical_continuation
        elseif profile_type isa LogLikelihood || profile_type isa EllipseApprox
            return bivariateΨ_continuation!
        end
    end

    return missing
end

"""
    get_λs_bivariate_ellipse_analytical!(boundary,
                                            num_points::Int,
                                            consistent::NamedTuple, 
                                            ind1::Int, 
                                            ind2::Int, 
                                            num_pars::Int,
                                            initGuess::Vector{<:Float64}, 
                                            θranges::Tuple{T, T, T}, 
                                            λranges::Tuple{T, T, T},
                                            samples_all_pars::Union{Missing, Matrix{Float64}}=missing) where T<:UnitRange

Determines the nuisance parameters for a [`EllipseApproxAnalytical`](@ref) boundary profile by optimising over the unbounded ellipse approximation of the loglikelihood centred at the MLE. At higher confidence levels, where the ellipse approximation is less accurate, it is likely that predictions produced by running the model with these optimised nuisance parameters will be unrealistic and/or the parameters themselves may be infeasible for the model. 
"""
function get_λs_bivariate_ellipse_analytical!(boundary,
                                                num_points::Int,
                                                consistent::NamedTuple, 
                                                ind1::Int, 
                                                ind2::Int, 
                                                num_pars::Int,
                                                initGuess::Vector{<:Float64}, 
                                                θranges::Tuple{T, T, T}, 
                                                λranges::Tuple{T, T, T},
                                                samples_all_pars::Union{Missing, Matrix{Float64}}=missing) where T<:UnitRange

    p=(ind1=ind1, ind2=ind2, initGuess=initGuess,
                θranges=θranges, λranges=λranges, consistent=consistent)
    
    if ismissing(samples_all_pars)
        samples_all_pars = zeros(num_pars, num_points)
        samples_all_pars[[ind1, ind2], :] .= boundary
    end

    for i in 1:num_points
        variablemapping2d!(@view(samples_all_pars[:, i]), bivariateΨ_ellipse_unbounded(boundary[:,i], p), θranges, λranges)
    end

    return samples_all_pars
end

"""
    bivariate_confidenceprofile(bivariate_optimiser::Function,
                                        model::LikelihoodModel, 
                                        num_points::Int,
                                        confidence_level::Float64,
                                        consistent::NamedTuple,
                                        ind1::Int,
                                        ind2::Int,
                                        profile_type::AbstractProfileType,
                                        method::AbstractBivariateMethod,
                                        mle_targetll::Float64,
                                        save_internal_points::Bool)

Returns a [`BivariateConfidenceStruct`] containing the `num_points` boundary points and internal points (if `save_internal_points=true`) for the specified combination of parameters `ind1` and `ind2`, and `profile_type` at `confidence_level` using `method`. Calls the desired method. Called by [`bivariate_confidenceprofiles!`](@ref).
"""
function bivariate_confidenceprofile(bivariate_optimiser::Function,
                                        model::LikelihoodModel, 
                                        num_points::Int,
                                        confidence_level::Float64,
                                        consistent::NamedTuple,
                                        ind1::Int,
                                        ind2::Int,
                                        profile_type::AbstractProfileType,
                                        method::AbstractBivariateMethod,
                                        mle_targetll::Float64,
                                        save_internal_points::Bool)
    internal=PointsAndLogLikelihood(zeros(model.core.num_pars,0), zeros(0))
    if method isa AnalyticalEllipseMethod
        boundary_ellipse = generate_N_clustered_points(
                                    num_points, consistent.data_analytic.Γmle, 
                                    consistent.data_analytic.θmle, ind1, ind2,
                                    confidence_level=confidence_level,
                                    start_point_shift=method.ellipse_start_point_shift,
                                    sqrt_distortion=method.ellipse_sqrt_distortion)

        _, _, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

        boundary = get_λs_bivariate_ellipse_analytical!(
                            boundary_ellipse, 
                            num_points,
                            consistent, ind1, ind2, 
                            model.core.num_pars, initGuess,
                            θranges, λranges)
           
    elseif method isa Fix1AxisMethod
        boundary, internal = bivariate_confidenceprofile_fix1axis(
                                bivariate_optimiser, model, 
                                num_points, consistent, ind1, ind2,
                                mle_targetll, save_internal_points)
        
    elseif method isa SimultaneousMethod
        boundary, internal = bivariate_confidenceprofile_vectorsearch(
                                bivariate_optimiser, model, 
                                num_points, consistent, ind1, ind2,
                                mle_targetll, save_internal_points)

    elseif method isa RadialRandomMethod
        boundary, internal = bivariate_confidenceprofile_vectorsearch(
                                bivariate_optimiser, model, 
                                num_points, consistent, ind1, ind2,
                                mle_targetll, save_internal_points,
                                num_radial_directions=method.num_radial_directions)

    elseif method isa RadialMLEMethod
        boundary, internal = bivariate_confidenceprofile_vectorsearch(
                                bivariate_optimiser, model, 
                                num_points, consistent, ind1, ind2,
                                mle_targetll, save_internal_points,
                                ellipse_confidence_level=0.1,
                                ellipse_start_point_shift=method.ellipse_start_point_shift,
                                ellipse_sqrt_distortion=method.ellipse_sqrt_distortion)

    elseif method isa ContinuationMethod
        if profile_type isa EllipseApproxAnalytical
            bivariate_optimiser_gradient = bivariateΨ_ellipse_analytical_gradient
        else
            bivariate_optimiser_gradient = bivariateΨ_gradient!
        end

        boundary, internal = bivariate_confidenceprofile_continuation(
                                bivariate_optimiser, bivariate_optimiser_gradient,
                                model, num_points, consistent, ind1, ind2, profile_type,
                                method.ellipse_confidence_level,
                                confidence_level, 
                                method.ellipse_start_point_shift,
                                method.num_level_sets,
                                method.level_set_spacing,
                                mle_targetll, save_internal_points)
    
    elseif method isa IterativeBoundaryMethod
        boundary, internal = bivariate_confidenceprofile_iterativeboundary(
                                bivariate_optimiser, model,
                                num_points, consistent, ind1, ind2,
                                method.initial_num_points, method.angle_points_per_iter,
                                method.edge_points_per_iter, method.radial_start_point_shift,
                                method.ellipse_sqrt_distortion, method.use_ellipse,
                                mle_targetll, save_internal_points)
    end
    
    return BivariateConfidenceStruct(boundary, internal)
end

"""
    bivariate_confidenceprofiles!(model::LikelihoodModel, θcombinations::Vector{Vector{Int}}, num_points::Int; <keyword arguments>)

    Finds `num_points` `profile_type` boundary points at a specified `confidence_level` for each combination of interest parameters using a specified `method`, optionally saving any found internal points. 
    
# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `θcombinations`: vector of pairs of parameters to profile, as a vector of vectors of model parameter indexes.
- `num_points`: number of points to find on the boundary at the specified confidence level. Depending on the method, if a region of the user-provided bounds is inside the boundary some of these points will be on the bounds and inside the boundary. 

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to find the `profile_type` boundary at. Default is 0.95 (95%).
- `profile_type`: whether to use the true loglikelihood function or an ellipse approximation of the loglikelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `method`: a method of type [`AbstractBivariateMethod`](@ref). For a list of available methods use `bivariate_methods()` ([`bivariate_methods`](@ref)). Default is `RadialRandomMethod(3)` ([`RadialRandomMethod`](@ref)).
- `θcombinations_is_unique`: boolean variable specifying whether all parameter combinations in `θcombinations` are ordered by parameter index and are unique. Default is false.
- `save_internal_points`: boolean variable specifying whether to save points found inside the boundary during boundary computation. Internal points can be plotted in bivariate profile plots and will be used to generate predictions from a given bivariate profile. Default is true.
- `existing_profiles`: Symbol ∈ [:ignore, :merge, :overwrite] specifying what to do if profiles already exist for a given `θcombination`, `confidence_level`, `profile_type` and `method`. See below for each symbol's meanings. Default is :merge.
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of `θcombinations` completed and estimated time of completion. Default is `model.show_progress`.

!!! note existing_profiles meanings
    * :ignore means profiles that already exist will not be recomputed even if they contain fewer `num_points` boundary points. 
    * :merge means profiles that already exist will be merged with profiles from the current algorithm run to reach `num_points`. If the existing profile already has at least `num_points` boundary points then that profile will not be recomputed. Otherwise, the specified method will be run starting from the difference between `num_points` and the number of points in the existing profile. The result of that method run will be merged with the existing profile.  
    * :overwrite means profiles that already exist will be overwritten, regardless of how many points they contain.

# Details

Modifies model in place. Using [`bivariate_confidenceprofile`](@ref) this function calls the algorithm/method specified by `method` for each parameter combination in `θcombinations` (depending on the setting for `existing_profiles` and `num_points` if these profiles already exist). Updates `model.biv_profiles_df` for each successful profile and saves their results as a [`BivariateConfidenceStruct`](@ref) in `model.biv_profiles_dict`, where the keys for the dictionary is the row number in `model.biv_profiles_df` of the corresponding profile.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        θcombinations::Vector{Vector{Int}}, 
                                        num_points::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=RadialRandomMethod(3),
                                        θcombinations_is_unique::Bool=false,
                                        save_internal_points::Bool=true,
                                        existing_profiles::Symbol=:merge,
                                        show_progress::Bool=model.show_progress)
                                    
    existing_profiles ∈ [:ignore, :merge, :overwrite] || throw(ArgumentError("existing_profiles can only take value :ignore, :merge or :overwrite"))

    # need at least 3 boundary points some algorithms to work
    num_points = max(3, num_points)

    if profile_type isa AbstractEllipseProfileType
        check_ellipse_approx_exists!(model)
    end

    if method isa AnalyticalEllipseMethod && !(profile_type isa EllipseApproxAnalytical)
        check_ellipse_approx_exists!(model)
        profile_type = EllipseApproxAnalytical()
    end

    bivariate_optimiser = get_bivariate_opt_func(profile_type, method)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)
    mle_targetll = get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), 2)

    # for each combination, enforce ind1 < ind2 and make sure only unique combinations are run
    if !θcombinations_is_unique 
        sort!.(θcombinations); unique!.(θcombinations)
        sort!(θcombinations); unique!(θcombinations)
    end
    extrema(length.(θcombinations)) == (2,2) || throw(ArgumentError("θcombinations must only contain vectors of length 2"))

    init_biv_profile_row_exists!(model, θcombinations, profile_type, method)

    θcombinations_to_keep = trues(length(θcombinations))
    θcombinations_to_reuse = falses(length(θcombinations))
    num_to_reuse = 0
    # check if profile has already been evaluated
    for (i, (ind1, ind2)) in enumerate(θcombinations)
        if model.biv_profile_row_exists[((ind1, ind2), profile_type, method)][confidence_level] != 0
            θcombinations_to_keep[i] = false
            θcombinations_to_reuse[i] = true
            num_to_reuse += 1
        end
    end
    if existing_profiles == :ignore
        θcombinations = θcombinations[θcombinations_to_keep]
        θcombinations_to_merge = θcombinations_to_reuse[θcombinations_to_keep]
        num_to_reuse = 0
    elseif existing_profiles == :merge
        θcombinations_to_merge = θcombinations_to_reuse
    elseif existing_profiles == :overwrite
        θcombinations_to_merge = falses(length(θcombinations))
    end

    len_θcombinations = length(θcombinations)
    len_θcombinations > 0 || return nothing

    num_rows_required = ((len_θcombinations-num_to_reuse) + model.num_biv_profiles) - nrow(model.biv_profiles_df)

    if num_rows_required > 0
        add_biv_profiles_rows!(model, num_rows_required)
    end

    num_new_points = zeros(Int, len_θcombinations) .+ num_points
    if existing_profiles == :merge
        pos_new_points = trues(len_θcombinations)
        for (i, (ind1, ind2)) in enumerate(θcombinations)
            if θcombinations_to_merge[i]
                local row_ind = model.biv_profile_row_exists[((ind1, ind2), profile_type, method)][confidence_level]
                num_new_points[i] = num_new_points[i] - model.biv_profiles_df[row_ind, :num_points]
                pos_new_points[i] = num_new_points[i] > 0
            end
        end
        θcombinations = θcombinations[pos_new_points]
        num_new_points = num_new_points[pos_new_points]
        θcombinations_to_reuse = θcombinations_to_reuse[pos_new_points]
        θcombinations_to_merge = θcombinations_to_merge[pos_new_points]
    end

    len_θcombinations = length(θcombinations)
    len_θcombinations > 0 || return nothing

    # p = Progress(length(θcombinations); desc="Computing bivariate profiles: ",
    #             dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)
    profiles_to_add = @showprogress PROGRESS__METER__DT "Computing bivariate profiles: " @distributed (vcat) for (ind1, ind2) in θcombinations
        [((ind1, ind2), bivariate_confidenceprofile(bivariate_optimiser, model, num_points, 
                                                        confidence_level, consistent, 
                                                        ind1, ind2, profile_type,
                                                        method, mle_targetll,
                                                        save_internal_points))]
        # next!(p)
        # out
    end
    # finish!(p)

    for (i, (inds, boundary_struct)) in enumerate(profiles_to_add)
        if θcombinations_to_reuse[i]
            row_ind = model.biv_profile_row_exists[(inds, profile_type, method)][confidence_level]
        else
            model.num_biv_profiles += 1
            row_ind = model.num_biv_profiles * 1
            model.biv_profile_row_exists[(inds, profile_type, method)][confidence_level] = row_ind
        end

        if θcombinations_to_merge[i]
            model.biv_profiles_dict[row_ind] = merge(model.biv_profiles_dict[row_ind], boundary_struct)
        else
            model.biv_profiles_dict[row_ind] = boundary_struct
        end

        set_biv_profiles_row!(model, row_ind, inds, !save_internal_points, true, confidence_level, profile_type, method, num_points)        
    end

    return nothing
end

"""
    bivariate_confidenceprofiles!(model::LikelihoodModel, θcombinations_symbols::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}}, num_points::Int; <keyword arguments>)

Profiles just the provided `θcombinations_symbols` parameter pairs, provided as either a vector of vectors or a vector of tuples.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        θcombinations_symbols::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}}, 
                                        num_points::Int;
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=RadialRandomMethod(3),
                                        θcombinations_is_unique::Bool=false,
                                        save_internal_points::Bool=true,
                                        existing_profiles::Symbol=:merge,
                                        show_progress::Bool=model.show_progress)

    θcombinations = convertθnames_toindices(model, θcombinations_symbols)

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method, θcombinations_is_unique=θcombinations_is_unique,
            save_internal_points=save_internal_points,
            existing_profiles=existing_profiles,
            show_progress=show_progress)
    return nothing
end

"""
    bivariate_confidenceprofiles!(model::LikelihoodModel, profile_m_random_combinations::Int, num_points::Int; <keyword arguments>)

Profiles m random two-way combinations of model parameters (sampling without replacement), where 0 < m ≤ binomial(model.core.num_pars,2).

`θcombinations_is_unique` is not a valid keyword argument for this function method as it internally produces the parameter combinations which are guaranteed to be unique.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        profile_m_random_combinations::Int, 
                                        num_points::Int;
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=RadialRandomMethod(3),
                                        save_internal_points::Bool=true,
                                        existing_profiles::Symbol=:merge,
                                        show_progress::Bool=model.show_progress)

    profile_m_random_combinations = max(0, min(profile_m_random_combinations, binomial(model.core.num_pars, 2)))
    profile_m_random_combinations > 0 || throw(DomainError("profile_m_random_combinations must be a strictly positive integer"))

    θcombinations = sample(collect(combinations(1:model.core.num_pars, 2)),
                            profile_m_random_combinations, replace=false, ordered=true)

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method, θcombinations_is_unique=true, 
            save_internal_points=save_internal_points,
            existing_profiles=existing_profiles,
            show_progress=show_progress)
    return nothing
end

"""
    bivariate_confidenceprofiles!(model::LikelihoodModel, num_points::Int; <keyword arguments>)

Profiles all two-way combinations of model parameters.

`θcombinations_is_unique` is not a valid keyword argument for this function method as it internally produces the parameter combinations which are guaranteed to be unique.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        num_points::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=RadialRandomMethod(3),
                                        save_internal_points::Bool=true,
                                        existing_profiles::Symbol=:merge,
                                        show_progress::Bool=model.show_progress)

    θcombinations = collect(combinations(1:model.core.num_pars, 2))

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method, θcombinations_is_unique=true,
            save_internal_points=save_internal_points,
            existing_profiles=existing_profiles,
            show_progress=show_progress)
    return nothing
end

