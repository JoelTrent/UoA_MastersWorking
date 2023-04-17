function init_biv_profile_row_exists!(model::LikelihoodModel, 
                                        θcombinations::Vector{Vector{Int}},
                                        profile_type::AbstractProfileType,
                                        method::AbstractBivariateMethod)
    for (ind1, ind2) in θcombinations
        if !haskey(model.biv_profile_row_exists, ((ind1, ind2), profile_type, method))
            model.biv_profile_row_exists[((ind1, ind2), profile_type, method)] = DefaultDict{Float64, Int}(0)
        end
    end
    return nothing
end

function add_biv_profiles_rows!(model::LikelihoodModel)
    new_rows = init_biv_profiles_df(model.core.num_pars, 
                                    existing_largest_row=nrow(model.biv_profiles_df))

    model.biv_profiles_df = vcat(model.biv_profiles_df, new_rows)
    return nothing
end

function update_biv_profiles_row!(model::LikelihoodModel, 
                                    θcombination::Tuple{Int, Int},
                                    evaluated_internal_points::Bool,
                                    confidence_level::Float64,
                                    profile_type::AbstractProfileType,
                                    method::AbstractBivariateMethod,
                                    num_points::Int)
    model.biv_profiles_df[model.num_biv_profiles, 2:end] .= θcombination, 
                                                            evaluated_internal_points,
                                                            confidence_level,
                                                            profile_type,
                                                            method,
                                                            num_points
    return nothing
end

function get_bivariate_opt_func(profile_type::AbstractProfileType, method::AbstractBivariateMethod)
    if method isa BracketingMethodFix1Axis
        if profile_type isa EllipseApproxAnalytical
            return bivariateΨ_ellipse_analytical
        elseif profile_type isa LogLikelihood || profile_type isa EllipseApprox
            return bivariateΨ!
        end

    elseif method isa BracketingMethodRadial || method isa BracketingMethodSimultaneous
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

function bivariate_confidenceprofile(bivariate_optimiser::Function,
                                        model::LikelihoodModel, 
                                        num_points::Int,
                                        confidence_level::Float64,
                                        consistent::NamedTuple,
                                        ind1::Int,
                                        ind2::Int,
                                        profile_type::AbstractProfileType,
                                        method::AbstractBivariateMethod,
                                        atol::Real)
    if method isa AnalyticalEllipseMethod
        boundarySamples = generate_N_equally_spaced_points(
                                num_points, consistent.data.Γmle, 
                                consistent.data.θmle, ind1, ind2)
        
    elseif method isa BracketingMethodFix1Axis
        boundarySamples = bivariate_confidenceprofile_fix1axis(
                    bivariate_optimiser, model, 
                    num_points, consistent, ind1, ind2, atol)
        
    elseif method isa BracketingMethodSimultaneous
        boundarySamples = bivariate_confidenceprofile_vectorsearch(
                    bivariate_optimiser, model, 
                    num_points, consistent, ind1, ind2, atol)
    elseif method isa BracketingMethodRadial
        boundarySamples = bivariate_confidenceprofile_vectorsearch(
                    bivariate_optimiser, model, 
                    num_points, consistent, ind1, ind2, atol,
                    num_radial_directions=method.num_radial_directions)
    elseif method isa ContinuationMethod
        if profile_type isa EllipseApproxAnalytical
            bivariate_optimiser_gradient = bivariateΨ_ellipse_analytical_gradient
        else
            bivariate_optimiser_gradient = bivariateΨ_gradient!
        end

        boundarySamples = bivariate_confidenceprofile_continuation(
                    bivariate_optimiser, bivariate_optimiser_gradient,
                    model, num_points, consistent, ind1, ind2, atol, profile_type,
                    method.ellipse_confidence_level,
                    confidence_level, 
                    method.ellipse_start_point_shift,
                    method.num_level_sets)
    end
    
    return BivariateConfidenceStruct(boundarySamples)
end

"""
atol is the absolute tolerance that decides if f(x) ≈ 0.0. I.e. if the loglikelihood function is approximately at the boundary of interest.

num_points is the number of points to compute for a given method, that are on the boundary and/or inside the boundary.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        θcombinations::Vector{Vector{Int}}, 
                                        num_points::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=BracketingMethodFix1Axis(),
                                        atol::Real=1e-8, 
                                        θcombinations_is_unique::Bool=false)
                                    
    atol > 0 || throw(DomainError("atol must be a strictly positive integer"))

    if profile_type isa AbstractEllipseProfileType
        check_ellipse_approx_exists!(model)
    end

    if method isa AnalyticalEllipseMethod && !(profile_type isa EllipseApproxAnalytical)
        check_ellipse_approx_exists!(model)
        profile_type = EllipseApproxAnalytical()
    end

    bivariate_optimiser = get_bivariate_opt_func(profile_type, method)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)

    # for each combination, enforce ind1 < ind2 and make sure only unique combinations are run
    θcombinations_is_unique || (sort!.(θcombinations); sort!(θcombinations); unique!(θcombinations))

    init_biv_profile_row_exists!(model, θcombinations, profile_type, method)

    # check if profile has already been evaluated
    θcombinations_to_keep = trues(length(θcombinations))
    for (i, (ind1, ind2)) in enumerate(θcombinations)
        if model.biv_profile_row_exists[((ind1, ind2), profile_type, method)][confidence_level] != 0
            θcombinations_to_keep[i] = false
        end
    end
    θcombinations = θcombinations[θcombinations_to_keep]
    length(θcombinations) > 0 || return nothing

    for (ind1, ind2) in θcombinations
        model.num_biv_profiles += 1
        model.biv_profile_row_exists[((ind1, ind2), profile_type, method)][confidence_level] = model.num_biv_profiles * 1

        boundary_struct = bivariate_confidenceprofile(bivariate_optimiser, model, num_points, 
                                                        confidence_level, consistent, 
                                                        ind1, ind2, profile_type,
                                                        method, atol)

        model.biv_profiles_dict[model.num_biv_profiles] = boundary_struct

        if nrow(model.biv_profiles_df) < model.num_biv_profiles
            add_biv_profiles_rows!(model)
        end
        update_biv_profiles_row!(model, (ind1, ind2), false, confidence_level, profile_type, method, num_points)        
    end

    return nothing
end

# profile just provided θcombinations_symbols
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        θcombinations_symbols::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}}, 
                                        num_points::Int;
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=BracketingMethodFix1Axis(),
                                        atol::Real=1e-8,
                                        θcombinations_is_unique::Bool=false)

    θcombinations = convertθnames_toindices(model, θcombinations_symbols)

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method, atol=atol, θcombinations_is_unique=θcombinations_is_unique)
    return nothing
end

# profile m random combinations of parameters (sampling without replacement), where 0 < m ≤ binomial(model.core.num_pars,2)
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        profile_m_random_combinations::Int, 
                                        num_points::Int;
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=BracketingMethodFix1Axis(),
                                        atol::Real=1e-8)

    profile_m_random_combinations = max(0, min(profile_m_random_combinations, binomial(model.core.num_pars, 2)))
    profile_m_random_combinations > 0 || throw(DomainError("profile_m_random_combinations must be a strictly positive integer"))

    θcombinations = sample(collect(combinations(1:model.core.num_pars, 2)), profile_m_random_combinations, replace=false)

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method, atol=atol, θcombinations_is_unique=true)
    return nothing
end

# profile all combinations
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        num_points::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=BracketingMethodFix1Axis(),
                                        atol::Real=1e-8)

    θcombinations = collect(combinations(1:model.core.num_pars, 2))

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method, atol=atol, θcombinations_is_unique=true)
    return nothing
end

