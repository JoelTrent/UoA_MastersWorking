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

# num_points is the number of points to compute for a given method, that are on the boundary and/or inside the boundary.
function bivariate_confidenceprofiles(model::LikelihoodModel, 
                                        θcombinations::Vector{Vector{Int64}}, 
                                        num_points::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=BracketingMethodFix1Axis())

    if profile_type isa AbstractEllipseProfileType
        check_ellipse_approx_exists!(model)
    end

    if method isa AnalyticalEllipseMethod && !(profile_type isa EllipseApproxAnalytical)
        check_ellipse_approx_exists!(model)
        profile_type = EllipseApproxAnalytical()
    end

    bivariate_optimiser = get_bivariate_opt_func(profile_type, method)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Tuple{Symbol,Symbol}, BivariateConfidenceStruct}()

    for (ind1, ind2) in θcombinations

        if ind2 < ind1
            ind1, ind2 = ind2, ind1
        end

        if method isa AnalyticalEllipseMethod
            boundarySamples = generate_N_equally_spaced_points(
                                    num_points, consistent.data.Γmle, 
                                    consistent.data.θmle, ind1, ind2)
            
        elseif method isa BracketingMethodFix1Axis
            boundarySamples =  bivariate_confidenceprofile_fix1axis(
                        bivariate_optimiser, model, 
                        num_points, consistent, ind1, ind2)
            
        elseif method isa BracketingMethodSimultaneous
            boundarySamples = bivariate_confidenceprofile_vectorsearch(
                        bivariate_optimiser, model, 
                        num_points, consistent, ind1, ind2)
        elseif method isa BracketingMethodRadial
            boundarySamples = bivariate_confidenceprofile_vectorsearch(
                        bivariate_optimiser, model, 
                        num_points, consistent, ind1, ind2, 
                        num_radial_directions=method.num_radial_directions)
        elseif method isa ContinuationMethod
            if profile_type isa EllipseApproxAnalytical
                bivariate_optimiser_gradient = bivariateΨ_ellipse_analytical_gradient
            else
                bivariate_optimiser_gradient = bivariateΨ_gradient!
            end

            # consistent = merge(consistent, (targetll=get_target_loglikelihood(model, 0.0, profile_type, 2), ))

            boundarySamples = bivariate_confidenceprofile_continuation(
                        bivariate_optimiser, bivariate_optimiser_gradient,
                        model, num_points, consistent, ind1, ind2, profile_type,
                        method.ellipse_confidence_level,
                        method.target_confidence_level, method.num_level_sets)
        end
        
        confidenceDict[(model.core.θnames[ind1], model.core.θnames[ind2])] = 
                BivariateConfidenceStruct((model.core.θmle[ind1], model.core.θmle[ind2]), 
                                            boundarySamples, model.core.θlb[[ind1, ind2]], model.core.θub[[ind1, ind2]])
    end

    return confidenceDict
end

# profile just provided θcombinations_symbols
function bivariate_confidenceprofiles(model::LikelihoodModel, 
                                        θcombinations_symbols::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}}, 
                                        num_points::Int;
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=BracketingMethodFix1Axis())

    θcombinations = convertθnames_toindices(model, θcombinations_symbols)

    return bivariate_confidenceprofiles(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method)
end

# profile m random combinations of parameters (sampling without replacement), where 0 < m ≤ binomial(model.core.num_pars,2)
function bivariate_confidenceprofiles(model::LikelihoodModel, 
                                        profile_m_random_combinations::Int, 
                                        num_points::Int;
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=BracketingMethodFix1Axis())

    profile_m_random_combinations = max(0, min(profile_m_random_combinations, binomial(model.core.num_pars,2)))

    if profile_m_random_combinations == 0
        @error "`profile_m_random_combinations` must be a strictly positive integer."
        return nothing
    end

    θcombinations = sample(collect(combinations(1:model.core.num_pars, 2)), profile_m_random_combinations, replace=false)

    return bivariate_confidenceprofiles(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method)
end

# profile all combinations
function bivariate_confidenceprofiles(model::LikelihoodModel, 
                                        num_points::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=BracketingMethodFix1Axis())

    θcombinations = collect(combinations(1:model.core.num_pars, 2))

    return bivariate_confidenceprofiles(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method)
end
