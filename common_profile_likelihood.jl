function convertθnames_toindices(model::LikelihoodModel, 
                                    θnames_to_convert::Vector{<:Symbol})

    indices = zeros(Int, length(θnames_to_convert))

    for (i, name) in enumerate(θnames_to_convert)
        indices[i] = model.core.θname_to_index[name]
    end

    return indices
end

function convertθnames_toindices(model::LikelihoodModel, 
                                    θnames_to_convert::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}})

    indices = [zeros(Int, 2) for _ in 1:length(θnames_to_convert)]

    for (i, names) in enumerate(θnames_to_convert)
        indices[i] .= getindex.(Ref(model.core.θname_to_index), names)
    end

    return indices
end

"""
If the `profile_type` is LogLikelihood, then corrects a log-likelihood such that an input log-likelihood (which has value of zero at the mle) will now have a value of model.core.maximisedmle at the mle. Otherwise, a copy of ll is returned.
"""
function ll_correction(model::LikelihoodModel, profile_type::AbstractProfileType, ll::Float64)
    if profile_type isa LogLikelihood
        return ll + model.core.maximisedmle
    end
    return ll * 1.0
end

function get_target_loglikelihood(model::LikelihoodModel, 
                                    confidence_level::Float64,
                                    profile_type::AbstractProfileType,
                                    df::Int)

    (0.0 ≤ confidence_level && confidence_level ≤ 1.0) || throw(DomainError("confidence_level must be in the interval [0,1]"))

    llstar = -quantile(Chisq(df), confidence_level)/2.0

    return ll_correction(model, profile_type, llstar)
end

function get_consistent_tuple(model::LikelihoodModel, 
                                confidence_level::Float64, 
                                profile_type::AbstractProfileType, 
                                df::Int)

    targetll = get_target_loglikelihood(model, confidence_level, profile_type, df)

    if profile_type isa LogLikelihood 
        return (targetll=targetll, num_pars=model.core.num_pars,
                 loglikefunction=model.core.loglikefunction, data=model.core.data)
    elseif profile_type isa AbstractEllipseProfileType
        return (targetll=targetll, num_pars=model.core.num_pars, 
                loglikefunction=ellipse_loglike, 
                data=(θmle=model.core.θmle, Hmle=model.ellipse_MLE_approx.Hmle),
                data_analytic=(θmle=model.core.θmle, Γmle=model.ellipse_MLE_approx.Γmle))
    end

    return (missing)
end
