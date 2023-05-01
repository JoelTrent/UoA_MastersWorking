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

function desired_df_subset(df::DataFrame, 
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            sample_types::Vector{<:AbstractSampleType};
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false,
                            include_higher_confidence_levels::Bool=false)

    row_subset = df.num_points .> 0
    if for_prediction_generation
        row_subset .= row_subset .&& df.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .&& .!(df.not_evaluated_predictions)
    end

    if !isempty(confidence_levels)
        if include_higher_confidence_levels
            row_subset .= row_subset .&& (df.conf_level .>= confidence_levels::Float64)
        else
            row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
        end
    end
    if !isempty(sample_types)
        row_subset .= row_subset .&& (df.sample_type .∈ Ref(sample_types))
    end

    return @view(df[row_subset, :])
end

function desired_df_subset(df::DataFrame, 
                            θs_of_interest::Vector{<:Int},
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            profile_types::Vector{<:AbstractProfileType};
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false)

    row_subset = df.num_points .> 0
    if for_prediction_generation
        row_subset .= row_subset .&& df.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .&& .!(df.not_evaluated_predictions)
    end

    if !isempty(θs_of_interest) 
        row_subset .= row_subset .&& (df.θindex .∈ Ref(θs_of_interest))
    end
    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end

    return @view(df[row_subset, :])
end

function desired_df_subset(df::DataFrame, 
                            θs_of_interest::Vector{Tuple{Int,Int}},
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            profile_types::Vector{<:AbstractProfileType},
                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[];
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false,
                            include_lower_confidence_levels::Bool=false)

    row_subset = df.num_points .> 0
    if for_prediction_generation
        row_subset .= row_subset .&& df.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .&& .!(df.not_evaluated_predictions)
    end

    if !isempty(θs_of_interest) 
        row_subset .= row_subset .&& (df.θindices .∈ Ref(θs_of_interest))
    end
    if !isempty(confidence_levels)
        if include_lower_confidence_levels
            row_subset .= row_subset .&& (df.conf_level .<= confidence_levels::Float64)
        else
            row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
        end
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end
    if !isempty(methods)
        row_subset .= row_subset .&& (df.method .∈ Ref(methods))
    end

    return @view(df[row_subset, :])
end