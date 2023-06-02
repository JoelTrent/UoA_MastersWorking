function setθmagnitudes!(model::LikelihoodModel,
                            θmagnitudes::AbstractVector{<:Real})

    length(θmagnitudes) == model.core.num_pars || throw(ArgumentError(string("θmagnitudes must have the same length as the number of model parameters (", model.core.num_pars, ")")))

    model.core.θmagnitudes .= θmagnitudes
    return nothing
end

function setbounds!(model::LikelihoodModel;
                    lb::AbstractVector{<:Real}=Float64[],
                    ub::AbstractVector{<:Real}=Float64[])

    if !isempty(lb)
        length(lb) == model.core.num_pars || throw(ArgumentError(string("lb must have the same length as the number of model parameters (", model.core.num_pars, ")")))
        model.core.θlb .= lb
    end
    if !isempty(ub)
        length(ub) == model.core.num_pars || throw(ArgumentError(string("ub must have the same length as the number of model parameters (", model.core.num_pars, ")")))
        model.core.θub .= ub
    end
    return nothing
end

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

    indices = [zeros(Int, dim) for dim in length.(θnames_to_convert)]

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


"""
    get_target_loglikelihood(model::LikelihoodModel, 
                                confidence_level::Float64,
                                profile_type::AbstractProfileType,
                                df::Int)
"""
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
                            num_used_rows::Int,
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            sample_types::Vector{<:AbstractSampleType};
                            sample_dimension::Int=0,
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false,
                            include_higher_confidence_levels::Bool=false)
    
    df_sub = @view(df[1:num_used_rows, :])    
    row_subset = df_sub.num_points .> 0
    if for_prediction_generation
        row_subset .= row_subset .&& df_sub.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .&& .!(df_sub.not_evaluated_predictions)
    end
    if sample_dimension > 0
        row_subset .= row_subset .&& df_sub.dimension .== sample_dimension
    end

    if !isempty(confidence_levels)
        if include_higher_confidence_levels
            row_subset .= row_subset .&& (df_sub.conf_level .>= confidence_levels::Float64)
        else
            row_subset .= row_subset .&& (df_sub.conf_level .∈ Ref(confidence_levels))
        end
    end
    if !isempty(sample_types)
        row_subset .= row_subset .&& (df_sub.sample_type .∈ Ref(sample_types))
    end

    return @view(df_sub[row_subset, :])
end

function desired_df_subset(df::DataFrame, 
                            num_used_rows::Int,
                            θs_of_interest::Vector{<:Int},
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            profile_types::Vector{<:AbstractProfileType};
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false)

    df_sub = @view(df[1:num_used_rows, :])    
    row_subset = df_sub.num_points .> 0
    if for_prediction_generation
        row_subset .= row_subset .&& df_sub.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .&& .!(df_sub.not_evaluated_predictions)
    end

    if !isempty(θs_of_interest) 
        row_subset .= row_subset .&& (df_sub.θindex .∈ Ref(θs_of_interest))
    end
    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df_sub.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df_sub.profile_type .∈ Ref(profile_types))
    end

    return @view(df_sub[row_subset, :])
end

function desired_df_subset(df::DataFrame, 
                            num_used_rows::Int,
                            θs_of_interest::Vector{Tuple{Int,Int}},
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            profile_types::Vector{<:AbstractProfileType},
                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[];
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false,
                            include_lower_confidence_levels::Bool=false)

    df_sub = @view(df[1:num_used_rows, :])    
    row_subset = df_sub.num_points .> 0
    if for_prediction_generation
        row_subset .= row_subset .&& df_sub.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .&& .!(df_sub.not_evaluated_predictions)
    end

    if !isempty(θs_of_interest) 
        row_subset .= row_subset .&& (df_sub.θindices .∈ Ref(θs_of_interest))
    end
    if !isempty(confidence_levels)
        if include_lower_confidence_levels
            row_subset .= row_subset .&& (df_sub.conf_level .<= confidence_levels::Float64)
        else
            row_subset .= row_subset .&& (df_sub.conf_level .∈ Ref(confidence_levels))
        end
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df_sub.profile_type .∈ Ref(profile_types))
    end
    if !isempty(methods)
        row_subset .= row_subset .&& (typeof.(df_sub.method) .∈ Ref(typeof.(methods)))
    end

    return @view(df_sub[row_subset, :])
end