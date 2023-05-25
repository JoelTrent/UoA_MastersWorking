function add_prediction_function!(model::LikelihoodModel,
                                    predictfunction::Function)

    corelikelihoodmodel = model.core
    model.core = @set corelikelihoodmodel.predictfunction = predictfunction
    model.core = @set corelikelihoodmodel.ymle = predictfunction(θmle, model.core.data)

    return nothing
end

function check_prediction_function_exists(model::LikelihoodModel)
    if ismissing(model.core.predictfunction) 
        @warn "LikelihoodModel does not contain a function for evaluating predictions. Please add a prediction function using add_prediction_function!"
        return false
    end

    return true
end

"""
If a model has multiple predictive variables, it assumes that `model.predictfunction` stores the prediction for each variable in its columns.
We are going to store values for each variable in the 3rd dimension (row=dim1, col=dim2, page/sheet=dim3)
"""
function generate_prediction(predictfunction::Function,
                                data,
                                t::Vector,
                                data_ymle::AbstractArray{<:Real},
                                parameter_points::Matrix{Float64},
                                proportion_to_keep::Real)

    num_points = size(parameter_points, 2)
    
    if ndims(data_ymle) > 2
        error("this function has not been written to handle predictions that are stored in higher than 2 dimensions")
    end

    if ndims(data_ymle) == 2
        predictions = zeros(length(t), num_points, size(data_ymle, 2))

        for i in 1:num_points
            predictions[:,i,:] .= predictfunction(parameter_points[:,i], data, t)
        end

    else
        predictions = zeros(length(t), num_points)

        for i in 1:num_points
            predictions[:,i] .= predictfunction(parameter_points[:,i], data, t)
        end
    end
    
    extrema = hcat(minimum(predictions, dims=2), maximum(predictions, dims=2))

    num_to_keep = convert(Int, round(num_points*proportion_to_keep, RoundUp))
    if num_points < 2
        predict_struct = PredictionStruct(predictions, extrema)
        return predict_struct
    elseif num_to_keep < 2
        num_to_keep = 2
    end

    keep_i = sample(1:num_points, num_to_keep, replace=false, ordered=true)
    if ndims(data_ymle) == 2
        predict_struct = PredictionStruct(predictions[:, keep_i,:], extrema)
    else
        predict_struct = PredictionStruct(predictions[:, keep_i], extrema)
    end
    return predict_struct
end

function generate_prediction_univariate(model::LikelihoodModel,
                                        sub_df,
                                        row_i::Int,
                                        t::Vector,
                                        proportion_to_keep::Real)

    boundary_col_indices = model.uni_profiles_dict[sub_df[row_i, :row_ind]].interval_points.boundary_col_indices
    boundary_range = boundary_col_indices[1]:boundary_col_indices[2]

    return generate_prediction(model.core.predictfunction, 
                model.core.data, t, model.core.ymle,
                model.uni_profiles_dict[sub_df[row_i, :row_ind]].interval_points.points[:, boundary_range], 
                                                                proportion_to_keep)
end

function generate_prediction_bivariate(model::LikelihoodModel,
                                        sub_df,
                                        row_i::Int,
                                        t::Vector,
                                        proportion_to_keep::Real)

    conf_struct = model.biv_profiles_dict[sub_df[row_i, :row_ind]]

    if !isempty(conf_struct.internal_points.points)
        return generate_prediction(model.core.predictfunction, 
                                    model.core.data, t, model.core.ymle,
                                    hcat(conf_struct.confidence_boundary, conf_struct.internal_points.points), 
                                    proportion_to_keep)
    end
    return generate_prediction(model.core.predictfunction, 
                                model.core.data, t, model.core.ymle,
                                conf_struct.confidence_boundary, 
                                proportion_to_keep)
end

"""
"""
function generate_predictions_univariate!(model::LikelihoodModel,
                                            t::Vector,
                                            proportion_to_keep::Real;
                                            confidence_levels::Vector{<:Float64}=Float64[],
                                            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                            show_progress=model.show_progress)

    check_prediction_function_exists(model) || return nothing

    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the interval (0.0,1.0)"))
    sub_df = desired_df_subset(model.uni_profiles_df, model.num_uni_profiles, Int[], confidence_levels, profile_types, for_prediction_generation=true)

    if nrow(sub_df) < 1
        return nothing
    end

    
    # p = Progress(nrow(sub_df); desc="Generating univariate profile predictions: ",
    #             dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)
    predictions = @showprogress PROGRESS__METER__DT "Generating univariate profile predictions: " @distributed (vcat) for i in 1:nrow(sub_df)
        [generate_prediction_univariate(model, sub_df, i, t, proportion_to_keep)]
        # next!(p)
        # out
    end
    # finish!(p)
    
    for (i, predict_struct) in enumerate(predictions)
        model.uni_predictions_dict[sub_df[i, :row_ind]] = predict_struct
    end

    sub_df[:, :not_evaluated_predictions] .= false

    return nothing
end

function generate_predictions_bivariate!(model::LikelihoodModel,
                                            t::Vector,
                                            proportion_to_keep::Real;
                                            confidence_levels::Vector{<:Float64}=Float64[],
                                            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                            show_progress=model.show_progress)

    check_prediction_function_exists(model) || return nothing
    
    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the interval (0.0,1.0)"))
    sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int, Int}[], confidence_levels, profile_types, methods, for_prediction_generation=true)

    if nrow(sub_df) < 1
        return nothing
    end

    # p = Progress(nrow(sub_df); desc="Generating bivariate profile predictions: ",
    #             dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)
    predictions = @showprogress PROGRESS__METER__DT "Generating bivariate profile predictions: " @distributed (vcat) for i in 1:nrow(sub_df)
        [generate_prediction_bivariate(model, sub_df, i,
                                        t, proportion_to_keep)]
        # next!(p)
        # out
    end
    # finish!(p)
    
    for (i, predict_struct) in enumerate(predictions)
        model.biv_predictions_dict[sub_df[i, :row_ind]] = predict_struct
    end

    sub_df[:, :not_evaluated_predictions] .= false

    return nothing
end

function generate_predictions_dim_samples!(model::LikelihoodModel,
                                            t::Vector,
                                            proportion_to_keep::Real;
                                            confidence_levels::Vector{<:Float64}=Float64[],
                                            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                            show_progress=model.show_progress)

    check_prediction_function_exists(model) || return nothing
    
    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the interval (0.0,1.0)"))
    sub_df = desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_levels, sample_types, for_prediction_generation=true)

    if nrow(sub_df) < 1
        return nothing
    end

    # p = Progress(nrow(sub_df); desc="Generating dimensional profile sample predictions: ",
    #             dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)
    predictions = @showprogress PROGRESS__METER__DT "Generating dimensional profile sample predictions: "  @distributed (vcat) for i in 1:nrow(sub_df)
        parameter_points = model.dim_samples_dict[sub_df[i, :row_ind]].points
        [generate_prediction(model.core.predictfunction, model.core.data, t, 
                                            model.core.ymle, parameter_points, proportion_to_keep)]
        # next!(p)
        # out
    end
    # finish!(p)
    
    for (i, predict_struct) in enumerate(predictions)
        model.dim_predictions_dict[sub_df[i, :row_ind]] = predict_struct
    end

    sub_df[:, :not_evaluated_predictions] .= false
    return nothing
end