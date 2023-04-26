function add_prediction_function!(model::LikelihoodModel,
                                    predictfunction::Function)

    corelikelihoodmodel = model.core
    model.core = @set corelikelihoodmodel.predictfunction = predictfunction
    model.core = @set corelikelihoodmodel.ymle = predictfunction(θmle, model.core.data)

    return nothing
end

"""
If a model has multiple predictive variables, it assumes that `model.predictfunction` stores the prediction for each variable in its columns.
We are going to store values for each variable in the 3rd dimension (row=dim1, col=dim2, page/sheet=dim3)
"""
function generate_prediction(predictfunction::Function,
                                data,
                                parameter_points::Matrix{Float64},
                                proportion_to_keep::Float64)

    num_points = size(parameter_points, 2)
    
    if ndims(model.core.ymle) > 2
        error("this function has not been written to handle predictions that are stored in higher than 2 dimensions")
    end

    if ndims(model.core.ymle) == 2
        predictions = zeros(size(model.core.ymle,1), num_points, size(model.core.ymle,2))

        for i in 1:num_points
            predictions[:,i,:] .= predictfunction(parameter_points[:,i], data)
        end

    else
        predictions = zeros(length(model.core.ymle), num_points)

        for i in 1:num_points
            predictions[:,i] .= predictfunction(parameter_points[:,i],data)
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
    predict_struct = PredictionStruct(predictions[:, keep_i], extrema)

    return predict_struct
end

"""
"""
function generate_predictions_univariate!(model::LikelihoodModel,
                                proportion_to_keep::Float64;
                                confidence_levels::Vector{<:Float64}=Float64[],
                                profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                use_distributed::Bool=false)

    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the interval (0.0,1.0)"))

    df = model.uni_profiles_df
    row_subset = df.num_points .> 0
    row_subset = df.not_evaluated_predictions .&& row_subset
    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end

    sub_df = @view(df[row_subset, :])

    if nrow(sub_df) < 1
        return nothing
    end

    if !use_distributed
        for i in 1:nrow(sub_df)
            boundary_col_indices = model.uni_profiles_dict[sub_df[i, :row_ind]].interval_points.boundary_col_indices
            boundary_range = boundary_col_indices[1]:boundary_col_indices[2]

            predict_struct = generate_prediction(model.core.predictfunction,
                model.core.data,
                model.uni_profiles_dict[sub_df[i, :row_ind]].interval_points.points[:, boundary_range], 
                                            proportion_to_keep)

            model.uni_predictions_dict[sub_df[i, :row_ind]] = predict_struct
        end

    else
        predictions = @distributed (vcat) for i in 1:nrow(sub_df)
            boundary_col_indices = model.uni_profiles_dict[sub_df[i, :row_ind]].interval_points.boundary_col_indices
            boundary_range = boundary_col_indices[1]:boundary_col_indices[2]

            generate_prediction(model.core.predictfunction, 
                model.core.data,
                model.uni_profiles_dict[sub_df[i, :row_ind]].interval_points.points[:, boundary_range], 
                                            proportion_to_keep)
        end
        
        for (i, predict_struct) in enumerate(predictions)
            model.uni_predictions_dict[sub_df[i, :row_ind]] = predict_struct
        end
    end

    sub_df[:, :not_evaluated_predictions] .= false

    return nothing
end

function generate_predictions_bivariate!(model::LikelihoodModel,
                                            proportion_to_keep::Float64;
                                            confidence_levels::Vector{<:Float64}=Float64[],
                                            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                            use_distributed::Bool=false)

    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the interval (0.0,1.0)"))

    df = model.biv_profiles_df
    row_subset = df.num_points .> 0
    row_subset = df.not_evaluated_predictions .&& row_subset
    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end
    if !isempty(methods)
        row_subset .= row_subset .&& (df.method .∈ Ref(methods))
    end

    sub_df = @view(df[row_subset, :])

    if nrow(sub_df) < 1
        return nothing
    end

    if !use_distributed
        for i in 1:nrow(sub_df)

            conf_struct = model.biv_profiles_dict[sub_df[i, :row_ind]]

            if !isempty(conf_struct.internal_points)
                predict_struct = generate_prediction(model.core.predictfunction, 
                                    model.core.data,
                                    hcat(conf_struct.confidence_boundary, conf_struct.internal_points), 
                                                proportion_to_keep)
            else
                predict_struct = generate_prediction(model.core.predictfunction, 
                                    model.core.data,
                                    conf_struct.confidence_boundary, 
                                                proportion_to_keep)
            end

            model.biv_predictions_dict[sub_df[i, :row_ind]] = predict_struct
        end

    else
        predictions = @distributed (vcat) for i in 1:nrow(sub_df)
            conf_struct = model.biv_profiles_dict[sub_df[i, :row_ind]]

            if !isempty(conf_struct.internal_points)
                generate_prediction(model.core.predictfunction, 
                                    model.core.data,
                                    hcat(conf_struct.confidence_boundary, conf_struct.internal_points), 
                                                proportion_to_keep)
            else
                generate_prediction(model.core.predictfunction, 
                                    model.core.data,
                                    conf_struct.confidence_boundary, 
                                                proportion_to_keep)
            end
        end
        
        for (i, predict_struct) in enumerate(predictions)
            model.biv_predictions_dict[sub_df[i, :row_ind]] = predict_struct
        end
    end

    sub_df[:, :not_evaluated_predictions] .= false

    return nothing
end