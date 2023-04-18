function add_prediction_function!(model::LikelihoodModel,
                                    predictfunction::Function)

    corelikelihoodmodel = model.core
    model.core = @set corelikelihoodmodel.predictfunction = predictfunction

    return nothing
end

"""
If a model has multiple predictive variables, it assumes that `model.predictfunction` stores the prediction for each variable in its columns.
We are going to store values for each variable in the 3rd dimension (row=dim1, col=dim2, )
"""
function generate_prediction(predictfunction::Function,
                                parameter_points::Matrix{Float64},
                                proportion_to_keep::Float64)

    num_points = size(parameter_points, 2)
    prediction_one = predictfunction(parameter_points[:,1], model.core.data)
    
    if ndims(prediction_one) > 2
        error("this function has not been written to handle predictions that are stored in higher than 2 dimensions")
    end

    if ndims(prediction_one) == 2
        predictions = zeros(size(prediction_one,1), num_points, size(prediction_one,2))

        predictions[:,1,:] .= prediction_one

        for i in 2:num_points
            predictions[:,i,:] .= predictfunction(parameter_points[:,i], model.core.data)
        end

    else
        predictions = zeros(length(prediction_one), num_points)
        predictions[:,1] .= prediction_one

        for i in 2:num_points
            predictions[:,i] .= predictfunction(parameter_points[:,i], model.core.data)
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
CURRENTLY BREAKS FOR ANALYTICAL INTERVALS WHICH PRESENTLY DON'T EVALUATE THE VALUES OF NUISANCE PARAMETERS
"""
function generate_predictions_univariate!(model::LikelihoodModel,
                                proportion_to_keep::Float64;
                                confidence_levels::Vector{<:Float64}=Float64[],
                                profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                use_distributed::Bool=false)

    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the interval (0.0,1.0)"))

    df = model.uni_profiles_df
    row_subset = trues(nrow(df))
    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end

    sub_df = @view(df[row_subset, :])

    println(sub_df)

    if !use_distributed
        for i in 1:nrow(sub_df)
            predict_struct = generate_prediction(model.core.predictfunction, 
                model.uni_profiles_dict[sub_df[i, :row_ind]].interval_points.points, 
                                            proportion_to_keep)

            model.uni_predictions_dict[sub_df[i, :row_ind]] = predict_struct
        end

    else
        predictions = @distributed (vcat) for i in 1:nrow(sub_df)
            generate_prediction(model.core.predictfunction, 
                model.uni_profiles_dict[sub_df[i, :row_ind]].interval_points.points, 
                                            proportion_to_keep)
        end
        
        for (i, predict_struct) in enumerate(predictions)
            model.uni_predictions_dict[sub_df[i, :row_ind]] = predict_struct
        end
    end

    return nothing
end

function generate_predictions_bivariate!(model::LikelihoodModel,
    proportion_to_keep::Float64;
    confidence_levels::Vector{<:Float64}=Float64[],
    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
    methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[])

    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the interval (0.0,1.0)"))






end