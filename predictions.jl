function add_prediction_function!(model::LikelihoodModel,
                                    predictfunction::Function)

    corelikelihoodmodel = model.core
    model.core = @set corelikelihoodmodel.predictfunction = predictfunction

    return nothing
end