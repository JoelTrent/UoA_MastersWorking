abstract type AbstractPredictionStruct end

struct PredictionStruct <: AbstractPredictionStruct
    predictions::Array{Real}
    extrema::Array{Real}
end