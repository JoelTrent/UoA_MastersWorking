abstract type AbstractLikelihoodModel end
abstract type AbstractCoreLikelihoodModel end
abstract type AbstractEllipseMLEApprox end

struct CoreLikelihoodModel <: AbstractCoreLikelihoodModel
    loglikefunction::Function
    predictfunction::Union{Function,Missing}
    data::Union{Tuple,NamedTuple}
    θnames::Vector{<:Symbol}
    θname_to_index::Dict{Symbol,Int}
    θlb::AbstractVector{<:Real}
    θub::AbstractVector{<:Real}
    θmagnitudes::AbstractVector{<:Real}
    θmle::Vector{<:Float64}
    ymle::Array{<:Real}
    maximisedmle::Float64
    num_pars::Int
end

struct EllipseMLEApprox <: AbstractEllipseMLEApprox
    Hmle::Matrix{<:Float64}
    Γmle::Matrix{<:Float64}
end

mutable struct LikelihoodModel <: AbstractLikelihoodModel
    core::CoreLikelihoodModel
    ellipse_MLE_approx::Union{Missing,EllipseMLEApprox}

    num_uni_profiles::Int
    num_biv_profiles::Int
    num_dim_samples::Int

    uni_profiles_df::DataFrame
    biv_profiles_df::DataFrame
    dim_samples_df::DataFrame

    uni_profile_row_exists::Dict{Tuple{Int,AbstractProfileType},DefaultDict{Float64,Int}}
    biv_profile_row_exists::Dict{Tuple{Tuple{Int,Int},AbstractProfileType,AbstractBivariateMethod},DefaultDict{Float64,Int}}
    dim_samples_row_exists::Dict{Union{AbstractSampleType,Tuple{Vector{Int},AbstractSampleType}},DefaultDict{Float64,Int}}

    uni_profiles_dict::Dict{Int,AbstractUnivariateConfidenceStruct}
    biv_profiles_dict::Dict{Int,AbstractBivariateConfidenceStruct}
    dim_samples_dict::Dict{Int,AbstractSampledConfidenceStruct}

    uni_predictions_dict::Dict{Int,AbstractPredictionStruct}
    biv_predictions_dict::Dict{Int,AbstractPredictionStruct}
    dim_predictions_dict::Dict{Int,AbstractPredictionStruct}

    # misc arguments
    show_progress::Bool
end