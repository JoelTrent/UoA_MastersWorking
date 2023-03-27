abstract type AbstractLikelihoodModel end
abstract type AbstractCoreLikelihoodModel end
abstract type AbstractEllipseMLEApprox end

abstract type AbstractBivariateMethod end

struct CoreLikelihoodModel <: AbstractCoreLikelihoodModel
    loglikefunction::Function
    data::Union{Tuple, NamedTuple}
    θnames::Vector{<:Symbol}
    θname_to_index::Dict{Symbol, Int}
    θlb::Vector{<:Float64}
    θub::Vector{<:Float64}
    θmle::Vector{<:Float64}
    maximisedmle::Float64
    num_pars::Int
end

struct EllipseMLEApprox <: AbstractEllipseMLEApprox
    Hmle::Matrix{<:Float64}
    Γmle::Matrix{<:Float64}
end

mutable struct LikelihoodModel <: AbstractLikelihoodModel
    core::CoreLikelihoodModel

    ellipse_MLE_approx::Union{Missing, EllipseMLEApprox}

    # perhaps something more like confidence sets evaluated.
    # Want to know:
    # - what levels evaluated at
    # - which parameters have been evaluated at a given level and the dimensionality used
    # - whether an ellipse approximation (analytical or profile) was used or full likelihood was used
    # - make it easy to check if a wider or smaller confidence level has been evaluated yet (can use that knowledge to change the search bounds in 1D or perhaps provide initial points in 2D that are guaranteed to be inside/outside a confidence level boundary)
    confidence_levels_evaluated::DefaultDict{Float64, Bool}
    confidence_intervals_evaluated::Dict{Float64, DefaultDict{Union{Int, Symbol}, Bool}}
    # univariate_intervals
    # bivariate_intervals

    # other relevant fields

end

abstract type AbstractConfidenceStruct end

struct UnivariateConfidenceStruct <: AbstractConfidenceStruct
    mle::Float64
    confidence_interval::Vector{<:Float64}
    lb::Float64
    ub::Float64
    # confidence_level::Float64
end

struct BivariateConfidenceStruct <: AbstractConfidenceStruct
    mle::Tuple{T,T} where T <: Float64
    # var_indexes::Vector{<:Int64}
    confidence_boundary_all_pars::Matrix{Float64}
    lb::Vector{<:Float64}
    ub::Vector{<:Float64}
    # confidence_level::Float64
end

struct BracketingMethodRadial <: AbstractBivariateMethod
    num_radial_directions::Int
end
struct BracketingMethodSimultaneous <: AbstractBivariateMethod end

struct BracketingMethodFix1Axis <: AbstractBivariateMethod end