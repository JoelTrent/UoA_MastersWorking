abstract type AbstractLikelihoodModel end
abstract type AbstractCoreLikelihoodModel end
abstract type AbstractEllipseMLEApprox end

abstract type AbstractConfidenceStruct end

abstract type AbstractProfileType end
abstract type AbstractEllipseProfileType <: AbstractProfileType end
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

struct UnivariateConfidenceStructAnalytical <: AbstractConfidenceStruct
    mle::Float64
    confidence_interval::Vector{<:Float64}
    lb::Float64
    ub::Float64
    # confidence_level::Float64
end

struct UnivariateConfidenceStruct <: AbstractConfidenceStruct
    mle::Float64
    confidence_interval::Vector{<:Float64}
    confidence_interval_all_pars::Matrix{Float64}
    lb::Float64
    ub::Float64
    # confidence_level::Float64
end

struct BivariateConfidenceStructAnalytical <: AbstractConfidenceStruct
    mle::Tuple{T,T} where T <: Float64
    # var_indexes::Vector{<:Int64}
    confidence_boundary_all_pars::Matrix{Float64}
    lb::Vector{<:Float64}
    ub::Vector{<:Float64}
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

struct LogLikelihood <: AbstractProfileType end
struct EllipseApprox <: AbstractEllipseProfileType end
struct EllipseApproxAnalytical <: AbstractEllipseProfileType end

struct BracketingMethodRadial <: AbstractBivariateMethod
    num_radial_directions::Int
    BracketingMethodRadial(x) = x < 1 ? error("num_radial_directions must be greater than zero") : new(x)
end

struct BracketingMethodSimultaneous <: AbstractBivariateMethod end
struct BracketingMethodFix1Axis <: AbstractBivariateMethod end

"""
`ellipse_confidence_level` is the confidence level at which to construct the initial ellipse.
`num_level_sets` the number of level sets used to get to the highest confidence level set specified in target_confidence_levels. `num_level_sets` ≥ length(target_confidence_levels)
"""
struct ContinuationMethod <: AbstractBivariateMethod 
    ellipse_confidence_level::Float64
    target_confidence_level::Float64
    # target_confidence_levels::Union{Float64, Vector{<:Float64}}
    num_level_sets::Int

    function ContinuationMethod(x,y,z)
        (0.0 < x && x < 1.0) || throw(DomainError("ellipse_confidence_level must be in the interval (0.0,1.0)"))

        (0.0 < y && y < 1.0) || throw(DomainError("target_confidence_level must be in the interval (0.0,1.0)"))

        # if y isa Float64

        z > 0 || throw(DomainError("num_level_sets must be greater than zero"))
        return new(x,y,z)
    end
end

struct AnalyticalEllipseMethod <: AbstractBivariateMethod end