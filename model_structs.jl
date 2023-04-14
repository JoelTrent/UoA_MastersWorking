abstract type AbstractLikelihoodModel end
abstract type AbstractCoreLikelihoodModel end
abstract type AbstractEllipseMLEApprox end

abstract type AbstractConfidenceStruct end
abstract type AbstractUnivariateConfidenceStruct <: AbstractConfidenceStruct end
abstract type AbstractBivariateConfidenceStruct <: AbstractConfidenceStruct end

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
    # confidence_levels_evaluated::DefaultDict{Float64, Bool}
    # confidence_intervals_evaluated::Dict{Float64, DefaultDict{Union{Int, Symbol}, Bool}}
    # univariate_intervals
    # bivariate_intervals
    num_uni_profiles::Int
    num_biv_profiles::Int

    uni_profiles_df::DataFrame
    biv_profiles_df::DataFrame

    uni_profile_row_exists::Dict{Tuple{Int, AbstractProfileType}, DefaultDict{Float64, Int}}
    biv_profile_row_exists::Dict{Tuple{Tuple{Int, Int}, AbstractProfileType, AbstractBivariateMethod}, DefaultDict{Float64, Int}}

    uni_profiles_dict::Dict{Int, AbstractUnivariateConfidenceStruct}
    biv_profiles_dict::Dict{Int, AbstractBivariateConfidenceStruct}

    # other relevant fields

end

struct UnivariateConfidenceStructAnalytical <: AbstractUnivariateConfidenceStruct
    confidence_interval::Vector{<:Float64}
    internal_points::Matrix{Float64}

    function UnivariateConfidenceStructAnalytical(x,y=Matrix{Float64}(undef,0,0))
        return new(x,y)
    end
end

struct UnivariateConfidenceStruct <: AbstractUnivariateConfidenceStruct
    confidence_interval::Vector{<:Float64}
    confidence_interval_all_pars::Matrix{Float64}
    internal_points::Matrix{Float64}

    function UnivariateConfidenceStruct(x,y,z=Matrix{Float64}(undef,0,0))
        return new(x,y,z)
    end
end

struct BivariateConfidenceStructAnalytical <: AbstractBivariateConfidenceStruct
    mle::Tuple{T,T} where T <: Float64
    confidence_boundary::Matrix{Float64}
    internal_points::Matrix{Float64}

    function BivariateConfidenceStructAnalytical(x,y,z=Matrix{Float64}(undef,0,0))
        return new(x,y,z)
    end
end

struct BivariateConfidenceStruct <: AbstractBivariateConfidenceStruct
    mle::Tuple{T,T} where T <: Float64
    confidence_boundary_all_pars::Matrix{Float64}
    internal_points::Matrix{Float64}

    function BivariateConfidenceStruct(x,y,z=Matrix{Float64}(undef,0,0))
        return new(x,y,z)
    end
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