abstract type AbstractLikelihoodModel end
abstract type AbstractCoreLikelihoodModel end
abstract type AbstractEllipseMLEApprox end

abstract type AbstractConfidenceStruct end
abstract type AbstractUnivariateConfidenceStruct <: AbstractConfidenceStruct end
abstract type AbstractBivariateConfidenceStruct <: AbstractConfidenceStruct end
abstract type AbstractSampledConfidenceStruct <: AbstractConfidenceStruct end

abstract type AbstractPredictionStruct end

abstract type AbstractProfileType end
abstract type AbstractEllipseProfileType <: AbstractProfileType end

"""
    AbstractBivariateMethod

Supertype for bivariate boundary finding methods. These include [`BracketingMethodIterativeBoundary`](@ref), [`BracketingMethodRadialRandom`](@ref), [`BracketingMethodRadialMLE`](@ref), [`BracketingMethodSimultaneous`](@ref), [`BracketingMethodFix1Axis`](@ref), [`ContinuationMethod`](@ref) and [`AnalyticalEllipseMethod`](@ref).
"""
abstract type AbstractBivariateMethod end

"""
    AbstractBivariateVectorMethod <: AbstractBivariateMethod

Supertype for bivariate boundary finding methods that . These include [`BracketingMethodIterativeBoundary`](@ref), [`BracketingMethodRadialRandom`](@ref), [`BracketingMethodRadialMLE`](@ref) and [`BracketingMethodSimultaneous`](@ref).
"""
abstract type AbstractBivariateVectorMethod <: AbstractBivariateMethod end

abstract type AbstractSampleType end

struct UniformGridSamples <: AbstractSampleType end
struct UniformRandomSamples <: AbstractSampleType end
struct LatinHypercubeSamples <: AbstractSampleType end

struct CoreLikelihoodModel <: AbstractCoreLikelihoodModel
    loglikefunction::Function
    predictfunction::Union{Function, Missing}
    data::Union{Tuple, NamedTuple}
    θnames::Vector{<:Symbol}
    θname_to_index::Dict{Symbol, Int}
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
    num_dim_samples::Int

    uni_profiles_df::DataFrame
    biv_profiles_df::DataFrame
    dim_samples_df::DataFrame

    uni_profile_row_exists::Dict{Tuple{Int, AbstractProfileType}, DefaultDict{Float64, Int}}
    biv_profile_row_exists::Dict{Tuple{Tuple{Int, Int}, AbstractProfileType, AbstractBivariateMethod}, DefaultDict{Float64, Int}}
    dim_samples_row_exists::Dict{Union{AbstractSampleType, Tuple{Vector{Int}, AbstractSampleType}}, DefaultDict{Float64, Int}}

    uni_profiles_dict::Dict{Int, AbstractUnivariateConfidenceStruct}
    biv_profiles_dict::Dict{Int, AbstractBivariateConfidenceStruct}
    dim_samples_dict::Dict{Int, AbstractSampledConfidenceStruct}

    uni_predictions_dict::Dict{Int, AbstractPredictionStruct}
    biv_predictions_dict::Dict{Int, AbstractPredictionStruct}
    dim_predictions_dict::Dict{Int, AbstractPredictionStruct}

    # misc arguments
    show_progress::Bool
end

struct PointsAndLogLikelihood
    points::Array{Float64}
    ll::Vector{<:Float64}
    boundary_col_indices::Vector{<:Int64}

    function PointsAndLogLikelihood(x,y,z=zeros(Int,0))
        new(x,y,z)
    end
end

struct PredictionStruct <: AbstractPredictionStruct
    predictions::Array{Real}
    extrema::Array{Real}
end

struct SampledConfidenceStruct <: AbstractSampledConfidenceStruct
    points::Array{Float64}
    ll::Vector{<:Float64}
end

struct UnivariateConfidenceStruct <: AbstractUnivariateConfidenceStruct
    confidence_interval::Vector{<:Float64}
    interval_points::PointsAndLogLikelihood
end

struct BivariateConfidenceStruct <: AbstractBivariateConfidenceStruct
    confidence_boundary::Matrix{Float64}
    internal_points::PointsAndLogLikelihood

    function BivariateConfidenceStruct(x,y=PointsAndLogLikelihood(zeros(size(x,1),0), zeros(0)))
        return new(x,y)
    end
end

function Base.merge(a::BivariateConfidenceStruct, b::BivariateConfidenceStruct)
    return BivariateConfidenceStruct(hcat(a.confidence_boundary, b.confidence_boundary),
            PointsAndLogLikelihood(hcat(a.internal_points.points, b.internal_points.points), vcat(a.internal_points.ll, b.internal_points.ll)))
end

struct LogLikelihood <: AbstractProfileType end
struct EllipseApprox <: AbstractEllipseProfileType end
struct EllipseApproxAnalytical <: AbstractEllipseProfileType end

struct BracketingMethodRadialRandom <: AbstractBivariateVectorMethod
    num_radial_directions::Int
    BracketingMethodRadialRandom(x) = x < 1 ? throw(DomainError("num_radial_directions must be greater than zero")) : new(x)
end

"""
    BracketingMethodRadialMLE(ellipse_start_point_shift::Float64, ellipse_sqrt_distortion::Float64)::BracketingMethodRadialMLE

Method for finding the bivariate boundary of a confidence profile by bracketing between the MLE point and points on the provided bounds in directions given by points found on the boundary of a ellipse approximation of the loglikelihood function around the MLE, `e`, using [EllipseSampling.jl](https://github.com/JoelTrent/EllipseSampling.jl).

# Arguments
- `ellipse_start_point_shift`: a number ∈ [0.0,1.0]. Default is `rand()` (defined on [0.0,1.0]), meaning that, by default, every time this function is called a different set of points will be generated.
- `ellipse_sqrt_distortion`: a number ∈ [0.0,1.0]. Default is `0.01`. 

# Details

For additional information on arguments see the keyword arguments for `generate_N_clustered_points` in [EllipseSampling.jl](https://github.com/JoelTrent/EllipseSampling.jl).

Recommended for use with the [`EllipseApprox`](@ref) profile type. Will produce reasonable results for the [`LogLikelihood`](@ref) profile type when bivariate profile boundaries are convex. Otherwise, [`BracketingMethodIterativeBoundary`](@ref), which can use a starting solution from [`BracketingMethodRadialMLE`](@ref), is preferred as it iteratively improves the quality of the boundary and can discover regions not explored by this method.

# Supertype Hiearachy

BracketingMethodRadialMLE <: AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any
"""
struct BracketingMethodRadialMLE <: AbstractBivariateVectorMethod
    ellipse_start_point_shift::Float64
    ellipse_sqrt_distortion::Float64

    function BracketingMethodRadialMLE(ellipse_start_point_shift=rand(), ellipse_sqrt_distortion=0.01) 
        (0.0 <= ellipse_start_point_shift && ellipse_start_point_shift <= 1.0) || throw(DomainError("ellipse_start_point_shift must be in the closed interval [0.0,1.0]"))
        (0.0 <= ellipse_sqrt_distortion && ellipse_sqrt_distortion <= 1.0) || throw(DomainError("ellipse_sqrt_distortion must be in the closed interval [0.0,1.0]"))
        return new(ellipse_start_point_shift, ellipse_sqrt_distortion)
    end
end

struct BracketingMethodIterativeBoundary <: AbstractBivariateVectorMethod 
    initial_num_points::Int
    angle_points_per_iter::Int
    edge_points_per_iter::Int
    radial_start_point_shift::Float64
    ellipse_sqrt_distortion::Float64
    use_ellipse::Bool
    function BracketingMethodIterativeBoundary(w,x,y,z=rand(),a=1.0; use_ellipse::Bool=false)
        w > 0 || throw(DomainError("initial_num_points must be greater than zero"))
        x ≥ 0 || throw(DomainError("angle_points_per_iter must be greater than or equal to zero"))
        y ≥ 0 || throw(DomainError("edge_points_per_iter must be greater than or equal zero"))
        x > 0 || y > 0 || throw(DomainError("at least one of angle_points_per_iter and edge_points_per_iter must be greater than zero"))
        (0.0 <= z && z <= 1.0) || throw(DomainError("radial_start_point_shift must be in the closed interval [0.0,1.0]"))
        (0.0 <= a && a <= 1.0) || throw(DomainError("ellipse_sqrt_distortion must be in the closed interval [0.0,1.0]"))
        return new(w,x,y,z, a, use_ellipse)
    end
end


struct BracketingMethodSimultaneous <: AbstractBivariateVectorMethod end
struct BracketingMethodFix1Axis <: AbstractBivariateMethod end

"""
`ellipse_confidence_level` is the confidence level at which to construct the initial ellipse.
`num_level_sets` the number of level sets used to get to the highest confidence level set specified in target_confidence_levels. `num_level_sets` ≥ length(target_confidence_levels)
"""
struct ContinuationMethod <: AbstractBivariateMethod 
    num_level_sets::Int
    ellipse_confidence_level::Float64
    # target_confidence_level::Float64
    # target_confidence_levels::Union{Float64, Vector{<:Float64}}
    ellipse_start_point_shift::Float64
    level_set_spacing::Symbol

    function ContinuationMethod(x,y,z=rand(),spacing=:loglikelihood)
        x > 0 || throw(DomainError("num_level_sets must be greater than zero"))

        (0.0 < y && y < 1.0) || throw(DomainError("ellipse_confidence_level must be in the open interval (0.0,1.0)"))

        # (0.0 < y && y < 1.0) || throw(DomainError("target_confidence_level must be in the interval (0.0,1.0)"))
        # if y isa Float64

        (0.0 <= z && z <= 1.0) || throw(DomainError("ellipse_start_point_shift must be in the closed interval [0.0,1.0]"))
        spacing ∈ [:confidence, :loglikelihood] || throw(ArgumentError("level_set_spacing must be either :confidence or :loglikelihood"))

        return new(x,y,z,spacing)
    end
end

struct AnalyticalEllipseMethod <: AbstractBivariateMethod end
