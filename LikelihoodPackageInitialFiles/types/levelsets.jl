"""
    PointsAndLogLikelihood(points::Array{Float64}, ll::Vector{<:Float64}, boundary_col_indices::Vector{<:Int64}=zeros(Int, 0)))

Struct that stores an array of parameter points, their corresponding loglikelihood value and, in the case of univariate profiles, the column indices in `points` of the confidence interval parameters.

# Fields
- `points`: an array of points stored in columns, with each row corresponding to the respective index of each model parameter. For the [`UnivariateConfidenceStruct`](@ref) type, these points are stored in column-wise order of increasing interest parameter magnitude. For the [`BivariateConfidenceStruct`](@ref) type these points are stored in the order they are found. 
- `ll`: a vector of loglikelihood function values corresponding to the point in each column of `points`. This number is standardised so that regardless of whether the true loglikelihood function or an ellipse approximation of the function is evaluated, the value of the MLE point is 0.0. 
- `boundary_col_indices`: a vector that is empty for the [`BivariateConfidenceStruct`](@ref) type and of length two for the [`UnivariateConfidenceStruct`](@ref) type. Contains the column indices in `points` of the confidence interval parameters for the [`UnivariateConfidenceStruct`](@ref) type. Default is an empty vector.
"""
struct PointsAndLogLikelihood
    points::Array{Float64}
    ll::Vector{<:Float64}
    boundary_col_indices::Vector{<:Int64}

    function PointsAndLogLikelihood(points, ll, boundary_col_indices=zeros(Int, 0))
        new(points, ll, boundary_col_indices)
    end
end

"""
    AbstractConfidenceStruct

Supertype for confidence boundary storage structs.

# Subtypes

[`SampledConfidenceStruct`](@ref)

[`UnivariateConfidenceStruct`](@ref)

[`BivariateConfidenceStruct`](@ref)
"""
abstract type AbstractConfidenceStruct end

"""
    SampledConfidenceStruct(points::Array{Float64}, ll::Vector{<:Float64})

Struct that stores samples produced by an [`AbstractSampleType`](@ref) that are within the confidence boundary at a given confidence level, with their corresponding loglikelihood values.

# Fields
- `points`: an array of points stored in columns, with each row corresponding to the respective index of each model parameter. 
- `ll`: a vector of loglikelihood function values corresponding to the point in each column of `points`. This number is standardised so that regardless of whether the true loglikelihood function or an ellipse approximation of the function is evaluated, the value of the MLE point is 0.0. 

# Supertype Hiearachy

SampledConfidenceStruct <: AbstractConfidenceStruct <: Any
"""
struct SampledConfidenceStruct <: AbstractSampledConfidenceStruct
    points::Array{Float64}
    ll::Vector{<:Float64}
end

"""
    UnivariateConfidenceStruct(confidence_interval::Vector{<:Float64}, interval_points::PointsAndLogLikelihood)

Struct that stores the confidence interval of a given interest parameter as well as points sampled within (and outside) the confidence interval and their corresponding loglikelihood values.

# Fields
- `confidence_interval`: a vector of length two with the confidence interval for a given interest parameter. If an entry has value `NaN`, that side of the confidence interval is outside the corresponding bound on the interest parameter.
- `interval_points`: a [`PointsAndLogLikelihood`](@ref) struct containing any points that have been evaluated inside or outside the interval by [`get_points_in_interval!`](@ref), their corresponding loglikelihood function value and the column indices of the `confidence_interval` points in `interval_points.points`. Points can be evaluated and stored that are outside the confidence interval so that loglikelihood profile plots are defined outside of the confidence interval. `interval_points.points` is stored in column-wise order of increasing interest parameter magnitude. 

# Supertype Hiearachy

UnivariateConfidenceStruct <: AbstractConfidenceStruct <: Any
"""
struct UnivariateConfidenceStruct <: AbstractUnivariateConfidenceStruct
    confidence_interval::Vector{<:Float64}
    interval_points::PointsAndLogLikelihood
end

"""
    BivariateConfidenceStruct(confidence_boundary::Matrix{Float64}, internal_points::PointsAndLogLikelihood=PointsAndLogLikelihood(zeros(size(x, 1), 0), zeros(0)))

Struct that stores samples produced by an [`AbstractBivariateMethod`](@ref) that are on the bivariate confidence boundary at a given confidence level and, if `save_internal_points=true`, any internal points found during the method with their corresponding loglikelihood values. Use `bivariate_methods()` for a list of available methods (see [`bivariate_methods`](@ref)).

# Fields
- `confidence_boundary`: an array of boundary points stored in columns, with each row corresponding to the respective index of each model parameter. This array can contain points that are inside the bivariate confidence boundary if the method being used brackets between an internal point and a point on the user-provided bounds: these points will be on a user-provided parameter bound.
- `internal_points`: a [`PointsAndLogLikelihood`](@ref) struct containing points and their corresponding loglikelihood values that were found during a method, if `save_internal_points=true`. Default is an empty [`PointsAndLogLikelihood`](@ref) struct (used if `save_internal_points=false`).

# Supertype Hiearachy

BivariateConfidenceStruct <: AbstractConfidenceStruct <: Any
"""
struct BivariateConfidenceStruct <: AbstractBivariateConfidenceStruct
    confidence_boundary::Matrix{Float64}
    internal_points::PointsAndLogLikelihood

    function BivariateConfidenceStruct(confidence_boundary, internal_points=PointsAndLogLikelihood(zeros(size(confidence_boundary, 1), 0), zeros(0)))
        return new(confidence_boundary, internal_points)
    end
end

"""
    Base.merge(a::BivariateConfidenceStruct, b::BivariateConfidenceStruct)

Specifies how to merge two variables with type [`BivariateConfidenceStruct`](@ref).
"""
function Base.merge(a::BivariateConfidenceStruct, b::BivariateConfidenceStruct)
    return BivariateConfidenceStruct(hcat(a.confidence_boundary, b.confidence_boundary),
        PointsAndLogLikelihood(hcat(a.internal_points.points, b.internal_points.points), vcat(a.internal_points.ll, b.internal_points.ll)))
end

