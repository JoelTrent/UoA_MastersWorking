abstract type AbstractSampleType end
struct UniformGridSamples <: AbstractSampleType end
struct UniformRandomSamples <: AbstractSampleType end
struct LatinHypercubeSamples <: AbstractSampleType end

abstract type AbstractConfidenceStruct end
abstract type AbstractUnivariateConfidenceStruct <: AbstractConfidenceStruct end
abstract type AbstractBivariateConfidenceStruct <: AbstractConfidenceStruct end
abstract type AbstractSampledConfidenceStruct <: AbstractConfidenceStruct end

struct PointsAndLogLikelihood
    points::Array{Float64}
    ll::Vector{<:Float64}
    boundary_col_indices::Vector{<:Int64}

    function PointsAndLogLikelihood(x, y, z=zeros(Int, 0))
        new(x, y, z)
    end
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

    function BivariateConfidenceStruct(x, y=PointsAndLogLikelihood(zeros(size(x, 1), 0), zeros(0)))
        return new(x, y)
    end
end

function Base.merge(a::BivariateConfidenceStruct, b::BivariateConfidenceStruct)
    return BivariateConfidenceStruct(hcat(a.confidence_boundary, b.confidence_boundary),
        PointsAndLogLikelihood(hcat(a.internal_points.points, b.internal_points.points), vcat(a.internal_points.ll, b.internal_points.ll)))
end

