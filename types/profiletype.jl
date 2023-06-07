abstract type AbstractProfileType end
abstract type AbstractEllipseProfileType <: AbstractProfileType end
struct LogLikelihood <: AbstractProfileType end
struct EllipseApprox <: AbstractEllipseProfileType end
struct EllipseApproxAnalytical <: AbstractEllipseProfileType end