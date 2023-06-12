"""
    AbstractProfileType

Supertype for profile types.

# Subtypes

[`LogLikelihood`](@ref)

[`AbstractEllipseProfileType`](@ref)
"""
abstract type AbstractProfileType end

"""
    AbstractProfileType

Supertype for ellipse approximation profile types.

# Subtypes

[`EllipseApprox`](@ref)

[`EllipseApproxAnalytical`](@ref)

# Supertype Hiearachy

AbstractProfileType <: AbstractProfileType <: Any
"""
abstract type AbstractEllipseProfileType <: AbstractProfileType end

"""
    LogLikelihood()

Use the true loglikelihood function for confidence profile evaluation. The methods [`IterativeBoundaryMethod`](@ref) and [`RadialRandomMethod`](@ref) are recommended for use with this profile type.

# Supertype Hiearachy

LogLikelihood <: AbstractProfileType <: Any
"""
struct LogLikelihood <: AbstractProfileType end

"""
    EllipseApprox()

Use an ellipse approximation of the loglikelihood function centred at the MLE with use of parameter bounds for confidence profile evaluation. The method [`RadialMLEMethod`](@ref) is recommended for use with this profile type.

# Supertype Hiearachy

EllipseApprox <: AbstractEllipseProfileType <: AbstractProfileType <: Any
"""
struct EllipseApprox <: AbstractEllipseProfileType end

"""
    EllipseApproxAnalytical()

Use an ellipse approximation of the loglikelihood function centred at the MLE without use of parameter bounds for confidence profile evaluation. As no parameter bounds are involved, it can be analytically evaluated. The method [`AnalyticalEllipseMethod`](@ref) is recommended for use with this profile type - it analytically samples points on the confidence profile boundary using [EllipseSampling.jl](https://github.com/JoelTrent/EllipseSampling.jl). Other methods can be used, but they will all be unable to find interest parameter points outside user-provided parameter bounds (although nuisance parameters will be allowed outside these bounds).

# Supertype Hiearachy

EllipseApproxAnalytical <: AbstractEllipseProfileType <: AbstractProfileType <: Any
"""
struct EllipseApproxAnalytical <: AbstractEllipseProfileType end