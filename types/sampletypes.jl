"""
    AbstractSampleType

Supertype for sample types.

# Subtypes

[`UniformGridSamples`](@ref)

[`UniformRandomSamples`](@ref)

[`LatinHypercubeSamples`](@ref)
"""
abstract type AbstractSampleType end

"""
    UniformGridSamples()

Evaluate the parameter bounds space on a uniform grid and keep samples that are inside the particular `profile_type`'s boundary. 

# Supertype Hiearachy

UniformGridSamples <: AbstractSampleType <: Any
"""
struct UniformGridSamples <: AbstractSampleType end

"""
    UniformRandomSamples()

Take uniform random samples of parameter bounds space and keep samples that are inside the particular `profile_type`'s boundary. 

# Supertype Hiearachy

UniformRandomSamples <: AbstractSampleType <: Any
"""
struct UniformRandomSamples <: AbstractSampleType end

"""
    LatinHypercubeSamples()

Create a Latin Hypercube sampling plan in parameter bounds space and keep samples that are inside the particular `profile_type`'s boundary. Uses [LatinHypercubeSampling.jl](https://github.com/MrUrq/LatinHypercubeSampling.jl).

# Supertype Hiearachy

LatinHypercubeSamples <: AbstractSampleType <: Any
"""
struct LatinHypercubeSamples <: AbstractSampleType end
