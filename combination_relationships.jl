abstract type AbstractRelationship end

# a - b -> will need to check that a != b
struct AMinusB <: AbstractRelationship
    a::Symbol
    b::Symbol
    lb::Float64
    ub::Float64
    name::Symbol
end

# a + b -> will need to check that a != b
struct APlusB <: AbstractRelationship
    a::Symbol
    b::Symbol
    lb::Float64
    ub::Float64
    name::Symbol
end

struct ATimesB <: AbstractRelationship
    a::Symbol
    b::Symbol
    lb::Float64
    ub::Float64
    name::Symbol
end

struct ADivB <: AbstractRelationship
    a::Symbol
    b::Symbol
    lb::Float64
    ub::Float64
    name::Symbol
end

function Ψmlegivenrelationship(θmle, indexA, indexB, relationship::AMinusB)
    return θmle[indexA] - θmle[indexB] 
end

function Ψmlegivenrelationship(θmle, indexA, indexB, relationship::APlusB)
    return θmle[indexA] + θmle[indexB] 
end

function Ψmlegivenrelationship(θmle, indexA, indexB, relationship::ATimesB)
    return θmle[indexA] * θmle[indexB] 
end

function Ψmlegivenrelationship(θmle, indexA, indexB, relationship::ADivB)
    return θmle[indexA] / θmle[indexB] 
end

function resolve_relationship!(θs, Ψ, λ, ind, λind, relationship::AMinusB)
    θs[ind] = Ψ + λ[λind]
    return nothing
end

function resolve_relationship!(θs, Ψ, λ, ind, λind, relationship::APlusB)
    θs[ind] = Ψ - λ[λind]
    return nothing
end

function resolve_relationship!(θs, Ψ, λ, ind, λind, relationship::ATimesB)
    θs[ind] = Ψ / λ[λind]
    return nothing
end

function resolve_relationship!(θs, Ψ, λ, ind, λind, relationship::ADivB)
    θs[ind] = Ψ * λ[λind]
    return nothing
end