using NLopt, Roots
using ForwardDiff
using Random
using StatsBase
using Combinatorics
using DataStructures

include("NLopt_optimiser.jl")
include("model_structs.jl")
include("model_initialiser.jl")
include("combination_relationships.jl")
include("transform_bounds.jl")
include("common_profile_likelihood.jl")
include("ellipse_likelihood.jl")
include("profile_likelihood_V2.jl")