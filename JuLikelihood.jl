using NLopt, Roots
# using NLsolve, LineSearches
using ForwardDiff
using Random
using StatsBase
using Combinatorics
using DataStructures
using LinearAlgebra
using EllipseSampling
using LatinHypercubeSampling
using DataFrames
using Accessors
using Distributed
using FLoops

using ProgressMeter
global const PROGRESS__METER__DT = 0.2

include("NLopt_optimiser.jl")
include("model_structs.jl")
include("model_initialiser.jl")
include("combination_relationships.jl")
include("transform_bounds.jl")
include("common_profile_likelihood.jl")
include("ellipse_likelihood.jl")
include("full_likelihood_sampling.jl")
include("dimensional_likelihood_sampling.jl")
include("predictions.jl")
include("plotting_functions.jl")

include("univariate_methods/init_and_array_mapping.jl")
include("univariate_methods/loglikelihood_functions.jl")
include("univariate_methods/univariate_profile_likelihood.jl")
include("univariate_methods/points_in_interval.jl")

include("bivariate_methods/init_and_array_mapping.jl")
include("bivariate_methods/findpointon2Dbounds.jl")
include("bivariate_methods/loglikelihood_functions.jl")
include("bivariate_methods/fix1axis.jl")
include("bivariate_methods/vectorsearch.jl")
include("bivariate_methods/continuation.jl")
include("bivariate_methods/bivariate_profile_likelihood.jl")