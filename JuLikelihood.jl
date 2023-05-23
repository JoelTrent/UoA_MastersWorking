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
using ConcaveHull
using Distances, TravelingSalesmanHeuristics
using Clustering, Meshes

using ProgressMeter
global const PROGRESS__METER__DT = 1.0

include("NLopt_optimiser.jl")
include("model_structs.jl")
include("model_initialiser.jl")
include("combination_relationships.jl")
include("transform_bounds.jl")
include("common_profile_likelihood.jl")
include("ellipse_likelihood.jl")
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
include("bivariate_methods/continuation_polygon_manipulation.jl")
include("bivariate_methods/continuation.jl")
include("bivariate_methods/bivariate_profile_likelihood.jl")
include("bivariate_methods/MPP_TSP.jl")

include("dimensional_methods/full_likelihood_sampling.jl")
include("dimensional_methods/dimensional_likelihood_sampling.jl")
include("dimensional_methods/bivariate_concave_hull.jl")