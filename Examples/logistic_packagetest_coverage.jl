# Section 1: set up packages and parameter values
# using BenchmarkTools

using Distributed
using LikelihoodBasedProfileWiseAnalysis
if nprocs()==1; addprocs(10) end
# if nprocs()==1; addprocs(4) end
@everywhere using Revise
@everywhere using DifferentialEquations, Random, Distributions
@everywhere using LikelihoodBasedProfileWiseAnalysis

@everywhere using Logging
@everywhere Logging.disable_logging(Logging.Warn) # Disable debug, info and warn

# Workflow functions ##########################################################################

@everywhere function solvedmodel(t, a)
    return (a[2]*a[3]) ./ ((a[2]-a[3]) .* (exp.(-a[1] .* t)) .+ a[3])
end

# Section 6: Define loglikelihood function
@everywhere function loglhood(a, data)
    # y=ODEmodel(data.t, a)
    y=solvedmodel(data.t, a)
    e=0
    e=sum(loglikelihood(data.dist, data.yobs .- y))
    return e
end

# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
@everywhere function funmle(a)
    return loglhood(a, data)
end

# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000; 
@everywhere σ=10.0;
tt=0:5:1000
a=[λ, K, C0]
θtrue=[λ, K, C0]

# true data
ytrue = solvedmodel(t, a)

Random.seed!(12348)
# noisy data
yobs = ytrue + σ*randn(length(t))
yobs = ytrue .+ rand(Normal(0, σ), length(t))

# Named tuple of all data required within the likelihood function
data = (yobs=yobs, σ=σ, t=t, dist=Normal(0, σ))

# Bounds on model parameters #################################################################
λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50., 150.)
C0min, C0max = (0.0, 50.)

θG = [λ, K, C0]
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]
par_magnitudes = [0.005, 10, 10]

θnames = [:λ, :K, :C0]
@everywhere function predictFunc(θ, data, t=data.t); solvedmodel(t, θ) end
@everywhere function errorFunc(predictions, θ, bcl); normal_error_σ_known(predictions, θ, bcl, σ) end

model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes);

univariate_confidenceintervals!(model)
generate_predictions_univariate!(model, tt, 1.0)

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    yobs = generator_args.ytrue .+ rand(generator_args.dist, length(generator_args.t))
    if generator_args.is_test_set
        return yobs
    end
    data = (yobs=yobs, generator_args...)
    return data
end
gen_args = (ytrue=ytrue, σ=σ, t=t, dist=Normal(0, σ), is_test_set=false)

# PARAMETER COVERAGE CHECKS
# uni_coverage_df = check_univariate_parameter_coverage(data_generator, gen_args, model, 100, θtrue, collect(1:3), show_progress=true, distributed_over_parameters=false)
# println(uni_coverage_df)

# biv_coverage_df = check_bivariate_parameter_coverage(data_generator, gen_args, model, 100, 50, θtrue, [[1, 2], [1, 3], [2, 3]], show_progress=true, distributed_over_parameters=true)
# println(biv_coverage_df)

# biv_coverage_df = check_bivariate_parameter_coverage(data_generator, gen_args, model, 100, [20, 30], θtrue, [[1, 2], [1, 3], [2, 3]], 
#     method=[RadialMLEMethod(0.0), RadialRandomMethod(3, false)], show_progress=true, distributed_over_parameters=true)
# println(biv_coverage_df)

# BIVARIATE THEORETICAL BOUNDARY COVERAGE CHECKS
# biv_coverage_df = check_bivariate_boundary_coverage(data_generator, gen_args, model, 10, 50, -1, θtrue, [[1, 2], [1, 3], [2, 3]], method=IterativeBoundaryMethod(30, 0, 10), show_progress=true, distributed_over_parameters=true, hullmethod=ConvexHullMethod())
# println(biv_coverage_df)

# biv_coverage_df = check_bivariate_boundary_coverage(data_generator, gen_args, model, 100, [20, 30], 2000, θtrue, [[1, 2], [1, 3], [2, 3]], 
#     method=[IterativeBoundaryMethod(15, 0, 5), RadialRandomMethod(2, false)], show_progress=true, distributed_over_parameters=true, 
#     hullmethod=ConvexHullMethod())
# println(biv_coverage_df)

# biv_coverage_df = check_bivariate_boundary_coverage(data_generator, gen_args, model, 500, 50, 2000, θtrue, [[1, 2], [1, 3], [2, 3]], method=IterativeBoundaryMethod(30, 0, 10), show_progress=true, distributed_over_parameters=true, hullmethod=ConcaveHullMethod())
# println(biv_coverage_df)

# biv_coverage_df = check_bivariate_boundary_coverage(data_generator, gen_args, model, 500, 50, 2000, θtrue, [[1, 2], [1, 3], [2, 3]], method=IterativeBoundaryMethod(30, 0, 10), show_progress=true, distributed_over_parameters=true, hullmethod=MPPHullMethod())
# # println(biv_coverage_df)

# PREDICTION COVERAGE ###############################################################################################################################
# FOR THE MEAN/MEDIAN
# Random.seed!(1)
# uni_prediction_coverage_df = check_univariate_prediction_coverage(data_generator, gen_args, collect(tt), model, 100, θtrue, collect(1:3), num_points_in_interval=100, show_progress=true, distributed_over_parameters=false)
# display(uni_prediction_coverage_df)

# Random.seed!(1)
# biv_prediction_coverage_df = check_bivariate_prediction_coverage(data_generator, gen_args, collect(tt), model, 100, [20, 30], θtrue, [[1, 2], [1, 3], [2, 3]], 
#     method=[RadialMLEMethod(0.0), RadialRandomMethod(3, false)], hullmethod=ConvexHullMethod(), num_internal_points=100, show_progress=true, distributed_over_parameters=true)
# display(biv_prediction_coverage_df)

# Random.seed!(1)
# dim_prediction_coverage_df = check_dimensional_prediction_coverage(data_generator, gen_args, collect(tt), model, 50, 5000, θtrue, [[3], [1,2], [2,3], [1, 2, 3]], show_progress=true, distributed_over_parameters=false)
# display(dim_prediction_coverage_df)

# FOR REALISATIONS
test_gen_args = (ytrue=ytrue, σ=σ, t=t, dist=Normal(0, σ), is_test_set=true)
Random.seed!(1)
uni_prediction_coverage_df = check_univariate_prediction_realisations_coverage(data_generator, gen_args, test_gen_args, collect(t), model, 1000, θtrue, collect(1:3), num_points_in_interval=100, show_progress=true, distributed_over_parameters=false)
display(uni_prediction_coverage_df)

Random.seed!(1)
biv_prediction_coverage_df = check_bivariate_prediction_realisations_coverage(data_generator, gen_args, test_gen_args, collect(t), model, 200, [20, 30], θtrue, [[1, 2], [1, 3], [2, 3]], 
    method=[RadialMLEMethod(0.0), RadialRandomMethod(3, false)], hullmethod=ConvexHullMethod(), num_internal_points=100, show_progress=true, distributed_over_parameters=false)
display(biv_prediction_coverage_df)

Random.seed!(1)
dim_prediction_coverage_df = check_dimensional_prediction_realisations_coverage(data_generator, gen_args, test_gen_args, collect(t), model, 1000, 100000, θtrue, [[1, 2, 3]], show_progress=true, distributed_over_parameters=false)
display(dim_prediction_coverage_df)
rmprocs(workers())