using Distributed
using Revise
using CSV, DataFrames, Arrow
if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO
@everywhere using Revise
@everywhere using Random, Distributions, DifferentialEquations, StaticArrays
@everywhere using PlaceholderLikelihood

@everywhere using Logging
@everywhere Logging.disable_logging(Logging.Warn) # Disable debug, info and warn

include(joinpath("Models", "lotka_volterra_5par.jl"))
output_location = joinpath("Experiments", "Outputs", "lotka_volterra_5par")

# do experiments
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
univariate_confidenceintervals!(model, optimizationsettings=opt_settings)


if isfile(joinpath(output_location, "full_sampling_prediction_coverage.csv"))
    num_points_iter = [500000, 1000000, 5000000]#, 10000000]
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=false, manual_GC_calls=true)

        new_df = filter(:n_random_combinations => ==(0), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_prediction_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_prediction_coverage.arrow"), coverage_df)
    end
    sleep(1)
end

if isfile(joinpath(output_location, "full_sampling_realisation_coverage.csv"))

    num_points_iter = [500000, 1000000, 5000000]#, 10000000]
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, 
            model, 1000, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=false, manual_GC_calls=true)

        new_df = filter(:n_random_combinations => ==(0), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_realisation_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_realisation_coverage.arrow"), coverage_df)
    end
end

if isfile(joinpath(output_location, "full_sampling_prediction_coverage_more_data.csv"))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb_more_data, ub_more_data, par_magnitudes, optimizationsettings=opt_settings)
    num_points_iter = [50000, 100000, 200000]
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_coverage(data_generator, training_gen_args_more_data, t_pred, model, 1000, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=false, manual_GC_calls=true)

        new_df = filter(:n_random_combinations => ==(0), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_prediction_coverage_more_data.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_prediction_coverage_more_data.arrow"), coverage_df)
    end
end

if isfile(joinpath(output_location, "full_sampling_realisation_coverage_more_data.csv"))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb_more_data, ub_more_data, par_magnitudes, optimizationsettings=opt_settings)
    num_points_iter = [50000, 100000, 200000]
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args_more_data, testing_gen_args, t_pred,
            model, 1000, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=false, manual_GC_calls=true)

        new_df = filter(:n_random_combinations => ==(0), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_realisation_coverage_more_data.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_realisation_coverage_more_data.arrow"), coverage_df)
    end
end


if !isfile(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.csv"))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:80)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred,
            model, 1000, θ_true, collect(1:model.core.num_pars),
            show_progress=true, num_points_in_interval=num_points, distributed_over_parameters=false,
            dof=model.core.num_pars,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.arrow"), coverage_df)
        @everywhere GC.gc()
        sleep(1)
    end
end