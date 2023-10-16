using Distributed
using Revise
using CSV, DataFrames
# if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO
@everywhere using Revise
@everywhere using DifferentialEquations, LSODA, Random, Distributions
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "STAT5.jl"))
output_location = joinpath("Experiments", "Outputs", "STAT5");

# do experiments
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes);

# univariate_confidenceintervals!(model)
# get_points_in_intervals!(model, 100, additional_width=0.2)

# generate_predictions_univariate!(model, t_pred, 1.0, use_distributed=false)

# using Plots; pyplot()

# plots = plot_univariate_profiles(model, 0.2, 0.4, palette_to_use=:Spectral_8)
# for i in eachindex(plots); display(plots[i]) end
# plot = plot_predictions_union(model, t_pred)

if true || isdefined(PlaceholderLikelihood, :find_zero_algo) || !isfile(joinpath(output_location, "confidence_interval_ll_calls_algos.csv"))

    using Roots
    function record_CI_LL_evaluations!(timer_df, algo, algo_key, iter)
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes)

        PlaceholderLikelihood.find_zero_algo = algo

        display(algo)
        sleep(0.5)

        for i in 1:model.core.num_pars
            univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite)

            timer_df[i+model.core.num_pars*(iter-1), :] .= i, algo_key, TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    find_zero_algos = [Roots.Bisection(), Roots.A42(), Roots.AlefeldPotraShi(), Roots.Brent(),
        Roots.Chandrapatla(), Roots.Ridders(), Roots.ITP(), Roots.FalsePosition()]

    len = model.core.num_pars * length(find_zero_algos)
    timer_df = DataFrame(parameter=zeros(Int, len),
        algo_key=zeros(Int, len),
        optimisation_calls=zeros(Int, len),
        likelihood_calls=zeros(Int, len))

    algo_df = DataFrame(algo_key=collect(1:length(find_zero_algos)),
        algo_name=string.(find_zero_algos))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter = 1
    for (i, algo) in enumerate(find_zero_algos)
        record_CI_LL_evaluations!(timer_df, algo, i, iter)
        global iter += 1
    end

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_algos.csv"), timer_df)
    CSV.write(joinpath(output_location, "algos.csv"), algo_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end


if !isdefined(PlaceholderLikelihood, :find_zero_algo)
    if true || !isfile(joinpath(output_location, "confidence_interval_ll_calls.csv"))

        function record_CI_LL_evaluations!(timer_df, abstol, iter)
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=abstol))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            for i in 1:model.core.num_pars
                univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite)

                timer_df[i+model.core.num_pars*(iter-1), :] .= i, abstol, TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
                TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
            return nothing
        end

        abstols = [0.0, 1e-20, 1e-16, 1e-12, 1e-8]
        len = model.core.num_pars * length(abstols)
        timer_df = DataFrame(parameter=zeros(Int, len), opt_abstol=zeros(len),
            optimisation_calls=zeros(Int, len),
            likelihood_calls=zeros(Int, len))

        TO.enable_debug_timings(PlaceholderLikelihood)
        TO.reset_timer!(PlaceholderLikelihood.timer)

        for (iter, abstol) in enumerate(abstols)
            record_CI_LL_evaluations!(timer_df, abstol, iter)
        end

        CSV.write(joinpath(output_location, "confidence_interval_ll_calls.csv"), timer_df)

        TO.disable_debug_timings(PlaceholderLikelihood)
    end

    if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
        uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 100, θ_true, collect(1:9), show_progress=true, distributed_over_parameters=false)
        display(uni_coverage_df)
        CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
    end
end