using Distributed
using Revise
using CSV, DataFrames
# if nprocs()==1; addprocs(10) end
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO
@everywhere using Revise
@everywhere using Random, Distributions
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "logistic.jl"))
output_location = joinpath("Experiments", "Outputs", "logistic")

# do experiments
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

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

        function record_CI_LL_evaluations!(timer_df, find_zero_atol, abstol, iter)
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=abstol))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            for i in 1:model.core.num_pars
                univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite, find_zero_atol=find_zero_atol)

                timer_df[i+model.core.num_pars*(iter-1), :] .= i, find_zero_atol, abstol, TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
                TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
            return nothing
        end

        abstols = round.(vcat(0.0, 10.0 .^ (-20:1:-2)), sigdigits=1)
        find_zero_atols = round.(vcat(0.0, 10.0 .^ (-20:1:-2)), sigdigits=1)
        len = model.core.num_pars*length(abstols)*length(find_zero_atols)
        timer_df = DataFrame(parameter=zeros(Int, len), 
                                find_zero_atol=zeros(len), 
                                opt_abstol=zeros(len), 
                                optimisation_calls=zeros(Int, len),
                                likelihood_calls=zeros(Int, len))

        TO.enable_debug_timings(PlaceholderLikelihood)
        TO.reset_timer!(PlaceholderLikelihood.timer)

        iter=1
        for find_zero_atol in find_zero_atols
            for abstol in abstols
                record_CI_LL_evaluations!(timer_df, find_zero_atol, abstol, iter)
                global iter+=1
            end
        end
        
        CSV.write(joinpath(output_location, "confidence_interval_ll_calls.csv"), timer_df)

        TO.disable_debug_timings(PlaceholderLikelihood)
    end

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

    if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
        uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 1000, θ_true, collect(1:3), show_progress=true, distributed_over_parameters=false)
        display(uni_coverage_df)
        CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
    end
end