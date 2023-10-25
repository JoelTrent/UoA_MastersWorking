using Distributed
using Revise
using CSV, DataFrames
# if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO
@everywhere using Revise
@everywhere using DifferentialEquations, Random, Distributions
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "STAT5.jl"))
output_location = joinpath("Experiments", "Outputs", "STAT5");

# do experiments
opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20, local_method=NLopt.LN_NELDERMEAD()))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes);

# univariate_confidenceintervals!(model)
# get_points_in_intervals!(model, 100, additional_width=0.2)

# generate_predictions_univariate!(model, t_pred, 1.0, use_distributed=false)

# using Plots; pyplot()

# plots = plot_univariate_profiles(model, 0.2, 0.4, palette_to_use=:Spectral_8)
# for i in eachindex(plots); display(plots[i]) end
# plot = plot_predictions_union(model, t_pred)

if isdefined(PlaceholderLikelihood, :find_zero_algo) || !isfile(joinpath(output_location, "confidence_interval_ll_calls_algos.csv"))

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
    if !isfile(joinpath(output_location, "confidence_interval_ll_calls_xtol_rel.csv"))

        function record_CI_LL_evaluations!(timer_df, xtol_rel, iter)
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, local_method=NLopt.LN_NELDERMEAD()))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=xtol_rel, local_method=NLopt.LN_NELDERMEAD()))

            for i in 1:model.core.num_pars
                univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite, optimizationsettings=opt_settings)

                timer_df[i+model.core.num_pars*(iter-1), :] .= i, xtol_rel, TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
                TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
            return nothing
        end

        xtol_rel_iter = [0.0, 1e-20, 1e-16, 1e-12, 1e-8]
        len = model.core.num_pars * length(xtol_rel_iter)
        timer_df = DataFrame(parameter=zeros(Int, len), xtol_rel=zeros(len),
            optimisation_calls=zeros(Int, len),
            likelihood_calls=zeros(Int, len))

        TO.enable_debug_timings(PlaceholderLikelihood)
        TO.reset_timer!(PlaceholderLikelihood.timer)

        for (iter, xtol_rel) in enumerate(xtol_rel_iter)
            record_CI_LL_evaluations!(timer_df, xtol_rel, iter)
        end

        CSV.write(joinpath(output_location, "confidence_interval_ll_calls_xtol_rel.csv"), timer_df)

        TO.disable_debug_timings(PlaceholderLikelihood)
    end

    if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
        Random.seed!(1234)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12, local_method=NLopt.LN_NELDERMEAD()))
        uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 250, θ_true, collect(1:9), show_progress=true, distributed_over_parameters=false, optimizationsettings=opt_settings)
        display(uni_coverage_df)
        CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
    end

    if !isfile(joinpath(output_location, "uni_profile_1.pdf"))

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20,))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        n = 40
        additional_width = 0.2
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=10, xtol_rel=1e-12, local_method=NLopt.LN_NELDERMEAD()))
        # univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical(), num_points_in_interval=n, additional_width=additional_width)
        univariate_confidenceintervals!(model, num_points_in_interval=n, additional_width=additional_width,
                                        optimizationsettings=opt_settings)

        using Plots
        gr()
        format = (size=(400, 400), dpi=300, title="", legend_position=:topright)
        plts = plot_univariate_profiles(model; label_only_lines=true, format...)

        for (i, plt) in enumerate(plts)
            if i < length(plts)
                plot!(plts[i], legend_position=nothing)
            end
            savefig(plts[i], joinpath(output_location, "uni_profile_" * string(i) * ".pdf"))
        end
    end

    if !isfile(joinpath(output_location, "biv_profile_1.pdf"))

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, local_method=NLopt.LN_NELDERMEAD()))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12, local_method=NLopt.LN_NELDERMEAD()))
        # bivariate_confidenceprofiles!(model, 100, method=AnalyticalEllipseMethod(0.0, 1.0))
        bivariate_confidenceprofiles!(model, 20, method=IterativeBoundaryMethod(10, 5, 5, 0.15, 1.0, use_ellipse=false), optimizationsettings=opt_settings)

        using Plots
        gr()
        format = (size=(400, 400), dpi=300, title="", legend_position=:topright)
        plts = plot_bivariate_profiles(model; label_only_MLE=true, format...)

        for (i, plt) in enumerate(plts)
            if i < length(plts)
                plot!(plts[i], legend_position=nothing)
            end
            savefig(plts[i], joinpath(output_location, "biv_profile_" * string(i) * ".pdf"))
        end
    end
end