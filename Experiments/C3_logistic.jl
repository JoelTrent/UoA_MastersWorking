using Distributed
using Revise
using CSV, DataFrames, Arrow
# if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO
@everywhere using Revise
@everywhere using Random, Distributions
@everywhere using PlaceholderLikelihood

@everywhere using Logging
@everywhere Logging.disable_logging(Logging.Warn) # Disable debug, info and warn

include(joinpath("Models", "logistic.jl"))
output_location = joinpath("Experiments", "Outputs", "logistic")

# do experiments
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
univariate_confidenceintervals!(model, optimizationsettings=opt_settings)

if !isfile(joinpath(output_location, "logistic_example.pdf"))
    using Plots; gr()
    using Plots.PlotMeasures
    using LaTeXStrings

    format = (palette=:Paired, size=(500, 400), dpi=300, title="", msw=0, legend_position=:bottomright, ylabel=L"C(t)", xlabel=L"t", minorgrid=true, minorticks=2, xlims=(0,1000), rightmargin=3mm)# ylims=(0.6, 1.0))
    plt = plot(; format...)

    lq, uq = reference_set_generator(θ_true, testing_gen_args, 0.95)
    plot!(testing_gen_args.t, lq, fillrange=uq, fillalpha=0.3, linealpha=0,
        label="95% population reference set")

    plot!(testing_gen_args.t, testing_gen_args.y_true, label="True model trajectory", lw=4)
    scatter!(data.t, data.y_obs, label="Example observations", msw=0, ms=7,)

    savefig(plt, joinpath(output_location, "logistic_example.pdf"))
end

if false || isdefined(PlaceholderLikelihood, :find_zero_algo) || !isfile(joinpath(output_location, "confidence_interval_ll_calls_algos.csv"))

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


if !isfile(joinpath(output_location, "confidence_interval_ll_calls.csv"))

    function record_CI_LL_evaluations!(timer_df, find_zero_atol, abstol, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=abstol))
        for i in 1:model.core.num_pars
            univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite, find_zero_atol=find_zero_atol, optimizationsettings=opt_settings)

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

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_xtol_rel.csv"))

    function record_CI_LL_evaluations!(timer_df, find_zero_atol, xtol_rel, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=xtol_rel, abstol=0.0))

        for i in 1:model.core.num_pars
            univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite, find_zero_atol=find_zero_atol, optimizationsettings=opt_settings)

            timer_df[i+model.core.num_pars*(iter-1), :] .= i, find_zero_atol, xtol_rel, TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    xtol_rels = round.(vcat(0.0, 10.0 .^ (-20:1:-2)), sigdigits=1)
    find_zero_atols = round.(vcat(0.0, 10.0 .^ (-20:1:-2)), sigdigits=1)
    len = model.core.num_pars * length(xtol_rels) * length(find_zero_atols)
    timer_df = DataFrame(parameter=zeros(Int, len),
        find_zero_atol=zeros(len),
        xtol_rel=zeros(len),
        optimisation_calls=zeros(Int, len),
        likelihood_calls=zeros(Int, len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter = 1
    for find_zero_atol in find_zero_atols
        for xtol_rel in xtol_rels
            record_CI_LL_evaluations!(timer_df, find_zero_atol, xtol_rel, iter)
            global iter += 1
        end
    end

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_xtol_rel.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_lower_and_upper.csv"))

    function record_CI_LL_evaluations!(timer_df, lower, upper, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
        if lower
            univariate_confidenceintervals!(model, confidence_level=0.90)
        end
        if upper
            univariate_confidenceintervals!(model, confidence_level=0.99)
        end

        TO.reset_timer!(PlaceholderLikelihood.timer)
        for i in 1:model.core.num_pars

            univariate_confidenceintervals!(model, [i], use_existing_profiles=true, existing_profiles=:overwrite)

            timer_df[i+model.core.num_pars*(iter-1), :] .= i, lower, upper, TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    lower = [false, true]
    upper = [false, true]
    len = model.core.num_pars * length(lower) * length(upper)
    timer_df = DataFrame(parameter=zeros(Int, len),
        lower_found=falses(len),
        upper_found=falses(len),
        optimisation_calls=zeros(Int, len),
        likelihood_calls=zeros(Int, len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter = 1
    for l in lower
        for u in upper
            record_CI_LL_evaluations!(timer_df, l, u, iter)
            global iter += 1
        end
    end

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_lower_and_upper.csv"), timer_df)
    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_ellipseapprox_start.csv"))

    function record_CI_LL_evaluations!(timer_df)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
        univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())

        TO.reset_timer!(PlaceholderLikelihood.timer)
        for i in 1:model.core.num_pars
            univariate_confidenceintervals!(model, [i], use_ellipse_approx_analytical_start=true, existing_profiles=:overwrite)

            timer_df[i, :] .= i, TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    len = model.core.num_pars 
    timer_df = DataFrame(parameter=zeros(Int, len),
        optimisation_calls=zeros(Int, len),
        likelihood_calls=zeros(Int, len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df)

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_ellipseapprox_start.csv"), timer_df)
    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_mean.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        total_opt_calls = zeros(Int, model.core.num_pars)
        total_ll_calls = zeros(Int, model.core.num_pars)

        function loglhood_new(Θ, data);
            return loglhood([exp(Θ[1]), Θ[2], Θ[3]], data)
        end
        θG_new, lb_new, ub_new = θG .* 1.0, lb .* 1.0, ub .* 1.0
        θG_new[1], lb_new[1], ub_new[1] = log(θG_new[1]), log(lb_new[1]+0.0001), log(ub_new[1])

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
            # model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
            model = initialise_LikelihoodModel(loglhood_new, predictFunc, errorFunc, training_data[j], θnames, θG_new, lb_new, ub_new, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
            for i in 1:model.core.num_pars

                TO.reset_timer!(PlaceholderLikelihood.timer)
                univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite, optimizationsettings=opt_settings)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:, 1] .= 1:model.core.num_pars
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = model.core.num_pars
    timer_df = DataFrame(parameter=zeros(Int, len),
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_mean.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_mean_ellipseapprox_start.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        total_opt_calls = zeros(Int, model.core.num_pars)
        total_ll_calls = zeros(Int, model.core.num_pars)

        function loglhood_new(Θ, data)
            return loglhood([exp(Θ[1]), Θ[2], Θ[3]], data)
        end
        θG_new, lb_new, ub_new = θG .* 1.0, lb .* 1.0, ub .* 1.0
        θG_new[1], lb_new[1], ub_new[1] = log(θG_new[1]), log(lb_new[1] + 0.0001), log(ub_new[1])

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
            # model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
            model = initialise_LikelihoodModel(loglhood_new, predictFunc, errorFunc, training_data[j], θnames, θG_new, lb_new, ub_new, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
            for i in 1:model.core.num_pars
                univariate_confidenceintervals!(model, [i], profile_type=EllipseApproxAnalytical(), existing_profiles=:overwrite)

                TO.reset_timer!(PlaceholderLikelihood.timer)
                univariate_confidenceintervals!(model, [i], use_ellipse_approx_analytical_start=true, existing_profiles=:overwrite, optimizationsettings=opt_settings)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:, 1] .= 1:model.core.num_pars
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = model.core.num_pars
    timer_df = DataFrame(parameter=zeros(Int, len),
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_mean_ellipseapprox_start.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_mean_ellipseapprox_start_internal_points.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        total_opt_calls = zeros(Int, model.core.num_pars)
        total_ll_calls = zeros(Int, model.core.num_pars)

        function loglhood_new(Θ, data)
            return loglhood([exp(Θ[1]), Θ[2], Θ[3]], data)
        end
        θG_new, lb_new, ub_new = θG .* 1.0, lb .* 1.0, ub .* 1.0
        θG_new[1], lb_new[1], ub_new[1] = log(θG_new[1]), log(lb_new[1] + 0.0001), log(ub_new[1])

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
            # model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
            model = initialise_LikelihoodModel(loglhood_new, predictFunc, errorFunc, training_data[j], θnames, θG_new, lb_new, ub_new, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
            for i in 1:model.core.num_pars
                univariate_confidenceintervals!(model, [i], profile_type=EllipseApproxAnalytical(), existing_profiles=:overwrite)

                TO.reset_timer!(PlaceholderLikelihood.timer)
                univariate_confidenceintervals!(model, [i], use_ellipse_approx_analytical_start=true, existing_profiles=:overwrite, optimizationsettings=opt_settings, num_points_in_interval=20)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:, 1] .= 1:model.core.num_pars
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = model.core.num_pars
    timer_df = DataFrame(parameter=zeros(Int, len),
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_mean_ellipseapprox_start_internal_points.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_mean_ellipseapprox_start_simultaneous_threshold_internal_points.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        total_opt_calls = zeros(Int, model.core.num_pars)
        total_ll_calls = zeros(Int, model.core.num_pars)

        function loglhood_new(Θ, data)
            return loglhood([exp(Θ[1]), Θ[2], Θ[3]], data)
        end
        θG_new, lb_new, ub_new = θG .* 1.0, lb .* 1.0, ub .* 1.0
        θG_new[1], lb_new[1], ub_new[1] = log(θG_new[1]), log(lb_new[1] + 0.0001), log(ub_new[1])

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
            # model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
            model = initialise_LikelihoodModel(loglhood_new, predictFunc, errorFunc, training_data[j], θnames, θG_new, lb_new, ub_new, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
            for i in 1:model.core.num_pars
                univariate_confidenceintervals!(model, [i], dof=model.core.num_pars, profile_type=EllipseApproxAnalytical(), existing_profiles=:overwrite)

                TO.reset_timer!(PlaceholderLikelihood.timer)
                univariate_confidenceintervals!(model, [i], dof=model.core.num_pars, use_ellipse_approx_analytical_start=true, existing_profiles=:overwrite, optimizationsettings=opt_settings, num_points_in_interval=20)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:, 1] .= 1:model.core.num_pars
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = model.core.num_pars
    timer_df = DataFrame(parameter=zeros(Int, len),
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_mean_ellipseapprox_start_simultaneous_threshold_internal_points.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
    Random.seed!(1234)
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

    opt_settings = default_OptimizationSettings()
    uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 1000, θ_true, collect(1:3), show_progress=true, distributed_over_parameters=false, optimizationsettings=opt_settings)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
end

if !isfile(joinpath(output_location, "uni_profile_1.pdf"))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    n=100
    additional_width=0.2
    univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical(), num_points_in_interval=n, additional_width=additional_width)
    univariate_confidenceintervals!(model, profile_type=LogLikelihood(), num_points_in_interval=n, additional_width=additional_width)

    using Plots; gr()
    format=(size=(400,400), dpi=300, title="", legend_position=:topright)
    plts = plot_univariate_profiles_comparison(model; label_only_lines=true, format...)

    for (i, plt) in enumerate(plts)
        if i<length(plts); plot!(plts[i], legend_position=nothing) end
        savefig(plts[i], joinpath(output_location, "uni_profile_"*string(i)*".pdf"))
    end
end

if !isfile(joinpath(output_location, "uni_profile_ellipse_1.pdf"))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    n = 100
    additional_width = 0.2
    univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical(), num_points_in_interval=n, additional_width=additional_width)
    univariate_confidenceintervals!(model, profile_type=EllipseApprox(), num_points_in_interval=n, additional_width=additional_width)

    using Plots
    gr()
    format = (size=(400, 400), dpi=300, title="", legend_position=:topright)
    plts = plot_univariate_profiles_comparison(model; label_only_lines=true, format...)

    for (i, plt) in enumerate(plts)
        if i<length(plts); plot!(plts[i], legend_position=nothing) end
        savefig(plts[i], joinpath(output_location, "uni_profile_ellipse_" * string(i) * ".pdf"))
    end
end

if !isfile(joinpath(output_location, "uni_profile_ellipse_adjustlbup_1.pdf"))

    lb_new = [lb[1:2]..., 5]
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb_new, ub, par_magnitudes, optimizationsettings=opt_settings)

    n = 100
    additional_width = 0.2
    univariate_confidenceintervals!(model, [1], profile_type=EllipseApproxAnalytical(), num_points_in_interval=n, additional_width=additional_width)
    univariate_confidenceintervals!(model, [1], profile_type=EllipseApprox(), num_points_in_interval=n, additional_width=additional_width)

    using Plots
    gr()
    format = (size=(400, 400), dpi=300, title="")
    plts = plot_univariate_profiles_comparison(model; label_only_lines=true, format...)

    for (i, plt) in enumerate(plts)
        plot!(plts[1], legend_position=nothing)
        savefig(plts[i], joinpath(output_location, "uni_profile_ellipse_adjustlbup_" * string(i) * ".pdf"))
    end

    lb_new = [lb[1:2]..., -5]
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb_new, ub, par_magnitudes, optimizationsettings=opt_settings)

    n = 100
    additional_width = 0.2
    univariate_confidenceintervals!(model, [1], profile_type=EllipseApproxAnalytical(), num_points_in_interval=n, additional_width=additional_width)
    univariate_confidenceintervals!(model, [1], profile_type=EllipseApprox(), num_points_in_interval=n, additional_width=additional_width)

    using Plots
    gr()
    plts = plot_univariate_profiles_comparison(model; label_only_lines=true, format...)

    for (i, plt) in enumerate(plts)
        plot!(plts[1], legend_position=:topright)
        savefig(plts[i], joinpath(output_location, "uni_profile_ellipse_adjustlbdown_" * string(i) * ".pdf"))
    end
end

if !isfile(joinpath(output_location, "biv_profile_1.pdf"))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    bivariate_confidenceprofiles!(model, 100, method=AnalyticalEllipseMethod(0.0, 1.0))
    bivariate_confidenceprofiles!(model, 100, method=IterativeBoundaryMethod(20, 5,5, 0.15, 1.0, use_ellipse=true), profile_type=LogLikelihood())

    using Plots
    gr()
    format = (size=(400, 400), dpi=300, title="", legend_position=:topright)
    plts = plot_bivariate_profiles_comparison(model; label_only_MLE=true, format...)

    for (i, plt) in enumerate(plts)
        if i < length(plts)
            plot!(plts[i], legend_position=nothing)
        end
        savefig(plts[i], joinpath(output_location, "biv_profile_" * string(i) * ".pdf"))
    end
end

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls.csv"))

    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, method, method_key, num_points, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
            bivariate_confidenceprofiles!(model, [pars], num_points, method=method, existing_profiles=:overwrite)

            timer_df[i+model.core.num_pars*(iter-1), :] .= pars, method_key, num_points, TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    methods = [Fix1AxisMethod(), SimultaneousMethod(0.5, true), RadialRandomMethod(5, true), RadialMLEMethod(0.15, 0.1), IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), 
        IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=false)]
    num_points_iter = collect(10:10:100)
    len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)
    timer_df = DataFrame(θindices=[zeros(Int,2) for _ in len],
                            method_key=zeros(Int, len), 
                            num_points=zeros(Int, len),
                            optimisation_calls=zeros(Int, len),
                            likelihood_calls=zeros(Int, len))

    method_df = DataFrame(method_key=collect(1:length(methods)),
        method_name=string.(methods))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter=1
    for (method_key, method) in enumerate(methods)
        for num_points in num_points_iter
            record_CI_LL_evaluations!(timer_df, method, method_key, num_points, iter)
            global iter+=1
        end
    end
    
    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls.csv"), timer_df)
    CSV.write(joinpath(output_location, "methods.csv"), method_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls_widerbounds.csv"))

    using Combinatorics
    lb_wider = [0.0, 25., 0.0]
    ub_wider = [0.075, 175., 75.]

    function record_CI_LL_evaluations!(timer_df, method, method_key, num_points, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb_wider, ub_wider, par_magnitudes, optimizationsettings=opt_settings)

        for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
            bivariate_confidenceprofiles!(model, [pars], num_points, method=method, existing_profiles=:overwrite)

            timer_df[i+model.core.num_pars*(iter-1), :] .= pars, method_key, num_points, TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    methods = [Fix1AxisMethod(), SimultaneousMethod(0.5, true), RadialRandomMethod(5, true), RadialMLEMethod(0.15, 0.1), IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), 
        IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=false)]
    num_points_iter = collect(10:10:100)
    len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)
    timer_df = DataFrame(θindices=[zeros(Int,2) for _ in len],
                            method_key=zeros(Int, len), 
                            num_points=zeros(Int, len),
                            optimisation_calls=zeros(Int, len),
                            likelihood_calls=zeros(Int, len))

    method_df = DataFrame(method_key=collect(1:length(methods)),
        method_name=string.(methods))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter=1
    for (method_key, method) in enumerate(methods)
        for num_points in num_points_iter
            record_CI_LL_evaluations!(timer_df, method, method_key, num_points, iter)
            global iter+=1
        end
    end
    
    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls_widerbounds.csv"), timer_df)
    CSV.write(joinpath(output_location, "methods.csv"), method_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls_mean.csv"))
    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]

        len_combos = length(collect(combinations(1:model.core.num_pars, 2)))
        total_opt_calls = zeros(Int, len_combos)
        total_ll_calls = zeros(Int, len_combos)

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20,))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                TO.reset_timer!(PlaceholderLikelihood.timer)
                bivariate_confidenceprofiles!(model, [pars], 50, method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), existing_profiles=:overwrite,
                    use_distributed=false, use_threads=false, optimizationsettings=opt_settings)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:, 1] .= collect(combinations(1:model.core.num_pars, 2))
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = length(collect(combinations(1:model.core.num_pars, 2)))
    timer_df = DataFrame(θindices=[zeros(Int, 2) for _ in len],
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls_mean.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold.csv"))
    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]

        len_combos = length(collect(combinations(1:model.core.num_pars, 2)))
        total_opt_calls = zeros(Int, len_combos)
        total_ll_calls = zeros(Int, len_combos)

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20,))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                TO.reset_timer!(PlaceholderLikelihood.timer)
                bivariate_confidenceprofiles!(model, [pars], 50, method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), existing_profiles=:overwrite,
                    use_distributed=false, use_threads=false, optimizationsettings=opt_settings, dof=model.core.num_pars)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:, 1] .= collect(combinations(1:model.core.num_pars, 2))
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = length(collect(combinations(1:model.core.num_pars, 2)))
    timer_df = DataFrame(θindices=[zeros(Int, 2) for _ in len],
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold_20pnts.csv"))
    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]

        len_combos = length(collect(combinations(1:model.core.num_pars, 2)))
        total_opt_calls = zeros(Int, len_combos)
        total_ll_calls = zeros(Int, len_combos)

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20,))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                TO.reset_timer!(PlaceholderLikelihood.timer)
                bivariate_confidenceprofiles!(model, [pars], 20, method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), existing_profiles=:overwrite,
                    use_distributed=false, use_threads=false, optimizationsettings=opt_settings, dof=model.core.num_pars)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:, 1] .= collect(combinations(1:model.core.num_pars, 2))
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = length(collect(combinations(1:model.core.num_pars, 2)))
    timer_df = DataFrame(θindices=[zeros(Int, 2) for _ in len],
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold_20pnts.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold_20pnts_xtol_rel.csv"))
    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]

        len_combos = length(collect(combinations(1:model.core.num_pars, 2)))
        total_opt_calls = zeros(Int, len_combos)
        total_ll_calls = zeros(Int, len_combos)

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20,))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-8))

            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                TO.reset_timer!(PlaceholderLikelihood.timer)
                bivariate_confidenceprofiles!(model, [pars], 20, method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), existing_profiles=:overwrite,
                    use_distributed=false, use_threads=false, optimizationsettings=opt_settings, dof=model.core.num_pars)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:, 1] .= collect(combinations(1:model.core.num_pars, 2))
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = length(collect(combinations(1:model.core.num_pars, 2)))
    timer_df = DataFrame(θindices=[zeros(Int, 2) for _ in len],
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold_20pnts_xtol_rel.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "bivariate_confidence_set_opt_calls_vs_points.csv"))

    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, method, num_points, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        
        for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            TO.reset_timer!(PlaceholderLikelihood.timer)

            if num_points < 51
                bivariate_confidenceprofiles!(model, [pars], num_points, method=method, existing_profiles=:overwrite)
    
                timer_df[i+model.core.num_pars*(iter-1), 2:end] .= pars, string(method), num_points+length(model.biv_profiles_dict[1].internal_points.ll), TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
                TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])
            else
                bivariate_confidenceprofiles!(model, [pars], 50, method=method, existing_profiles=:overwrite)
                sample_bivariate_internal_points!(model, num_points-50)

                timer_df[i+model.core.num_pars*(iter-1), 2:end] .= pars, string(method), num_points + length(model.biv_profiles_dict[1].internal_points.ll),
                    TO.ncalls(
                        PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"])+
                    TO.ncalls(
                        PlaceholderLikelihood.timer["Sample bivariate internal points"]["Likelihood nuisance parameter optimisation"]),
                    TO.ncalls(
                        PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])+
                    TO.ncalls(
                        PlaceholderLikelihood.timer["Sample bivariate internal points"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])
            end

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    methods = [IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true)]
    num_points_iter = collect(10:10:100)
    num_sample_points_iter = collect(100:100:3000)
    len = length(collect(combinations(1:model.core.num_pars, 2))) * (length(num_points_iter) + length(num_sample_points_iter))
    timer_df = DataFrame(iter=[div(i-1,3)+1 for i in 1:len],
                            θindices=[zeros(Int,2) for _ in len],
                            method=fill("", len), 
                            num_points=zeros(Int, len),
                            optimisation_calls=zeros(Int, len),
                            likelihood_calls=zeros(Int, len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter=1
    for (method_key, method) in enumerate(methods)
        for num_points in num_points_iter
            record_CI_LL_evaluations!(timer_df, method, num_points, iter)
            global iter+=1
        end
    end

    function record_dim_LL_evaluations!(timer_df, num_points, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        
        for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
            TO.reset_timer!(PlaceholderLikelihood.timer)

            dimensional_likelihood_samples!(model, [pars], num_points)
            timer_df[i+model.core.num_pars*(iter-1), 2:end] .= pars, "Latin Hypercube", model.dim_samples_df[1, :num_points],
            TO.ncalls(
                PlaceholderLikelihood.timer["Dimensional likelihood sample"]["Likelihood nuisance parameter optimisation"]), 
            TO.ncalls(
                PlaceholderLikelihood.timer["Dimensional likelihood sample"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    for num_points in num_sample_points_iter
        record_dim_LL_evaluations!(timer_df, num_points, iter)
        global iter+=1
    end
    
    CSV.write(joinpath(output_location, "bivariate_confidence_set_opt_calls_vs_points.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "bivariate_confidence_set_opt_calls_vs_points_narrowerbounds.csv"))

    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, method, num_points, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))

        for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, [0.005, 75, 0.0], [0.0300, 125, 25], par_magnitudes, optimizationsettings=opt_settings)

            TO.reset_timer!(PlaceholderLikelihood.timer)

            if num_points < 51
                bivariate_confidenceprofiles!(model, [pars], num_points, method=method, θlb_nuisance=lb, θub_nuisance=ub, existing_profiles=:overwrite)

                timer_df[i+model.core.num_pars*(iter-1), 2:end] .= pars, string(method), num_points + length(model.biv_profiles_dict[1].internal_points.ll), TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
                TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])
            else
                bivariate_confidenceprofiles!(model, [pars], 50, method=method, existing_profiles=:overwrite, θlb_nuisance=lb, θub_nuisance=ub)
                sample_bivariate_internal_points!(model, num_points - 50, θlb_nuisance=lb, θub_nuisance=ub)

                timer_df[i+model.core.num_pars*(iter-1), 2:end] .= pars, string(method), num_points + length(model.biv_profiles_dict[1].internal_points.ll),
                TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]) +
                TO.ncalls(
                    PlaceholderLikelihood.timer["Sample bivariate internal points"]["Likelihood nuisance parameter optimisation"]),
                TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"]) +
                TO.ncalls(
                    PlaceholderLikelihood.timer["Sample bivariate internal points"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])
            end

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    methods = [IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true)]
    num_points_iter = collect(10:10:200)
    num_sample_points_iter = collect(100:100:3000)
    len = length(collect(combinations(1:model.core.num_pars, 2))) * (length(num_points_iter) + length(num_sample_points_iter))
    timer_df = DataFrame(iter=[div(i - 1, 3) + 1 for i in 1:len],
        θindices=[zeros(Int, 2) for _ in len],
        method=fill("", len),
        num_points=zeros(Int, len),
        optimisation_calls=zeros(Int, len),
        likelihood_calls=zeros(Int, len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter = 1
    for (method_key, method) in enumerate(methods)
        for num_points in num_points_iter
            record_CI_LL_evaluations!(timer_df, method, num_points, iter)
            global iter += 1
        end
    end

    function record_dim_LL_evaluations!(timer_df, num_points, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))

        for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, [0.005, 75, 0.0], [0.0300, 125, 25], par_magnitudes, optimizationsettings=opt_settings)
            TO.reset_timer!(PlaceholderLikelihood.timer)

            dimensional_likelihood_samples!(model, [pars], num_points, θlb_nuisance=lb, θub_nuisance=ub)
            timer_df[i+model.core.num_pars*(iter-1), 2:end] .= pars, "Latin Hypercube", model.dim_samples_df[1, :num_points],
            TO.ncalls(
                PlaceholderLikelihood.timer["Dimensional likelihood sample"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Dimensional likelihood sample"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    for num_points in num_sample_points_iter
        record_dim_LL_evaluations!(timer_df, num_points, iter)
        global iter += 1
    end

    CSV.write(joinpath(output_location, "bivariate_confidence_set_opt_calls_vs_points_narrowerbounds.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end



if !isfile(joinpath(output_location, "confidence_boundary_ll_calls_xtol_rel.csv"))

    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, method, num_points, xtol_rel, conf_level, iter)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=xtol_rel))

        for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
            bivariate_confidenceprofiles!(model, [pars], num_points, method=method, existing_profiles=:overwrite, optimizationsettings=opt_settings,
            confidence_level=conf_level)

            timer_df[i+model.core.num_pars*(iter-1), :] .= pars, string(method),  num_points, xtol_rel, conf_level, TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    methods = [IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true)]
    num_points_iter = collect(30:10:50)
    xtol_rel_iter = [0.0, 1e-20, 1e-16, 1e-12, 1e-8, 1e-6]
    conf_levels = [0.95, 0.979906]
    len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * 
        length(num_points_iter) * length(xtol_rel_iter) * length(conf_levels)
    timer_df = DataFrame(θindices=[zeros(Int, 2) for _ in len],
        method_name=fill(string(methods[1]), len),
        num_points=zeros(Int, len),
        xtol_rel=zeros(len),
        conf_level=zeros(len),
        optimisation_calls=zeros(Int, len),
        likelihood_calls=zeros(Int, len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter = 1
    for method in methods
        for num_points in num_points_iter
            for xtol_rel in xtol_rel_iter
                for conf_level in conf_levels
                    record_CI_LL_evaluations!(timer_df, method, num_points, xtol_rel, conf_level, iter)
                    global iter += 1
                end
            end
        end
    end

    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls_xtol_rel.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "bivariate_boundary_coverage.csv"))

    using Combinatorics

    function record_bivariate_boundary_coverage(method, method_key, num_points, hullmethods)
        Random.seed!(1234)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
        biv_coverage_df = check_bivariate_boundary_coverage(data_generator, training_gen_args, model, 500, num_points, 4000, θ_true,
            collect(combinations(1:model.core.num_pars, 2)); method=method, distributed_over_parameters=false, hullmethod=hullmethods, 
            coverage_estimate_quantile_level=0.9,
            optimizationsettings=opt_settings)

        biv_coverage_df.method_key .= method_key
        biv_coverage_df.num_points .= num_points
        return biv_coverage_df
    end

    methods = [Fix1AxisMethod(), SimultaneousMethod(0.5, true), RadialRandomMethod(5, true), RadialMLEMethod(0.15, 0.1), IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=false)]
    num_points_iter = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    hullmethods = [MPPHullMethod(), ConvexHullMethod()]
    len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)

    method_df = DataFrame(method_key=collect(1:length(methods)),
        method_name=string.(methods))
    CSV.write(joinpath(output_location, "methods.csv"), method_df)

    coverage_df = DataFrame()

    for (method_key, method) in enumerate(methods)
        for num_points in num_points_iter
            global coverage_df = vcat(coverage_df, record_bivariate_boundary_coverage(method, method_key, num_points, hullmethods))
            CSV.write(joinpath(output_location, "bivariate_boundary_coverage.csv"), coverage_df)
        end
    end
end

if !isfile(joinpath(output_location, "bivariate_parameter_coverage.csv"))
    using Combinatorics
    Random.seed!(1234)
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
    biv_coverage_df = check_bivariate_parameter_coverage(data_generator, training_gen_args, model, 1000, 50, θ_true, collect(combinations(1:model.core.num_pars, 2)),
                        method = IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), 
                        show_progress=true, distributed_over_parameters=false, optimizationsettings = opt_settings)
    display(biv_coverage_df)
    CSV.write(joinpath(output_location, "bivariate_parameter_coverage.csv"), biv_coverage_df)
end

if !isfile(joinpath(output_location, "full_sampling_prediction_coverage.csv"))        
    num_points_iter = collect(5000:5000:40000)
    coverage_df = DataFrame()
    
    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=false)

        new_df = filter(:θname => !=(:union), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_prediction_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_prediction_coverage.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "univariate_prediction_coverage.csv"))
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:20:60)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, θ_true, collect(1:model.core.num_pars),
            num_points_in_interval=num_points, show_progress=true, distributed_over_parameters=false, manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_prediction_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_prediction_coverage.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "univariate_prediction_coverage_simultaneous_threshold.csv"))
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:20:60)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, θ_true, collect(1:model.core.num_pars),
            num_points_in_interval=num_points, show_progress=true, distributed_over_parameters=false, dof=model.core.num_pars,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_prediction_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_prediction_coverage_simultaneous_threshold.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_prediction_coverage.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:40:80)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 50, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:40:80)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 50, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false, 
            dof=model.core.num_pars,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "full_sampling_realisation_coverage.csv"))

    num_points_iter = collect(5000:5000:40000)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, 
            model, 1000, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=false)

        new_df = filter(:θname => !=(:union), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_realisation_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_realisation_coverage.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "univariate_realisation_coverage.csv"))
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:20:60)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred,
            model, 1000, θ_true, collect(1:model.core.num_pars),
            show_progress=true, num_points_in_interval=num_points, distributed_over_parameters=false,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_realisation_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_realisation_coverage.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.csv"))
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:20:60)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred,
            model, 1000, θ_true, collect(1:model.core.num_pars),
            show_progress=true, num_points_in_interval=num_points, distributed_over_parameters=false,
            dof=model.core.num_pars, manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_realisation_coverage.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:40:80)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 50, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_realisation_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_realisation_coverage.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:40:80)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 50, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            dof=model.core.num_pars,
            show_progress=true, distributed_over_parameters=false,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold.arrow"), coverage_df)
    end
end

#####################################################################################################################
#####################################################################################################################

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_two_combinations.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(10:10:50)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, num_points, θ_true, [[1,2], [1,3]],
            method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
            show_progress=true, distributed_over_parameters=false,
            dof=model.core.num_pars,
            optimizationsettings=opt_settings)

        new_df.num_boundary_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_two_combinations.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_two_combinations.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_two_combinations_xtol_rel.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-8))

    num_points_iter = collect(20:10:20)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, num_points, θ_true, [[1,2], [1,3]],
            method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
            show_progress=true, distributed_over_parameters=false,
            dof=model.core.num_pars,
            optimizationsettings=opt_settings)

        new_df.num_boundary_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_two_combinations_xtol_rel.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_two_combinations_xtol_rel.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_20points_xtol_rel.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-8))

    num_points_iter = collect(20:10:20)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, num_points, θ_true, [[1, 2], [1, 3], [2, 3]],
            method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
            show_progress=true, distributed_over_parameters=false,
            dof=model.core.num_pars,
            optimizationsettings=opt_settings)

        new_df.num_boundary_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_20points_xtol_rel.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_20points_xtol_rel.arrow"), coverage_df)
    end
end