using Distributed
using Revise
using CSV, DataFrames
if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
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

if !isdefined(PlaceholderLikelihood, :find_zero_algo)

    if !isfile(joinpath(output_location, "confidence_interval_ll_calls.csv"))

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

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

    if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
        uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 1000, θ_true, collect(1:3), show_progress=true, distributed_over_parameters=false)
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

    if !isfile(joinpath(output_location, "confidence_profile_ll_calls.csv"))

        using Combinatorics

        function record_CI_LL_evaluations!(timer_df, method, method_key, num_points, iter)
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                bivariate_confidenceprofiles!(model, [pars], num_points, method=method, existing_profiles=:overwrite)

                timer_df[i+model.core.num_pars*(iter-1), :] .= pars[1], pars[2], method_key, num_points, TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
                TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
            return nothing
        end

        methods = [Fix1AxisMethod(), SimultaneousMethod(0.2, true), RadialRandomMethod(5, true), RadialMLEMethod(0.15, 0.1), IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1)]
        num_points_iter = collect(10:10:100)
        len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)
        timer_df = DataFrame(parameter1=zeros(Int, len), 
                                parameter2=zeros(Int, len), 
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
        
        CSV.write(joinpath(output_location, "confidence_profile_ll_calls.csv"), timer_df)
        CSV.write(joinpath(output_location, "methods.csv"), method_df)

        TO.disable_debug_timings(PlaceholderLikelihood)
    end

    if true || !isfile(joinpath(output_location, "bivariate_boundary_coverage.csv"))

        using Combinatorics

        function record_bivariate_boundary_coverage(method, method_key, num_points, hullmethods)
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=0.5,))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            biv_coverage_df = check_bivariate_boundary_coverage(data_generator, training_gen_args, model, 100, num_points, 2000, θ_true,
                collect(combinations(1:model.core.num_pars, 2)); method=method, distributed_over_parameters=false, hullmethod=hullmethods)

            biv_coverage_df.method_key .= method_key
            biv_coverage_df.num_points .= num_points
            return biv_coverage_df
        end

        methods = [Fix1AxisMethod(), SimultaneousMethod(0.2, true), RadialRandomMethod(5, true), RadialMLEMethod(0.15, 0.1), IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1)]
        num_points_iter = [10, 20, 40, 60, 80, 100]
        hullmethods = [MPPHullMethod(), ConvexHullMethod()]
        len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)

        method_df = DataFrame(method_key=collect(1:length(methods)),
            method_name=string.(methods))

        coverage_df = DataFrame()

        for (method_key, method) in enumerate(methods)
            for num_points in num_points_iter
                global coverage_df = vcat(coverage_df, record_bivariate_boundary_coverage(method, method_key, num_points, hullmethods))
            end
        end

        CSV.write(joinpath(output_location, "bivariate_boundary_coverage.csv"), coverage_df)
        CSV.write(joinpath(output_location, "methods.csv"), method_df)

    end
end