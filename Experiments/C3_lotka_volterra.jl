using Distributed
using Revise
using CSV, DataFrames, Arrow
if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO
@everywhere using Revise
@everywhere using Random, Distributions, DifferentialEquations, StaticArrays
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "lotka_volterra.jl"))
output_location = joinpath("Experiments", "Outputs", "lotka_volterra")

# do experiments
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

if !isfile(joinpath(output_location, "lotka_volterra_example.pdf"))
    using Plots; gr()
    using Plots.PlotMeasures
    using LaTeXStrings

    format = (palette=:Paired, size=(500, 400), dpi=300, title="", msw=0,
        legend_position=:topright, minorgrid=true, minorticks=2, rightmargin=3mm,
        background_color_legend=RGBA(1, 1, 1, 0.6), xlims=(0,10), ylims=(-0.2, 2.6))

    t = collect(LinRange(0, 10, 21))
    Random.seed!(12348)
    # noisy data
    y_obs = hcat(ODEmodel(t, θ_true)...) .+ σ * randn(21, 2)

    plt1 = plot(; format..., legend_position=nothing,)

    lq, uq = reference_set_generator(θ_true, testing_gen_args, 0.95)

    vspan!([0, 7], label="", linealpha=0, alpha=0.15, color=4)
    vspan!([7, 10], label="", linealpha=0, alpha=0.15, color=8)

    plot!(testing_gen_args.t, lq[:,1], fillrange=uq[:,1], fillalpha=0.3, linealpha=0,
        label="95% population reference set", ylabel=L"x(t)", color=1)

    plot!(testing_gen_args.t, testing_gen_args.y_true[:,1], label="True model trajectory", lw=4, color=2)
    scatter!(t, y_obs[:,1], label="Example observations", msw=0, ms=7,color=3)

    plt2 = plot(; format...,  xlabel = L"t")

    vspan!([0, 7], label="", linealpha=0, alpha=0.15, color=4)
    vspan!([7, 10], label="", linealpha=0, alpha=0.15, color=8)

    plot!([0, 1], [-100, -150], fillrange=[-200, -250], label="Estimation and prediction", linealpha=0, alpha=0.15, color=4)
    plot!([0, 1], [-100, -150], fillrange=[-200, -250], label="Prediction only", linealpha=0, alpha=0.15, color=8)

    plot!(testing_gen_args.t, lq[:, 2], fillrange=uq[:, 2], fillalpha=0.3, linealpha=0,
    label="95% population reference set", ylabel=L"y(t)", color=1)
    plot!(testing_gen_args.t, testing_gen_args.y_true[:, 2], label="True model trajectory", lw=4, color=2)
    scatter!(t, y_obs[:, 2], label="Example observations", msw=0, ms=7, color=3)

    plt = plot(plt1, plt2, layout=(2,1))

    savefig(plt, joinpath(output_location, "lotka_volterra_example.pdf"))
end


if !isfile(joinpath(output_location, "confidence_interval_ll_calls.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        total_opt_calls=zeros(Int, model.core.num_pars)
        total_ll_calls=zeros(Int, model.core.num_pars)
        
        for j in 1:N 
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
            for i in 1:model.core.num_pars
                univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite, optimizationsettings=opt_settings)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:,1] .= 1:model.core.num_pars
        timer_df[:,2] .= total_opt_calls ./ N
        timer_df[:,3] .= total_ll_calls ./ N 
        return nothing
    end

    len = model.core.num_pars 
    timer_df = DataFrame(parameter=zeros(Int, len),
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_ellipseapprox_start.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        total_opt_calls = zeros(Int, model.core.num_pars)
        total_ll_calls = zeros(Int, model.core.num_pars)

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

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

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_ellipseapprox_start.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_interval_ll_calls_ellipseapprox_start_simultaneous_threshold.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        total_opt_calls = zeros(Int, model.core.num_pars)
        total_ll_calls = zeros(Int, model.core.num_pars)

        equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 1)

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
            for i in 1:model.core.num_pars
                univariate_confidenceintervals!(model, [i], confidence_level=equiv_simul_conf_level, profile_type=EllipseApproxAnalytical(), existing_profiles=:overwrite)

                TO.reset_timer!(PlaceholderLikelihood.timer)
                univariate_confidenceintervals!(model, [i], confidence_level=equiv_simul_conf_level, use_ellipse_approx_analytical_start=true, existing_profiles=:overwrite, optimizationsettings=opt_settings)

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

    CSV.write(joinpath(output_location, "confidence_interval_ll_calls_ellipseapprox_start_simultaneous_threshold.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end


if !isfile(joinpath(output_location, "univariate_parameter_coverage.csv"))
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
    uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 1000, θ_true, collect(1:4), 
                        show_progress=true, distributed_over_parameters=false, optimizationsettings=opt_settings)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
end

if !isfile(joinpath(output_location, "uni_profile_1.pdf"))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    n = 100
    additional_width = 0.2
    univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical(), num_points_in_interval=n, additional_width=additional_width)
    univariate_confidenceintervals!(model, profile_type=LogLikelihood(), num_points_in_interval=n, additional_width=additional_width)

    using Plots
    gr()
    format = (size=(400, 400), dpi=300, title="", legend_position=:topright)
    plts = plot_univariate_profiles_comparison(model; label_only_lines=true, format...)

    for (i, plt) in enumerate(plts)
        if i < length(plts)
            plot!(plts[i], legend_position=nothing)
        end
        savefig(plts[i], joinpath(output_location, "uni_profile_" * string(i) * ".pdf"))
    end
end

if !isfile(joinpath(output_location, "biv_profile_1.pdf"))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    bivariate_confidenceprofiles!(model, 100, method=AnalyticalEllipseMethod(0.0, 1.0))
    bivariate_confidenceprofiles!(model, 100, method=RadialMLEMethod(0.15, 0.5), profile_type=LogLikelihood())

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

        len_combos = length(collect(combinations(1:model.core.num_pars, 2)))

        for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
            bivariate_confidenceprofiles!(model, [pars], num_points, method=method, existing_profiles=:overwrite, use_distributed=false, use_threads=false)

            timer_df[i + len_combos*(iter-1), :] .= pars, method_key, num_points, TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

            TO.reset_timer!(PlaceholderLikelihood.timer)
        end
        return nothing
    end

    methods = [IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true)]
    num_points_iter = collect(10:10:50)
    len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)
    timer_df = DataFrame(θindices=[zeros(Int, 2) for _ in len],
        method_key=zeros(Int, len),
        num_points=zeros(Int, len),
        optimisation_calls=zeros(Int, len),
        likelihood_calls=zeros(Int, len))

    method_df = DataFrame(method_key=collect(1:length(methods)),
        method_name=string.(methods))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    iter = 1
    for (method_key, method) in enumerate(methods)
        for num_points in num_points_iter
            if method isa IterativeBoundaryMethod && method.initial_num_points > num_points
                new_method = IterativeBoundaryMethod(num_points, method.angle_points_per_iter, method.edge_points_per_iter,
                    method.radial_start_point_shift, method.ellipse_sqrt_distortion, use_ellipse=method.use_ellipse)
            else
                new_method = method
            end
            record_CI_LL_evaluations!(timer_df, new_method, method_key, num_points, iter)
            global iter += 1
        end
    end

    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls.csv"), timer_df)
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
                bivariate_confidenceprofiles!(model, [pars], 30, method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true), existing_profiles=:overwrite,
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

        equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20,))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                TO.reset_timer!(PlaceholderLikelihood.timer)
                bivariate_confidenceprofiles!(model, [pars], 30, method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true), existing_profiles=:overwrite,
                    use_distributed=false, use_threads=false, optimizationsettings=opt_settings, confidence_level=equiv_simul_conf_level)

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

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold_10points_xtol_rel.csv"))
    using Combinatorics

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]

        len_combos = length(collect(combinations(1:model.core.num_pars, 2)))
        total_opt_calls = zeros(Int, len_combos)
        total_ll_calls = zeros(Int, len_combos)

        equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

        for j in 1:N
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20,))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-8))

            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                TO.reset_timer!(PlaceholderLikelihood.timer)
                bivariate_confidenceprofiles!(model, [pars], 10, method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), existing_profiles=:overwrite,
                    use_distributed=false, use_threads=false, optimizationsettings=opt_settings, confidence_level=equiv_simul_conf_level)

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

    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold_10points_xtol_rel.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "bivariate_boundary_coverage.csv"))

    using Combinatorics

    function record_bivariate_boundary_coverage(method, method_key, num_points, hullmethods)
        Random.seed!(1234)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        biv_coverage_df = check_bivariate_boundary_coverage(data_generator, training_gen_args, model, 100, num_points, 5000, θ_true,
            collect(combinations(1:model.core.num_pars, 2)); method=method, distributed_over_parameters=false, hullmethod=hullmethods,
            coverage_estimate_quantile_level=0.9)

        biv_coverage_df.method_key .= method_key
        biv_coverage_df.num_points .= num_points
        return biv_coverage_df
    end

    methods = [IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true)]
    num_points_iter = [10, 20, 30, 40]
    hullmethods = [MPPHullMethod(), ConvexHullMethod()]
    len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)

    method_df = DataFrame(method_key=collect(1:length(methods)),
        method_name=string.(methods))
    CSV.write(joinpath(output_location, "methods.csv"), method_df)

    coverage_df = DataFrame()

    for (method_key, method) in enumerate(methods)
        for num_points in num_points_iter
            if method isa IterativeBoundaryMethod && method.initial_num_points > num_points 
                new_method = IterativeBoundaryMethod(num_points, method.angle_points_per_iter, method.edge_points_per_iter,
                    method.radial_start_point_shift, method.ellipse_sqrt_distortion, use_ellipse=method.use_ellipse)
            else
                new_method = method
            end
            global coverage_df = vcat(coverage_df, record_bivariate_boundary_coverage(new_method, method_key, num_points, hullmethods))
            CSV.write(joinpath(output_location, "bivariate_boundary_coverage.csv"), coverage_df)
        end
    end
end

if !isfile(joinpath(output_location, "bivariate_parameter_coverage.csv"))
    using Combinatorics
    Random.seed!(1234)
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
    biv_coverage_df = check_bivariate_parameter_coverage(data_generator, training_gen_args, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
        method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
        show_progress=true, distributed_over_parameters=false, optimizationsettings=opt_settings)
    display(biv_coverage_df)
    CSV.write(joinpath(output_location, "bivariate_parameter_coverage.csv"), biv_coverage_df)
end

if !isfile(joinpath(output_location, "full_sampling_prediction_coverage.csv"))
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))

    num_points_iter = [10000, 50000, 250000, 500000]#, 1000000]
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

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 1)
    # PlaceholderLikelihood.get_target_loglikelihood(model, 0.95, LogLikelihood(), model.core.num_pars) ≈ 
    # PlaceholderLikelihood.get_target_loglikelihood(model, equiv_simul_conf_level, LogLikelihood(), 1)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, θ_true, collect(1:model.core.num_pars),
            num_points_in_interval=num_points, show_progress=true, distributed_over_parameters=false, confidence_level=equiv_simul_conf_level, 
            manual_GC_calls=true, optimizationsettings=opt_settings)

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
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
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

    num_points_iter = collect(0:40:40)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "full_sampling_realisation_coverage.csv"))

    num_points_iter = [10000, 50000, 250000, 500000]
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
            show_progress=true, num_points_in_interval=num_points, distributed_over_parameters=false, manual_GC_calls=true,
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
    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 1)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred,
            model, 1000, θ_true, collect(1:model.core.num_pars),
            show_progress=true, num_points_in_interval=num_points, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level, manual_GC_calls=true,
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
        new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator, 
            training_gen_args, testing_gen_args, t_pred, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
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

    num_points_iter = collect(0:40:40)
    coverage_df = DataFrame()
    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator,
            training_gen_args, testing_gen_args, t_pred, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold.arrow"), coverage_df)
    end
end

#############################################################################################################################

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_three_combinations_a.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:40:0)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, [[1,4],[2,3],[2,4]],
            method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_three_combinations_a.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_three_combinations_a.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:40:0)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, [[1, 4], [2, 3], [2, 4], [3,4]],
            method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations.arrow"), coverage_df)
    end
end


if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_three_combinations_b.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(0:40:0)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, [[1,2],[2,4],[3,4]],
            method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_three_combinations_b.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_three_combinations_b.arrow"), coverage_df)
    end
end

#############################################################################################################################

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_less_points.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

    num_points_iter = collect(10:10:20)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, num_points, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(num_points, 5, 5, 0.15, 0.1, use_ellipse=true),
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            optimizationsettings=opt_settings)

        new_df.num_boundary_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_less_points.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_less_points.arrow"), coverage_df)
    end
end


if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_less_points_xtol_rel.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-8))

    num_points_iter = collect(10:10:10)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, num_points, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=IterativeBoundaryMethod(num_points, 5, 5, 0.15, 0.1, use_ellipse=true),
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            optimizationsettings=opt_settings)

        new_df.num_boundary_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_less_points_xtol_rel.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_less_points_xtol_rel.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations_less_points_xtol_rel.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-8))

    num_points_iter = collect(0:40:0)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, [[1, 4], [2, 3], [2, 4], [3, 4]],
            method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations_less_points_xtol_rel.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations_less_points_xtol_rel.arrow"), coverage_df)
    end
end
