using Distributed
using Revise
using CSV, DataFrames, Arrow
if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO
@everywhere using Revise
@everywhere using Random, Distributions, DifferentialEquations
@everywhere using PlaceholderLikelihood

# @everywhere using Logging
# @everywhere Logging.disable_logging(Logging.Warn) # Disable debug, info and warn

include(joinpath("Models", "logistic_twospecies_logitnormal.jl"))
output_location = joinpath("Experiments", "Outputs", "logistic_twospecies_logitnormal");

# do experiments
opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

using Combinatorics
# bivariate_combinations = vcat(collect(combinations([1,4,5,7], 2)), [[2,3]])

# getMLE_ellipse_approximation!(model)
# inds = [3,6]; model.ellipse_MLE_approx.Hmle[inds, inds]


if !isfile(joinpath(output_location, "confidence_interval_ll_calls.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        total_opt_calls=zeros(Int, model.core.num_pars)
        total_ll_calls=zeros(Int, model.core.num_pars)

        for j in eachindex(training_data)
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, abstol=0.0))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
            for i in 1:model.core.num_pars
                # univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite, optimizationsettings=opt_settings,
                #     θlb_nuisance=lb_nuisance, θub_nuisance=ub_nuisance)
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
        optimisation_calls=zeros(len),
        likelihood_calls=zeros(len))

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
                univariate_confidenceintervals!(model, [i], profile_type=EllipseApproxAnalytical(), existing_profiles=:overwrite)

                TO.reset_timer!(PlaceholderLikelihood.timer)
                univariate_confidenceintervals!(model, [i], use_ellipse_approx_analytical_start=true, existing_profiles=:overwrite, confidence_level=equiv_simul_conf_level, optimizationsettings=opt_settings)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
            println("Iteration: ", j)
        end

        timer_df[:, 1] .= 1:model.core.num_pars
        timer_df[:, 2] .= total_opt_calls ./ N
        timer_df[:, 3] .= total_ll_calls ./ N
        return nothing
    end

    len = model.core.num_parso
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
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
    uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 1000, θ_true, collect(1:7), 
                        # θlb_nuisance=lb_nuisance, θub_nuisance=ub_nuisance, 
                        show_progress=true, distributed_over_parameters=false,
                        optimizationsettings=opt_settings)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage.csv"), uni_coverage_df)
end


# if !isfile(joinpath(output_location, "univariate_parameter_coverage_more_data.csv"))
#     opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
#     uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args_more_data, model, 1000, θ_true, collect(1:7), 
#                         # θlb_nuisance=lb_nuisance, θub_nuisance=ub_nuisance, 
#                         show_progress=true, distributed_over_parameters=false,
#                         optimizationsettings=opt_settings)
#     display(uni_coverage_df)
#     CSV.write(joinpath(output_location, "univariate_parameter_coverage_more_data.csv"), uni_coverage_df)
# end

if !isfile(joinpath(output_location, "univariate_parameter_coverage_significantly_more_data.csv"))
    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
    uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args_more_data, model, 1000, θ_true, collect(1:7),
        show_progress=true, distributed_over_parameters=false,
        optimizationsettings=opt_settings)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage_significantly_more_data.csv"), uni_coverage_df)
end

# if !isfile(joinpath(output_location, "univariate_parameter_coverage_higher_threshold.csv"))
#     opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
#     uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 1000, θ_true, collect(1:7), 
#                         confidence_level=0.98,
#                         # θlb_nuisance=lb_nuisance, θub_nuisance=ub_nuisance, 
#                         show_progress=true, distributed_over_parameters=false,
#                         optimizationsettings=opt_settings)
#     display(uni_coverage_df)
#     CSV.write(joinpath(output_location, "univariate_parameter_coverage_higher_threshold.csv"), uni_coverage_df)
# end

if !isfile(joinpath(output_location, "uni_profile_1.pdf"))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=120,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=120, xtol_rel=1e-12))

    n = 40
    additional_width = 0.2
    univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical(), num_points_in_interval=n, additional_width=additional_width)
    univariate_confidenceintervals!(model, profile_type=LogLikelihood(),
        use_distributed=false, num_points_in_interval=n,
        additional_width=additional_width, optimizationsettings=opt_settings)

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

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, ))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
    bivariate_confidenceprofiles!(model, 100, method=AnalyticalEllipseMethod(0.0, 0.01))
    bivariate_confidenceprofiles!(model, 100, method=RadialMLEMethod(0.15, 0.01), profile_type=LogLikelihood(),
        optimizationsettings=opt_settings)

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



# opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, local_method=NLopt.LN_NELDERMEAD(), xtol_rel=1e-12))
# model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
# bivariate_confidenceprofiles!(model, 20, θlb_nuisance=lb_nuisance, θub_nuisance=ub_nuisance, method=IterativeBoundaryMethod(10, 5, 5, 0.15, 1.0, use_ellipse=true), existing_profiles=:overwrite)

# if !isfile(joinpath(output_location, "confidence_boundary_ll_calls.csv"))

#     using Combinatorics

#     function record_CI_LL_evaluations!(timer_df, method, method_key, num_points, iter)
#         opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
#         model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

#         len_combos = length(collect(combinations(1:model.core.num_pars, 2)))

#         for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
#             println(pars)
#             bivariate_confidenceprofiles!(model, [pars], num_points, method=method, existing_profiles=:overwrite, use_distributed=false, use_threads=false)

#             timer_df[i+len_combos*(iter-1), :] .= pars, method_key, num_points, TO.ncalls(
#                 PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
#             TO.ncalls(
#                 PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

#             TO.reset_timer!(PlaceholderLikelihood.timer)
#         end
#         return nothing
#     end

#     methods = [RadialMLEMethod(0.15, 0.01)]#[IterativeBoundaryMethod(30, 5, 5, 0.15, 0.1, use_ellipse=true)]
#     # num_points_iter = collect(10:10:60)
#     num_points_iter = collect(30:10:30)
#     len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)
#     timer_df = DataFrame(θindices=[zeros(Int, 2) for _ in len],
#         method_key=zeros(Int, len),
#         num_points=zeros(Int, len),
#         optimisation_calls=zeros(Int, len),
#         likelihood_calls=zeros(Int, len))

#     method_df = DataFrame(method_key=collect(1:length(methods)),
#         method_name=string.(methods))

#     TO.enable_debug_timings(PlaceholderLikelihood)
#     TO.reset_timer!(PlaceholderLikelihood.timer)

#     iter = 1
#     for (method_key, method) in enumerate(methods)
#         for num_points in num_points_iter
#             record_CI_LL_evaluations!(timer_df, method, method_key, num_points, iter)
#             global iter += 1
#         end
#     end

#     CSV.write(joinpath(output_location, "confidence_boundary_ll_calls.csv"), timer_df)

#     TO.disable_debug_timings(PlaceholderLikelihood)
# end

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        
        len_combos = length(collect(combinations(1:model.core.num_pars, 2)))
        total_opt_calls=zeros(Int, len_combos)
        total_ll_calls=zeros(Int, len_combos)

        for j in 1:N 
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, ))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
            
            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                TO.reset_timer!(PlaceholderLikelihood.timer)
                bivariate_confidenceprofiles!(model, [pars], 30, method=RadialMLEMethod(0.15, 0.01), existing_profiles=:overwrite, 
                    use_distributed=false, use_threads=false, optimizationsettings=opt_settings)

                total_opt_calls[i] +=  TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
        end

        timer_df[:,1] .= collect(combinations(1:model.core.num_pars, 2))
        timer_df[:,2] .= total_opt_calls ./ N
        timer_df[:,3] .= total_ll_calls ./ N 
        return nothing
    end

    len = length(collect(combinations(1:model.core.num_pars, 2)))
    timer_df = DataFrame(θindices=[zeros(Int, 2) for _ in len],
        mean_optimisation_calls=zeros(len),
        mean_likelihood_calls=zeros(len))

    TO.enable_debug_timings(PlaceholderLikelihood)
    TO.reset_timer!(PlaceholderLikelihood.timer)

    record_CI_LL_evaluations!(timer_df, 100)

    CSV.write(joinpath(output_location, "confidence_boundary_ll_calls.csv"), timer_df)

    TO.disable_debug_timings(PlaceholderLikelihood)
end

if !isfile(joinpath(output_location, "confidence_boundary_ll_calls_simultaneous_threshold.csv"))

    function record_CI_LL_evaluations!(timer_df, N)
        Random.seed!(1234)
        training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
        
        len_combos = length(collect(combinations(1:model.core.num_pars, 2)))
        total_opt_calls=zeros(Int, len_combos)
        total_ll_calls=zeros(Int, len_combos)

        equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

        for j in 1:N 
            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, ))
            model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

            opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
            
            for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
                TO.reset_timer!(PlaceholderLikelihood.timer)
                bivariate_confidenceprofiles!(model, [pars], 30, method=RadialMLEMethod(0.15, 0.01), existing_profiles=:overwrite, 
                    use_distributed=false, use_threads=false, optimizationsettings=opt_settings, confidence_level=equiv_simul_conf_level)

                total_opt_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"])

                total_ll_calls[i] += TO.ncalls(
                    PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

                TO.reset_timer!(PlaceholderLikelihood.timer)
            end
            println("Iteration: ", j)
        end

        timer_df[:,1] .= collect(combinations(1:model.core.num_pars, 2))
        timer_df[:,2] .= total_opt_calls ./ N
        timer_df[:,3] .= total_ll_calls ./ N 
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

if !isfile(joinpath(output_location, "bivariate_boundary_coverage.csv"))

    function record_bivariate_boundary_coverage(method, method_key, num_points, hullmethods)
        Random.seed!(1234)
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20,))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb_sample, ub_sample, par_magnitudes, optimizationsettings=opt_settings)

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
        biv_coverage_df = check_bivariate_boundary_coverage(data_generator, training_gen_args, model, 100, num_points, 2000, θ_true,
            collect(combinations(1:model.core.num_pars, 2));
            confidence_level=0.95,
            method=method, distributed_over_parameters=false, hullmethod=hullmethods, 
            coverage_estimate_quantile_level=0.9, optimizationsettings=opt_settings)

        biv_coverage_df.method_key .= method_key
        biv_coverage_df.num_points .= num_points
        return biv_coverage_df
    end

    methods = [RadialMLEMethod(0.15, 0.01)] 
    num_points_iter = [10,20,30,40]
    hullmethods = [MPPHullMethod(), ConvexHullMethod()]
    # len = length(combinations(1:model.core.num_pars, 2)) * length(methods) * length(num_points_iter)

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


bivariate_combinations = vcat(collect(combinations(1:7, 2)))
if !isfile(joinpath(output_location, "bivariate_parameter_coverage.csv"))
    Random.seed!(1234)

    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
    biv_coverage_df = check_bivariate_parameter_coverage(data_generator, training_gen_args, model, 1000, 30, θ_true, 
        bivariate_combinations,
        # method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.01, use_ellipse=true),
        method=RadialMLEMethod(0.15, 0.01),
        confidence_level=0.95,
        show_progress=true, distributed_over_parameters=false, optimizationsettings=opt_settings)
    display(biv_coverage_df)
    CSV.write(joinpath(output_location, "bivariate_parameter_coverage.csv"), biv_coverage_df)
end

bivariate_combinations = vcat(collect(combinations(1:7, 2)))
if !isfile(joinpath(output_location, "bivariate_parameter_coverage_significantly_more_data.csv"))
    Random.seed!(1234)

    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb_sample, ub_sample, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))
    biv_coverage_df = check_bivariate_parameter_coverage(data_generator, training_gen_args_more_data, model, 1000, 30, θ_true,
        bivariate_combinations,
        # method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.01, use_ellipse=true),
        method=RadialMLEMethod(0.15, 0.01),
        confidence_level=0.95,
        show_progress=true, distributed_over_parameters=false, optimizationsettings=opt_settings)
    display(biv_coverage_df)
    CSV.write(joinpath(output_location, "bivariate_parameter_coverage_significantly_more_data.csv"), biv_coverage_df)
end

if !isfile(joinpath(output_location, "full_sampling_prediction_coverage.csv"))

    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb_sample, ub_sample, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = [500000, 2000000, 10000000]
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_coverage(data_generator, training_gen_args, t_pred, model, 200, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=true, use_threads=true, manual_GC_calls=true)

        new_df = filter(:θname => !=(""), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_prediction_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_prediction_coverage.arrow"), coverage_df)
        GC.gc()
    end
end

if isfile(joinpath(output_location, "full_sampling_prediction_coverage_significantly_more_data.csv"))

    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb_sample_more_data, ub_sample_more_data, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = [500000, 1000000, 2000000]
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_coverage(data_generator, training_gen_args_more_data, t_pred, model, 200, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=true, use_threads=true)

        new_df = filter(:θname => !=(""), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_prediction_coverage_significantly_more_data.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_prediction_coverage_significantly_more_data.arrow"), coverage_df)
        GC.gc()
    end
end

if !isfile(joinpath(output_location, "univariate_prediction_coverage.csv"))
    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:20:60)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, θ_true, collect(1:model.core.num_pars),
            num_points_in_interval=num_points, show_progress=true, distributed_over_parameters=false,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_prediction_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_prediction_coverage.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "univariate_prediction_coverage_simultaneous_threshold.csv"))

    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:20:60)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 1)
    # PlaceholderLikelihood.get_target_loglikelihood(model, 0.95, LogLikelihood(), model.core.num_pars) ≈ 
    # PlaceholderLikelihood.get_target_loglikelihood(model, equiv_simul_conf_level, LogLikelihood(), 1)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, θ_true, collect(1:model.core.num_pars),
            num_points_in_interval=num_points, show_progress=true, distributed_over_parameters=false, confidence_level=equiv_simul_conf_level,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_prediction_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_prediction_coverage_simultaneous_threshold.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_prediction_coverage.csv"))
    using Combinatorics
    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:80)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=RadialMLEMethod(0.15, 0.01),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage.arrow"), coverage_df)
    end
end

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold.csv"))
    using Combinatorics

    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:40)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=RadialMLEMethod(0.15, 0.01),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold.arrow"), coverage_df)
        @everywhere GC.gc()
    end
end


if !isfile(joinpath(output_location, "full_sampling_realisation_coverage.csv"))

    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb_sample, ub_sample, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = [500000, 2000000, 10000000]
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred,
            model, 200, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=true, use_threads=true, manual_GC_calls=true)

        new_df = filter(:θname => !=(:union), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_realisation_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_realisation_coverage.arrow"), coverage_df)
        @everywhere GC.gc()
    end
end

if !isfile(joinpath(output_location, "full_sampling_realisation_coverage_significantly_more_data.csv"))

    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb_sample_more_data, ub_sample_more_data, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = [500000, 1000000, 2000000]
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_dimensional_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args_more_data, testing_gen_args, t_pred,
            model, 200, num_points, θ_true, [collect(1:model.core.num_pars)],
            show_progress=true, distributed_over_parameters=true, use_threads=true, manual_GC_calls=true)

        new_df = filter(:θname => !=(:union), new_df)
        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "full_sampling_realisation_coverage_significantly_more_data.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "full_sampling_realisation_coverage_significantly_more_data.arrow"), coverage_df)
        @everywhere GC.gc()
    end
end

if !isfile(joinpath(output_location, "univariate_realisation_coverage.csv"))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

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
        @everywhere GC.gc()
    end
end

if !isfile(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.csv"))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb, ub, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:20:60)
    coverage_df = DataFrame()
    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 1)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred,
            model, 1000, θ_true, collect(1:model.core.num_pars),
            show_progress=true, num_points_in_interval=num_points, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold.arrow"), coverage_df)
        @everywhere GC.gc()
        sleep(1)
    end
end

if !isfile(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold_significantly_more_data.csv"))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θ_true, lb_sample, ub_sample, par_magnitudes, optimizationsettings=create_OptimizationSettings(solve_kwargs=(maxtime=20,)))

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:40)
    coverage_df = DataFrame()
    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 1)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_univariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args_more_data, testing_gen_args, t_pred,
            model, 200, θ_true, collect(1:model.core.num_pars),
            show_progress=true, num_points_in_interval=num_points, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold_significantly_more_data.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "univariate_realisation_coverage_simultaneous_threshold_significantly_more_data.arrow"), coverage_df)
        @everywhere GC.gc()
        sleep(1)
    end
end

if !isfile(joinpath(output_location, "bivariate_realisation_coverage.csv"))
    using Combinatorics

    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:40)
    coverage_df = DataFrame()

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=RadialMLEMethod(0.15, 0.01),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            # confidence_level=equiv_simul_conf_level,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_realisation_coverage.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_realisation_coverage.arrow"), coverage_df)
        @everywhere GC.gc()
    end
end

if !isfile(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold.csv"))
    using Combinatorics

    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:40)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 30, θ_true, collect(combinations(1:model.core.num_pars, 2)),
            method=RadialMLEMethod(0.15, 0.01),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level,
            manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold.arrow"), coverage_df)
        @everywhere GC.gc()
    end
end

#########################################################################################################################
#########################################################################################################################

if !isfile(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations.csv"))
    using Combinatorics

    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:0)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_coverage(data_generator, training_gen_args, t_pred, model, 1000, 30, θ_true, [[1, 4], [1, 5], [2, 3], [2, 6]],
            method=RadialMLEMethod(0.15, 0.01),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            confidence_level=equiv_simul_conf_level, manual_GC_calls=true,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_prediction_coverage_simultaneous_threshold_four_combinations.arrow"), coverage_df)
        @everywhere GC.gc()
    end
end

if !isfile(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_four_combinations.csv"))
    using Combinatorics

    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:0)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 30, θ_true, [[1, 4], [1, 5], [2, 3], [2, 6]],
            method=RadialMLEMethod(0.15, 0.01),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            manual_GC_calls=true,
            confidence_level=equiv_simul_conf_level,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_four_combinations.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_four_combinations.arrow"), coverage_df)
        @everywhere GC.gc()
    end
end

if !isfile(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_five_combinations.csv"))
    using Combinatorics

    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

    num_points_iter = collect(0:40:0)
    coverage_df = DataFrame()

    equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

    for num_points in num_points_iter
        Random.seed!(1234)
        new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 30, θ_true, [[1, 4], [1, 5], [2, 3], [2, 6], [2, 7]],
            method=RadialMLEMethod(0.15, 0.01),
            num_internal_points=num_points,
            show_progress=true, distributed_over_parameters=false,
            manual_GC_calls=true,
            confidence_level=equiv_simul_conf_level,
            optimizationsettings=opt_settings)

        new_df.num_points .= num_points
        global coverage_df = vcat(coverage_df, new_df)
        CSV.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_five_combinations.csv"), coverage_df)
        Arrow.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_five_combinations.arrow"), coverage_df)
        @everywhere GC.gc()
    end
end

# if !isfile(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_nosigma.csv"))
#     using Combinatorics

#     opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,))
#     model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

#     opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=20, xtol_rel=1e-12))

#     num_points_iter = collect(0:40:0)
#     coverage_df = DataFrame()

#     equiv_simul_conf_level = PlaceholderLikelihood.get_equivalent_confidence_level_chisq(0.95, model.core.num_pars, 2)

#     for num_points in num_points_iter
#         Random.seed!(1234)
#         new_df = check_bivariate_prediction_realisations_coverage(data_generator, reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 100, 30, θ_true,collect(combinations(1:model.core.num_pars-1, 2)),
#             method=RadialMLEMethod(0.15, 0.01),
#             num_internal_points=num_points,
#             show_progress=true, distributed_over_parameters=false,
#             confidence_level=equiv_simul_conf_level,
#             optimizationsettings=opt_settings,
#             manual_GC_calls=true)

#         new_df.num_points .= num_points
#         global coverage_df = vcat(coverage_df, new_df)
#         CSV.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_nosigma.csv"), coverage_df)
#         Arrow.write(joinpath(output_location, "bivariate_realisation_coverage_simultaneous_threshold_nosigma.arrow"), coverage_df)
#         # @everywhere GC.gc()
#     end
# end