using Distributed
using Revise
using CSV, DataFrames
if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO
@everywhere using Revise
@everywhere using Random, Distributions, DifferentialEquations
@everywhere using PlaceholderLikelihood

include(joinpath("Models", "logistic_twospecies.jl"))
output_location = joinpath("Experiments", "Outputs", "logistic_twospecies");

# do experiments
opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20,local_method=NLopt.LN_NELDERMEAD()))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);

if !isfile(joinpath(output_location, "univariate_parameter_coverage_new.csv"))
    uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args, model, 500, θ_true, collect(1:7), 
                        θlb_nuisance=lb_nuisance, θub_nuisance=ub_nuisance, show_progress=true, distributed_over_parameters=false,
                        optimizationsettings=opt_settings)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage_new.csv"), uni_coverage_df)
end

if !isfile(joinpath(output_location, "univariate_parameter_coverage_more_data.csv"))
    uni_coverage_df = check_univariate_parameter_coverage(data_generator, training_gen_args_more_data, model, 500, θ_true, collect(1:7), show_progress=true, distributed_over_parameters=false)
    display(uni_coverage_df)
    CSV.write(joinpath(output_location, "univariate_parameter_coverage_more_data.csv"), uni_coverage_df)
end

if true || !isfile(joinpath(output_location, "uni_profile_1.pdf"))

    opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20, local_method=NLopt.LN_NELDERMEAD()))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    n = 40
    additional_width = 0.2
    univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical(), num_points_in_interval=n, additional_width=additional_width)
    univariate_confidenceintervals!(model, profile_type=LogLikelihood(), θlb_nuisance=lb_nuisance, θub_nuisance=ub_nuisance, use_distributed=false, num_points_in_interval=n, additional_width=additional_width)

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

    opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
    model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

    # bivariate_confidenceprofiles!(model, 100, method=AnalyticalEllipseMethod(0.0, 1.0))
    # bivariate_confidenceprofiles!(model, 50, method=IterativeBoundaryMethod(10, 5, 5, 0.15, 1.0, use_ellipse=true), profile_type=LogLikelihood())

    # using Plots
    # gr()
    # format = (size=(400, 400), dpi=300, title="", legend_position=:topright)
    # plts = plot_bivariate_profiles_comparison(model; label_only_MLE=true, format...)

    # for (i, plt) in enumerate(plts)
    #     if i < length(plts)
    #         plot!(plts[i], legend_position=nothing)
    #     end
    #     savefig(plts[i], joinpath(output_location, "biv_profile_" * string(i) * ".pdf"))
    # end
end


# if !isfile(joinpath(output_location, "confidence_boundary_ll_calls.csv"))

#     using Combinatorics

#     function record_CI_LL_evaluations!(timer_df, method, method_key, num_points, iter)
#         opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
#         model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

#         len_combos = length(collect(combinations(1:model.core.num_pars, 2)))

#         for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
#             bivariate_confidenceprofiles!(model, [pars], num_points, method=method, existing_profiles=:overwrite, use_distributed=false, use_threads=false)

#             timer_df[i+len_combos*(iter-1), :] .= pars, method_key, num_points, TO.ncalls(
#                 PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
#             TO.ncalls(
#                 PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

#             TO.reset_timer!(PlaceholderLikelihood.timer)
#         end
#         return nothing
#     end

#     methods = [IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true)]
#     num_points_iter = collect(10:10:60)
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
#     CSV.write(joinpath(output_location, "methods.csv"), method_df)

#     TO.disable_debug_timings(PlaceholderLikelihood)
# end

# if !isfile(joinpath(output_location, "bivariate_boundary_coverage.csv"))

#     using Combinatorics

#     function record_bivariate_boundary_coverage(method, method_key, num_points, hullmethods)
#         Random.seed!(1234)
#         opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
#         model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

#         biv_coverage_df = check_bivariate_boundary_coverage(data_generator, training_gen_args, model, 500, num_points, 4000, θ_true,
#             collect(combinations(1:model.core.num_pars, 2)); method=method, distributed_over_parameters=false, hullmethod=hullmethods, 
#             coverage_estimate_quantile_level=0.9)

#         biv_coverage_df.method_key .= method_key
#         biv_coverage_df.num_points .= num_points
#         return biv_coverage_df
#     end

#     methods = [IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true)]
#     num_points_iter = [10, 20, 30, 40, 50, 60]
#     hullmethods = [MPPHullMethod(), ConvexHullMethod()]
#     len = length(collect(combinations(1:model.core.num_pars, 2))) * length(methods) * length(num_points_iter)

#     method_df = DataFrame(method_key=collect(1:length(methods)),
#         method_name=string.(methods))
#     CSV.write(joinpath(output_location, "methods.csv"), method_df)

#     coverage_df = DataFrame()

#     for (method_key, method) in enumerate(methods)
#         for num_points in num_points_iter
#             global coverage_df = vcat(coverage_df, record_bivariate_boundary_coverage(method, method_key, num_points, hullmethods))
#             CSV.write(joinpath(output_location, "bivariate_boundary_coverage.csv"), coverage_df)
#         end
#     end
# end