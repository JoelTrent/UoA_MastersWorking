
using LikelihoodProfiler

using Distributed
using Revise
using CSV, DataFrames, Arrow
using LikelihoodBasedProfileWiseAnalysis
@everywhere using Revise
@everywhere using Random, Distributions, DifferentialEquations

# @everywhere using Logging
# @everywhere Logging.disable_logging(Logging.Warn) # Disable debug, info and warn

include(joinpath("Models", "logistic_twospecies_logitnormal.jl"))
output_location = joinpath("Experiments", "Outputs", "logistic_twospecies_logitnormal")

# search CI with LikelihoodProfiler

opt_settings = create_OptimizationSettings(solve_alg=NLopt.LN_BOBYQA(), solve_kwargs=(maxtime=20, ))
model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);
p_best = model.core.θmle
num_params = length(p_best)

intervals = Vector{ParamInterval}(undef, num_params)
function loss_func(θ); return -loglhood(θ, data) end

α = loss_func(p_best) + 1.92073 # chisq with 1 df

tbounds = [(lb[i]+0.00001, ub[i]) for i in 1:num_params]
sbounds = [(lb[i]+0.00002, ub[i]-0.0001) for i in 1:num_params]
scan_tol=[1e-5, 1e-5, 1e-5, 1e-3, 1e-4, 1e-4, 1e-4]
for i in 1:num_params
    @time intervals[i] = get_interval(
        p_best,
        i,
        loss_func,
        :CICO_ONE_PASS,
        loss_crit=α,
        scale=[:log, :log, :log, :direct, :direct, :direct, :direct],
        theta_bounds=tbounds,
        scan_bounds=sbounds[i],
        scan_tol=scan_tol[i],
        local_alg=:LN_NELDERMEAD,
    )
end;

ENV["COLUMNS"] = 120
res = DataFrame(
    params=θnames,
    LSatus=[k.result[1].status for k in intervals],
    UStatus=[k.result[2].status for k in intervals],
    LBound=[isnothing(k.result[1].value) ? NaN : k.result[1].value for k in intervals],
    UBound=[isnothing(k.result[2].value) ? NaN : k.result[2].value for k in intervals],
    LCount=[k.result[1].counter for k in intervals],
    UCount=[k.result[2].counter for k in intervals],
    TotCount=[k.result[1].counter + k.result[2].counter for k in intervals],
    InitValues=p_best
)



function record_CI_LL_evaluations!(N)
    Random.seed!(1234)
    training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]

    # display(first.(training_data))

    total_ll_calls = zeros(Int, 7)

    for j in 1:N

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, abstol=0.0))
        model = initialise_LikelihoodModel(loglhood, predictFunc, errorFunc, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        function loss_func_training(θ)
            return -loglhood(θ, training_data[j])
        end

        α_training = loss_func_training(model.core.θmle) + 1.92073 # chisq with 1 df

        intervals_training = Vector{ParamInterval}(undef, num_params)
        for i in 1:num_params
            intervals_training[i] = get_interval(
                model.core.θmle,
                i,
                loss_func_training,
                :CICO_ONE_PASS,
                loss_crit=α_training,
                scale=[:log, :log, :log, :direct, :direct, :direct, :direct],
                theta_bounds=tbounds,
                scan_bounds=sbounds[i],
                scan_tol=scan_tol[i],
                local_alg=:LN_NELDERMEAD,
            )
        end

        # println([k.result[1].value for k in intervals_training])
        # println([k.result[2].value for k in intervals_training])
        # println([k.result[1].counter for k in intervals_training])
        # println([k.result[2].counter for k in intervals_training])
        total_ll_calls .+= [k.result[1].counter + k.result[2].counter for k in intervals_training]
    end

    return total_ll_calls ./ N
end

mean_count = record_CI_LL_evaluations!(100)
res.MeanCount = mean_count

CSV.write(joinpath(output_location, "likelihoodprofiler_conf_int_calls.csv"), res)