
using LikelihoodProfiler

using Distributed
using Revise
using CSV, DataFrames, Arrow
@everywhere using Revise
@everywhere using Random, Distributions

@everywhere using Logging
@everywhere Logging.disable_logging(Logging.Warn) # Disable debug, info and warn

include(joinpath("Models", "logistic.jl"))
output_location = joinpath("Experiments", "Outputs", "logistic")

# search CI with LikelihoodProfiler
p_best = [0.01053235563279569, 100.06748817567728, 8.820501748728539]
num_params = length(p_best)

intervals = Vector{ParamInterval}(undef, num_params)
function loss_func(θ); return -loglhood(θ, data) end

α = loss_func(p_best) + 1.92 # chisq with 1 df

tbounds = [(lb[i], ub[i]) for i in 1:num_params]
sbounds = [(lb[i]+0.0001, ub[i]-0.0001) for i in 1:num_params]
scan_tol=[1e-6, 1e-3, 1e-3]
for i in 1:num_params
    @time intervals[i] = get_interval(
        p_best,
        i,
        loss_func,
        :CICO_ONE_PASS,
        loss_crit=α,
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
    LBound=[k.result[1].value for k in intervals],
    UBound=[k.result[2].value for k in intervals],
    LCount=[k.result[1].counter for k in intervals],
    UCount=[k.result[2].counter for k in intervals],
    TotCount=[k.result[1].counter + k.result[2].counter for k in intervals],
    InitValues=p_best
)

CSV.write(joinpath(output_location, "likelihood_profile_conf_int_calls.csv"), res)