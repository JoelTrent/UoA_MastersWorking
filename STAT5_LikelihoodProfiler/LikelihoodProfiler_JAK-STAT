using DiffEqBase, CSV, DataFrames, LikelihoodProfiler, LSODA
using Plots
using Distributions

# constants
const cyt = 1.4
const nuc = 0.45
const ratio = 0.693
const specC17 = 0.107


# ode system
function stat5_ode(du, u, p, time)
    # 8 states:
    (STAT5A, pApA, STAT5B, pApB, pBpB, nucpApA, nucpApB, nucpBpB) = u
    # 6 parameters
    (Epo_degradation_BaF3, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos) =  p
    
    BaF3_Epo = 1.25e-7*exp(-1*Epo_degradation_BaF3*time)

    v1 = BaF3_Epo*(STAT5A^2)*k_phos
    v2 = BaF3_Epo*STAT5A*STAT5B*k_phos
    v3 = BaF3_Epo*(STAT5B^2)*k_phos
    v4 = k_imp_homo*pApA
    v5 = k_imp_hetero*pApB
    v6 = k_imp_homo*pBpB
    v7 = k_exp_homo*nucpApA
    v8 = k_exp_hetero*nucpApB
    v9 = k_exp_homo*nucpBpB

    du[1] = -2*v1 - v2 + 2*v7*(nuc/cyt) + v8*(nuc/cyt)
    du[2] = v1 - v4
    du[3] = -v2 -2*v3 + v8*(nuc/cyt) + 2*v9*(nuc/cyt)
    du[4] = v2 - v5
    du[5] = v3 - v6
    du[6] = v4*(cyt/nuc) - v7
    du[7] = v5*(cyt/nuc) - v8
    du[8] = v6*(cyt/nuc) - v9
end;


data = CSV.read(joinpath("STAT5_LikelihoodProfiler", "data_stat5.csv"), DataFrame);

saveat = Float64.(data[!, :time])
tspan = (0.0, saveat[end])

u0 = zeros(8)
u0[1] = 207.6 * ratio         # STAT5A
u0[3] = 207.6 - 207.6 * ratio # STAT5B

prob(p) = ODEProblem(stat5_ode, eltype(p).(u0), tspan, p)

function solve_prob(p)
    _prob = prob(p)

    # solution
    sol = solve(_prob, lsoda(), saveat=saveat, reltol=1e-7, abstol=1e-7) #save_idxs=[1,2,3,4,5] 
    STAT5A = sol[1, :]
    pApA = sol[2, :]
    STAT5B = sol[3, :]
    pApB = sol[4, :]
    pBpB = sol[5, :]

    # observables
    pSTAT5A_rel = (100 * pApB + 200 * pApA * specC17) ./ (pApB + STAT5A * specC17 + 2 * pApA * specC17)
    pSTAT5B_rel = -(100 * pApB - 200 * pBpB * (specC17 - 1)) ./ ((STAT5B * (specC17 - 1) - pApB) + 2 * pBpB * (specC17 - 1))
    rSTAT5A_rel = (100 * pApB + 100 * STAT5A * specC17 + 200 * pApA * specC17) ./ (2 * pApB + STAT5A * specC17 + 2 * pApA * specC17 - STAT5B * (specC17 - 1) - 2 * pBpB * (specC17 - 1))

    return [pSTAT5A_rel, pSTAT5B_rel, rSTAT5A_rel]
end;


p_best = [
    0.026982514033029,      # Epo_degradation_BaF3
    0.0000100067973851508,  # k_exp_hetero
    0.006170228086381,      # k_exp_homo
    0.0163679184468,        # k_imp_hetero
    97749.3794024716,       # k_imp_homo
    15766.5070195731,       # k_phos
    3.85261197844677,       # sd_pSTAT5A_rel
    6.59147818673419,       # sd_pSTAT5B_rel
    3.15271275648527        # sd_rSTAT5A_rel
];

sol = solve_prob(p_best)
plot(saveat, sol, label=string.([:pSTAT5A_rel :pSTAT5B_rel :rSTAT5A_rel]), xlabel=:time)



function loss_func(p_init)
    p = exp10.(p_init)

    sim = solve_prob(p)
    σ = p[7:9]
    # loss
    return loss(sim, data, σ)
end

function loss(sim, data, σ)
    loss = 0.0
    obs = names(data)[2:end]

    for i in 1:length(obs)
        loss_i = loss_component(sim[i], data[!, i+1], σ[i])
        loss += loss_i
    end
    return loss
end

function loss_component(sim, data, σ)
    loss_i = 0.0

    for i in eachindex(sim)
        loss_i += ((sim[i] - data[i]) / σ)^2 + 2 * log(sqrt(2π) * σ)
    end
    return loss_i
end;

α = loss_func(log10.(p_best)) + 3.84 # chisq with 1 df

# search CI with LikelihoodProfiler
num_params = length(p_best)

intervals = Vector{ParamInterval}(undef, num_params)
p_log = log10.(p_best)

tbounds = fill((-7.0, 7.0), num_params)
sbounds = (-5.0, 5.0)
for i in 1:num_params
    @time intervals[i] = get_interval(
        p_log,
        i,
        loss_func,
        :CICO_ONE_PASS,
        loss_crit=α,
        theta_bounds=tbounds,
        scan_bounds=sbounds,
        scan_tol=1e-2,
        local_alg=:LN_NELDERMEAD,
    )
end;

ENV["COLUMNS"] = 120
res = DataFrame(
    params=[:Epo_degradation_BaF3, :k_exp_hetero, :k_exp_homo, :k_imp_hetero, :k_imp_homo, :k_phos, :sd_pSTAT5A_rel, :sd_pSTAT5B_rel, :sd_rSTAT5A_rel],
    LSatus=[k.result[1].status for k in intervals],
    UStatus=[k.result[2].status for k in intervals],
    LBound=[k.result[1].value for k in intervals],
    UBound=[k.result[2].value for k in intervals],
    LCount=[k.result[1].counter for k in intervals],
    UCount=[k.result[2].counter for k in intervals],
    InitValues=p_log
)

for i in 1:num_params
    update_profile_points!(intervals[i])
end
for i in 1:num_params
    plt=plot(intervals[i])
    display(plt)
end

# normal loglikelihood
function loss_func_normal(p_init)
    p = exp10.(p_init)

    sim = solve_prob(p)
    σ = p[7:9]
    # loss
    return loss_normal(sim, data, σ)
end

function loss_normal(sim, data, σ)
    loss = 0.0
    obs = names(data)[2:end]

    for i in 1:length(obs)
        loss_i = loss_component_normal(sim[i], data[!, i+1], σ[i])
        loss += loss_i
    end
    return loss
end

function loss_component_normal(sim, data, σ)
    loss_i = 0.0

    for i in eachindex(sim)
        dist = Normal(data[i], σ)
        loss_i += -2*loglikelihood(dist, sim[i]) # equivalent loss function
        # loss_i += ((sim[i] - data[i]) / σ)^2 + 2 * log(sqrt(2π) * σ)
    end
    return loss_i
end;

α = loss_func_normal(log10.(p_best)) + 3.84 # chisq with 1 df

# search CI with LikelihoodProfiler
num_params = length(p_best)

intervals = Vector{ParamInterval}(undef, num_params)
p_log = log10.(p_best)

tbounds = fill((-7.0, 7.0), num_params)
sbounds = (-5.0, 5.0)
for i in 1:num_params
    @time intervals[i] = get_interval(
        p_log,
        i,
        loss_func,
        :CICO_ONE_PASS,
        loss_crit=α,
        theta_bounds=tbounds,
        scan_bounds=sbounds,
        scan_tol=1e-2,
        local_alg=:LN_NELDERMEAD,
    )
end;

res = DataFrame(
    params=[:Epo_degradation_BaF3, :k_exp_hetero, :k_exp_homo, :k_imp_hetero, :k_imp_homo, :k_phos, :sd_pSTAT5A_rel, :sd_pSTAT5B_rel, :sd_rSTAT5A_rel],
    LSatus=[k.result[1].status for k in intervals],
    UStatus=[k.result[2].status for k in intervals],
    LBound=[k.result[1].value for k in intervals],
    UBound=[k.result[2].value for k in intervals],
    LCount=[k.result[1].counter for k in intervals],
    UCount=[k.result[2].counter for k in intervals],
    InitValues=p_log
)