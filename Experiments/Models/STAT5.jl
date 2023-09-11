##############################################################################
################################ STAT5 MODEL #################################
############### ORIGINAL FROM https://doi.org/10.1021/pr5006923 ##############
# MODEL IMPLEMENTATION FROM https://github.com/insysbio/likelihoodprofiler-cases/blob/master/notebook/STAT5%20Dimerization.ipynb #
##############################################################################

# Model functions
###########################################################################
@everywhere function stat5_ode(du, u, p, time)
    # 8 states:
    (STAT5A, pApA, STAT5B, pApB, pBpB, nucpApA, nucpApB, nucpBpB) = u
    # 6 parameters
    (Epo_degradation_BaF3, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos) = p

    BaF3_Epo = 1.25e-7 * exp(-1 * Epo_degradation_BaF3 * time)

    v1 = BaF3_Epo * (STAT5A^2) * k_phos
    v2 = BaF3_Epo * STAT5A * STAT5B * k_phos
    v3 = BaF3_Epo * (STAT5B^2) * k_phos
    v4 = k_imp_homo * pApA
    v5 = k_imp_hetero * pApB
    v6 = k_imp_homo * pBpB
    v7 = k_exp_homo * nucpApA
    v8 = k_exp_hetero * nucpApB
    v9 = k_exp_homo * nucpBpB

    du[1] = -2 * v1 - v2 + 2 * v7 * (nuc / cyt) + v8 * (nuc / cyt)
    du[2] = v1 - v4
    du[3] = -v2 - 2 * v3 + v8 * (nuc / cyt) + 2 * v9 * (nuc / cyt)
    du[4] = v2 - v5
    du[5] = v3 - v6
    du[6] = v4 * (cyt / nuc) - v7
    du[7] = v5 * (cyt / nuc) - v8
    du[8] = v6 * (cyt / nuc) - v9
end

@everywhere function odesolver(t, θ)
    u0 = zeros(8)
    u0[1] = 207.6 * ratio         # STAT5A
    u0[3] = 207.6 - 207.6 * ratio # STAT5B
    tspan=(0.0, t[end])
    prob = ODEProblem(stat5_ode, u0, tspan, θ)

    # solution
    sol = solve(prob, lsoda(), saveat=t, reltol=1e-7, abstol=1e-7) #save_idxs=[1,2,3,4,5] 
    STAT5A = sol[1, :]
    pApA = sol[2, :]
    STAT5B = sol[3, :]
    pApB = sol[4, :]
    pBpB = sol[5, :]

    # observables
    pSTAT5A_rel = (100 * pApB + 200 * pApA * specC17) ./ (pApB + STAT5A * specC17 + 2 * pApA * specC17)
    pSTAT5B_rel = -(100 * pApB - 200 * pBpB * (specC17 - 1)) ./ ((STAT5B * (specC17 - 1) - pApB) + 2 * pBpB * (specC17 - 1))
    rSTAT5A_rel = (100 * pApB + 100 * STAT5A * specC17 + 200 * pApA * specC17) ./ (2 * pApB + STAT5A * specC17 + 2 * pApA * specC17 - STAT5B * (specC17 - 1) - 2 * pBpB * (specC17 - 1))

    return pSTAT5A_rel, pSTAT5B_rel, rSTAT5A_rel
end


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
     
@everywhere function loglhood(θ, data)
    (y1, y2, y3) = odesolver(data.t, θ)
    dist1 = Normal(0.0, θ[7])
    dist2 = Normal(0.0, θ[8])
    dist3 = Normal(0.0, θ[9])
    e = loglikelihood(dist1, data.y_obs[:,1] - y1) + 
        loglikelihood(dist2, data.y_obs[:,2] - y2) + 
        loglikelihood(dist3, data.y_obs[:,3] - y3)
    return e
end

@everywhere function predictFunc(θ, data, t=data.t)
    (y1, y2, y3) = odesolver(t, θ)
    y = hcat(y1,y2,y3)
    return y
end

@everywhere function errorFunc(predictions, θ, bcl)
    q1 = normal_error_σ_estimated(predictions[:,1], θ, bcl, 7)
    q2 = normal_error_σ_estimated(predictions[:,2], θ, bcl, 8)
    q3 = normal_error_σ_estimated(predictions[:,3], θ, bcl, 9)
    return hcat(q1[1], q2[1], q3[1]), hcat(q1[2], q2[2], q3[2])
end

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = zeros(size(generator_args.y_true))
    for (i, dist) in enumerate((generator_args.dist1, generator_args.dist2, generator_args.dist3))
        y_obs[:,i] = generator_args.y_true[:,i] .+ rand(dist, length(generator_args.t))
    end
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

# Data setup ###########################################################################
data_location = joinpath("Experiments", "Models", "Data", "STAT5.csv")
function data_setup()
    return eachcol(CSV.read(data_location, DataFrame))
end

function parameter_and_data_setup()
    t, y1, y2, y3 = data_setup()

    # Named tuple of all data required within the log-likelihood function
    data = (y_obs=hcat(y1, y2, y3), t=t)

    @everywhere const global cyt = 1.4
    @everywhere const global nuc = 0.45
    @everywhere const global ratio = 0.693
    @everywhere const global specC17 = 0.107

    u0 = zeros(8)
    u0[1] = 207.6 * ratio         # STAT5A
    u0[3] = 207.6 - 207.6 * ratio # STAT5B

    # for the purposes of coverage testing (not actually known)
    # MLE values to 3 s.f.
    θ_true = [
        0.027,      # Epo_degradation_BaF3
        0.00001,    # k_exp_hetero
        0.00617,    # k_exp_homo
        0.0164,     # k_imp_hetero
        97700.,     # k_imp_homo
        15800.,     # k_phos
        3.85,       # sd_pSTAT5A_rel
        6.59,       # sd_pSTAT5B_rel
        3.15        # sd_rSTAT5A_rel
        ]
    y_true = predictFunc(θ_true, data)
    training_gen_args = (y_true=y_true, t=t, dist1=Normal(0, θ_true[7]), dist2=Normal(0, θ_true[8]), 
        dist3=Normal(0, θ_true[9]), is_test_set=false)
    testing_gen_args = (y_true=y_true, t=t, dist1=Normal(0, θ_true[7]), dist2=Normal(0, θ_true[8]),
        dist3=Normal(0, θ_true[9]), is_test_set=true)

    t_pred=LinRange(t[1], t[end], 100)

    # Bounds on model parameters 
    lb = [0., 0.000001, 0., 0., 0.01, 5000, 0.5, 0.5, 0.5]
    ub = [0.1, 0.001, 0.02, 0.1, 120000, 40000, 20., 20., 20.]
    
    θG = [0.01, 0.00001, 0.006, 0.02, 100000, 20000, 4, 4, 4] .* 1.
    
    θnames = [:Epo_degradation_BaF3, :k_exp_hetero, :k_exp_homo, :k_imp_hetero, :k_imp_homo, :k_phos,
        :σ_pSTAT5A_rel, :σ_pSTAT5B_rel, :σ_rSTAT5A_rel]
    par_magnitudes = [0.1, 0.001, 0.01, 1000., 10000., 1., 1., 1.]

    return data, training_gen_args, testing_gen_args, θ_true, y_true, t_pred, θnames, 
        θG, lb, ub, par_magnitudes
end

data, training_gen_args, testing_gen_args, θ_true, y_true, t_pred, θnames, 
    θG, lb, ub, par_magnitudes = parameter_and_data_setup()
