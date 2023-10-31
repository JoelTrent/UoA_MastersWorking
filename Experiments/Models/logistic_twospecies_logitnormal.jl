##############################################################################
######################### TWO SPECIES LOGISTIC MODEL #########################
##### ORIGINAL FROM https://github.com/ProfMJSimpson/profile_predictions #####
##############################################################################

# Model functions
###########################################################################
@everywhere function DE!(dC, C, p, t)
    λ1, λ2, δ, KK = p
    S = C[1] + C[2]
    dC[1] = λ1 * C[1] * (1.0 - S/KK)
    dC[2] = λ2 * C[2] * (1.0 - S/KK) - δ*C[2]*C[1]/KK
end

@everywhere function odesolver(t, λ1, λ2, δ, KK, C01, C02)
    p=(λ1, λ2, δ, KK)
    C0=[C01, C02]
    tspan=(0.0, maximum(t))
    prob=ODEProblem(DE!, C0, tspan, p)
    sol=solve(prob, saveat=t)
    return sol[1,:], sol[2,:]
end

@everywhere function ODEmodel(t, θ)
    (y1, y2) = odesolver(t, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6])
    return y1, y2
end

@everywhere function logit(p)
    return log(p/(one(p)-p))
end

@everywhere function loglhood(θ, data)
    (y1, y2) = ODEmodel(data.t, θ)
    e=0.0
    for i in axes(data.y_obs,1)
        e += (loglikelihood(LogitNormal(logit(y1[i]/100.), θ[7]), data.y_obs[i,1]/100.) + 
                loglikelihood(LogitNormal(logit(y2[i]/100.), θ[7]), data.y_obs[i,2]/100.))
    end
    return e
end

@everywhere function predictFunc(θ, data, t=data.t)
    y1, y2 = ODEmodel(t, θ) 
    y = hcat(y1,y2)
    return y
end

@everywhere function errorFunc(predictions, θ, cl)
    THalpha = 1.0 - cl
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    for i in eachindex(predictions)
        dist = LogitNormal(logit(predictions[i] / 100.), θ[7])
        lq[i] = quantile(dist, THalpha / 2.0) / 1000.
        uq[i] = quantile(dist, 1 - (THalpha / 2.0)) / 1000.
    end
    return lq, uq
end

# DATA GENERATION FUNCTION AND ARGUMENTS
# @everywhere function data_generator(θtrue, generator_args::NamedTuple)
#     y_obs = generator_args.y_true .+ rand(generator_args.dist, length(generator_args.t), 2)
#     if generator_args.is_test_set; return y_obs end
#     data = (y_obs=y_obs, generator_args...)
#     return data
# end

@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = zeros(size(generator_args.y_true))
    for i in eachindex(generator_args.y_true)
        y_obs[i] = rand(LogitNormal(logit(generator_args.y_true[i]/100.), θtrue[7]))
    end
    y_obs .= y_obs .* 100
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

# Data setup ###########################################################################
data_location = joinpath("Experiments", "Models", "Data", "logistic_twospecies.csv")
function data_setup()
    return eachcol(CSV.read(data_location, DataFrame))
end

function parameter_and_data_setup()
    t, data11, data12 = data_setup()

    # Named tuple of all data required within the log-likelihood function
    data = (y_obs=hcat(data11,data12), t=t)

    # for the purposes of coverage testing (not actually known)
    # MLE values to 3 s.f.
    # θ_true = [0.00298, 0.000629, 0.00055, 78.7, 0.265, 1.0, 1.0]
    θ_true = [0.003, 0.0004, 0.0004, 80.0, 0.4, 1.2, 0.1]
    # θmle = [0.00263, 0.00036, 0.000398, 81.3, 0.384, 1.19, 0.451]
    y_true = predictFunc(θ_true, data)
    training_gen_args = (y_true=y_true, t=t, is_test_set=false)

    t_more = LinRange(t[1], t[end], length(t)*2)
    n=2
    t_more = zeros(length(t)*(n+1) - n)
    t_more[1:(n+1):end] .= t
    for i in 1:(n+1):length(t_more)-n, j=i+(n+1)
        t_more[(i+1):(j-1)] .= LinRange(t_more[i], t_more[j], n+2)[2:end-1]
    end

    y_true_more = predictFunc(θ_true, data, t_more)
    training_gen_args_more_data = (y_true=y_true_more, t=t_more, is_test_set=false)
    testing_gen_args = (y_true=y_true, t=t, is_test_set=true)

    t_pred=LinRange(t[1], t[end], 400)

    # Bounds on model parameters 
    lb = [0.0005, 0.00001, 0.00001, 60.0, 0.01, 0.1, 0.01]
    ub = [0.01, 0.005, 0.005, 98.0, 2.0, 3.0, 1.0]
    lb_nuisance = max.(lb, θ_true ./ 2.5)
    ub_nuisance = min.(ub, θ_true .* 2.5)
    
    λ1g=0.002; λ2g=0.002; δg=0.001; KKg=80.0; C0g=[1.0, 1.0]; σg=0.5
    θG = [λ1g, λ2g, δg, KKg, C0g[1], C0g[2], σg]
    
    θnames = [:λ1, :λ2, :δ, :K, :C01, :C02, :σ]
    par_magnitudes = [0.001, 0.001, 0.001, 10, 1, 1, 1]

    return data, training_gen_args, training_gen_args_more_data, testing_gen_args, θ_true, y_true, t_pred, θnames,
        θG, lb, ub, lb_nuisance, ub_nuisance, par_magnitudes
end

data, training_gen_args, training_gen_args_more_data, testing_gen_args, θ_true, y_true, t_pred, θnames,
    θG, lb, ub, lb_nuisance, ub_nuisance, par_magnitudes = parameter_and_data_setup()
