##############################################################################
############################ LOTKA VOLTERRA MODEL ############################
########### ORIGINAL FROM https://github.com/ProfMJSimpson/Workflow ##########
##############################################################################

# Model functions
###########################################################################
@everywhere function lotka_static(C,p,t)
    dC_1=p[1]*C[1] - C[1]*C[2];
    dC_2=p[2]*C[1]*C[2] - C[2];
    SA[dC_1, dC_2]
end

@everywhere function odesolver(t,α,β,C01,C02)
    p=SA[α,β]
    C0=SA[C01,C02]
    tspan=(0.0,t[end])
    prob=ODEProblem(lotka_static,C0,tspan,p)
    sol=solve(prob, AutoTsit5(Rosenbrock23()), saveat=t);
    return sol[1,:], sol[2,:]
end

@everywhere function ODEmodel(t,θ)
    return odesolver(t,θ[1],θ[2],θ[3],θ[4])
end

@everywhere function loglhood(θ, data)
    (y1, y2) = ODEmodel(data.t, θ)
    dist = Normal(0.0, θ[5])
    e=loglikelihood(dist, data.y_obs[:,1] .- y1)  
    f=loglikelihood(dist, data.y_obs[:,2] .- y2)
    return e+f
end

@everywhere function predictFunc(θ, data, t=data.t)
    y1, y2 = ODEmodel(t, θ) 
    y = hcat(y1,y2)
    return y
end

@everywhere function errorFunc(predictions, θ, cl); normal_error_σ_estimated(predictions, θ, cl, 4) end

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = generator_args.y_true .+ rand(generator_args.dist, length(generator_args.t), 2)
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

@everywhere function reference_set_generator(θtrue, generator_args::NamedTuple, region::Float64)
    lq, uq = errorFunc(generator_args.y_true, θtrue, region)
    return (lq, uq)
end

# Data setup ###########################################################################
using Random
data_location = joinpath("Experiments", "Models", "Data", "lotka_volterra.csv")
function data_setup(t, θ_true, generate_new_data=false)
    if !generate_new_data && isfile(data_location)
        t, y_true1, y_true2, y_obs1, y_obs2 = eachcol(CSV.read(data_location, DataFrame))
        y_true = hcat(y_true1, y_true2)
        y_obs = hcat(y_obs1, y_obs2)
    else
        # true data
        y_true = hcat(ODEmodel(t, θ_true)...)

        Random.seed!(12348)
        # noisy data
        y_obs = y_true .+ σ*randn(length(t),2)

        data_df = DataFrame(t=t, y_true1=y_true[:,1], y_true2=y_true[:,2],
                            y_obs1=y_obs[:,1], y_obs2=y_obs[:,2])
        display(data_df)
        CSV.write(data_location, data_df)
    end
    return t, y_true, y_obs
end

function parameter_and_data_setup()
    # true parameters
    α_true=0.9; β_true=1.1; x0_true=0.8; y0_true=0.3
    t=LinRange(0,10,21)
    @everywhere global σ=0.2

    θ_true=[α_true, β_true, x0_true, y0_true, σ]
    t, y_true, y_obs = data_setup(t, θ_true)
    
    # Named tuple of all data required within the log-likelihood function
    data = (y_obs=y_obs[1:15,:], t=t[1:15], dist=Normal(0, σ))
    training_gen_args = (y_true=y_true[1:15,:], t=t[1:15], dist=Normal(0, σ), is_test_set=false)

    t_more = LinRange(0, 7, 111)
    y_true_more = predictFunc(θ_true, data, t_more)
    training_gen_args_more_data = (y_true=y_true_more, t=t_more, dist=Normal(0, σ), is_test_set=false)
    
    t_pred=LinRange(0,10,201)
    testing_gen_args = (y_true=hcat(ODEmodel(t_pred, θ_true)...), dist=Normal(0, σ), t=t_pred, is_test_set=true)

    # Bounds on model parameters 
    # αmin, αmax   = (0.7, 1.2)
    # βmin, βmax   = (0.7, 1.4)
    # x0min, x0max = (0.5, 1.2)
    # y0min, y0max = (0.1, 0.5)
    αmin, αmax = (0.4, 1.3)
    βmin, βmax = (0.7, 1.6)
    x0min, x0max = (0.4, 1.3)
    y0min, y0max = (0.02, 0.6)
    σmin, σmax = (0.01, 0.5)
    lb = [αmin,βmin,x0min,y0min,σmin]
    ub = [αmax,βmax,x0max,y0max,σmax]

    lb_more_data = [0.82, 0.95, 0.68, 0.23, 0.14]
    ub_more_data = [1.0, 1.2, 0.93, 0.42, 0.27]

    θG = θ_true
    θnames = [:α, :β, :x0, :y0, :σ]
    par_magnitudes = [1,1,1,1,1]

    return data, training_gen_args, training_gen_args_more_data, testing_gen_args, θ_true, t_pred, θnames, 
        θG, lb, ub, lb_more_data, ub_more_data, par_magnitudes
end

data, training_gen_args, training_gen_args_more_data, testing_gen_args, θ_true, t_pred, θnames,
    θG, lb, ub, lb_more_data, ub_more_data, par_magnitudes = parameter_and_data_setup()
