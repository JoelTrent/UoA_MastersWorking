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
    e=loglikelihood(data.dist, data.y_obs[:, 1] .- y1)  
    f=loglikelihood(data.dist, data.y_obs[:, 2] .- y2)
    return e+f
end

@everywhere function predictFunc(θ, data, t=data.t)
    y1, y2 = ODEmodel(t, θ) 
    y = hcat(y1,y2)
    return y
end

@everywhere function errorFunc(predictions, θ, bcl); normal_error_σ_known(predictions, θ, bcl, σ) end

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = generator_args.y_true .+ rand(generator_args.dist, length(generator_args.t), 2)
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

# Data setup ###########################################################################
using CSV, DataFrames
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

    θ_true=[α_true, β_true, x0_true, y0_true]
    t, y_true, y_obs = data_setup(t, θ_true)
    
    # Named tuple of all data required within the log-likelihood function
    data = (y_obs=y_obs, t=t, dist=Normal(0, σ))
    training_gen_args = (y_true=y_true, t=t, dist=Normal(0, σ), is_test_set=false)
    testing_gen_args = (y_true=y_true, t=t, dist=Normal(0, σ), is_test_set=true)
    
    t_pred=LinRange(0,10,2001)

    # Bounds on model parameters 
    αmin, αmax   = (0.7, 1.2)
    βmin, βmax   = (0.7, 1.4)
    x0min, x0max = (0.5, 1.2)
    y0min, y0max = (0.1, 0.5)
    lb = [αmin,βmin,x0min,y0min]
    ub = [αmax,βmax,x0max,y0max]

    θG = θ_true
    θnames = [:α, :β, :x0, :y0]
    par_magnitudes = [1,1,1,1]

    return data, training_gen_args, testing_gen_args, θ_true, y_true, t_pred, θnames, 
        θG, lb, ub, par_magnitudes
end

data, training_gen_args, testing_gen_args, θ_true, y_true, t_pred, θnames, 
    θG, lb, ub, par_magnitudes = parameter_and_data_setup()
