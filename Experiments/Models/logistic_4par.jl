##############################################################################
######################### ONE SPECIES LOGISTIC MODEL #########################
########### ORIGINAL FROM https://github.com/ProfMJSimpson/Workflow ##########
##############################################################################

# Model functions
###########################################################################
@everywhere function solvedmodel(t, θ)
    return (θ[2]*θ[3]) ./ ((θ[2]-θ[3]) .* (exp.(-θ[1] .* t)) .+ θ[3])
end

@everywhere function loglhood(θ, data)
    y=solvedmodel(data.t, θ)
    dist = Normal(0.0, θ[4])
    e=0
    e=sum(loglikelihood(dist, data.y_obs .- y))
    return e
end

@everywhere function predictFunc(θ, data, t=data.t); solvedmodel(t, θ) end

@everywhere function errorFunc(predictions, θ, cl); normal_error_σ_estimated(predictions, θ, cl, 4) end

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = generator_args.y_true .+ rand(generator_args.dist, length(generator_args.t))
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

@everywhere function reference_set_generator(θtrue, generator_args::NamedTuple, confidence_level::Float64)
    lq, uq = errorFunc(generator_args.y_true, θtrue, confidence_level)
    return (lq, uq)
end

# Data setup ###########################################################################
using Random
data_location = joinpath("Experiments", "Models", "Data", "logistic.csv")
function data_setup(t, θ_true, generate_new_data=false)
    if !generate_new_data && isfile(data_location)
        t, y_true, y_obs = eachcol(CSV.read(data_location, DataFrame))
    else
        # true data
        y_true = solvedmodel(t, θ_true)

        Random.seed!(12348)
        # noisy data
        y_obs = y_true + σ*randn(length(t))

        data_df = DataFrame(t=t, y_true=y_true, y_obs=y_obs)
        display(data_df)
        CSV.write(data_location, data_df)
    end
    return t, y_true, y_obs
end

function parameter_and_data_setup()
    # true parameters
    λ_true=0.01; K_true=100.0; C0_true=10.0; t=0:100:1000; 
    @everywhere global σ=10.0;
    θ_true=[λ_true, K_true, C0_true, σ]
    t, y_true, y_obs = data_setup(t, θ_true)

    # Named tuple of all data required within the log-likelihood function
    data = (y_obs=y_obs, t=t, dist=Normal(0, σ))
    training_gen_args = (y_true=y_true, t=t, dist=Normal(0, σ), is_test_set=false)
    
    t_more = LinRange(0, 1000, 111)
    y_true_more = predictFunc(θ_true, data, t_more)
    training_gen_args_more_data = (y_true=y_true_more, t=t_more, dist=Normal(0, σ),is_test_set = false)

    t_pred=0:5:1000
    testing_gen_args = (y_true=solvedmodel(t_pred, θ_true), t=t_pred, dist=Normal(0, σ), is_test_set=true)

    # Bounds on model parameters 
    λ_min, λ_max = (0.00, 0.1)
    K_min, K_max = (50., 150.)
    C0_min, C0_max = (0.0, 50.)
    σ_min, σ_max = (1.0, 30.0)
    lb = [λ_min, K_min, C0_min, σ_min]
    ub = [λ_max, K_max, C0_max, σ_max]

    lb_more_data = [0.005, 90, 4, 6]
    ub_more_data = [0.015, 110, 20, 13]
    
    θnames = [:λ, :K, :C0, :σ]
    θG = θ_true
    par_magnitudes = [0.005, 10, 10, 1]

    return data, training_gen_args, training_gen_args_more_data, testing_gen_args, θ_true, y_true, t_pred, θnames, 
        θG, lb, ub, lb_more_data, ub_more_data, par_magnitudes
end

data, training_gen_args, training_gen_args_more_data, testing_gen_args, θ_true, y_true, t_pred, θnames,
    θG, lb, ub, lb_more_data, ub_more_data, par_magnitudes = parameter_and_data_setup()