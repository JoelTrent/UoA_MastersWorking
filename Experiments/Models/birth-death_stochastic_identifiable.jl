##############################################################################
######################## BIRTH-DEATH STOCHASTIC MODEL ########################
##############################################################################

using BSON

# Model functions
###########################################################################
function birth_death_firstreact(t, β, δ=1.0, N0=1, t_init=0.0)

    t_current = t_init * 1
    t_max = t[end]
    N = zeros(length(t))
    δ_total = zeros(length(t))

    N_prev = -1
    N_current = N0 * 1
    δ_current = 1

    i = 1
    max_i = length(t)
    while t_current < t_max && N_current != 0
        h_i = [β * N_current, δ * N_current]

        # delta t's for each process
        delta_t_birth = rand(Exponential(1 / h_i[1]))
        delta_t_death = rand(Exponential(1 / h_i[2]))

        if delta_t_birth <= delta_t_death
            N_prev = N_current*1
            N_current += 1
            delta_t = delta_t_birth
        else
            N_prev = N_current*1
            N_current -= 1
            δ_current += 1
            delta_t = delta_t_death
        end

        t_current += delta_t

        while i <= max_i && t_current > t[i]
            if N_prev == -1
                N[i] = N_current*1
            else
                N[i] = N_prev*1 
            end
            δ_total[i] = δ_current*1 
            i+=1
        end
    end

    return N, δ_total
end

struct SurrogateTerms
    μ_ε::Vector{<:Float64}
    μ_θ::Vector{<:Float64}
    Σ_εε::Matrix{<:Float64}
    Σ_εθ::Matrix{<:Float64}
    Σ_θθ_inv::Matrix{<:Float64}
    # dist::MvNormal
end

@everywhere function birth_death_deterministic_total(t, β, δ=1.0, N0=1)
    return N0 .* exp.((β - δ) .* t)
end

@everywhere function birth_death_deterministic(t, β, δ=1.0, N0=1)
    N = birth_death_deterministic_total(t, β, δ, N0)

    f(u, p) = sum(birth_death_deterministic_total.(u, Ref(β), Ref(δ), Ref(N0)))
    area = [solve(IntegralProblem(f, 0., t_i), QuadGKJL()).u for t_i in t]
    δ_total = area .* δ

    return N, δ_total
end

function generate_surrogate(t_single, N0, lb, ub, num_points, num_dims, len_t)
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    y_stochastic = zeros(len_t, num_points)
    y_deterministic = zeros(len_t, num_points)

    p = Progress(num_points; dt=1.0)
    Threads.@threads for i in 1:num_points
        y_stochastic[:, i] .= vcat(birth_death_firstreact(t_single, grid[:, i]..., N0)...)
        y_deterministic[:, i] .= vcat(birth_death_deterministic(t_single, grid[:, i]..., N0)...)
        next!(p)
    end
    finish!(p)

    Y = vcat(y_stochastic .- y_deterministic, grid)
    μ_Nd = mean(Y, dims=2)
    μ_ε = μ_Nd[1:len_t]
    μ_θ = μ_Nd[(end-num_dims+1):end]

    Y .-= μ_Nd
    Σ_Nd =  (Y*Y') ./ (num_points-1)
    Σ_εε = Σ_Nd[1:len_t, 1:len_t]
    Σ_εθ = Σ_Nd[1:len_t, (end-num_dims+1):end]
    Σ_θθ = Σ_Nd[(end-num_dims+1):end, (end-num_dims+1):end]
    Σ_θθ_inv = inv(Σ_θθ)
    # Σ_εgθ = Σ_εε - Σ_εθ*Σ_θθ_inv*Σ_εθ'
    # Σ_εgθ .= (Σ_εgθ+Σ_εgθ') ./2.

    # dist = MvNormal(zeros(length(μ_ε)), Σ_εgθ)

    # surrogate_terms = (μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, dist)
    surrogate_terms = (μ_ε, μ_θ, Σ_εε, Σ_εθ, Σ_θθ_inv)
    bson(surrogate_location, Dict(:s=>SurrogateTerms(surrogate_terms...)))
    return surrogate_terms
end

@everywhere function stochastic_error(θ, y, Σ_εε, Σ_εθ, Σ_θθ_inv)
    Σ_εgθ = Σ_εε - Σ_εθ*Σ_θθ_inv*(θ*y')/(length(θ)*length(y'))
    Σ_εgθ .= (Σ_εgθ+Σ_εgθ') ./2.
    return Σ_εgθ
end

@everywhere function mean_correction(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)
    return μ_ε + Σ_εθ*Σ_θθ_inv*(θ - μ_θ) 
end

@everywhere function loglhood(θ, data)
    # μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, dist = data.surrogate_terms
    μ_ε, μ_θ, Σ_εε, Σ_εθ, Σ_θθ_inv = data.surrogate_terms

    y = vcat(birth_death_deterministic(data.t_single, θ[1], θ[2], N0)...)

    Σ_εgθ = stochastic_error(θ, y, Σ_εε, Σ_εθ, Σ_θθ_inv)
    dist = MvNormal(zeros(length(μ_ε)), Σ_εgθ)

    μ_εgθ = mean_correction(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)
    e = loglikelihood(dist, y .+ μ_εgθ .- data.y_obs)
    return e
end

# predicts the mean of the data distribution
@everywhere function predictfunction(θ, data, t=data.t_single)
    # μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, _ = data.surrogate_terms
    μ_ε, μ_θ, _, Σ_εθ, Σ_θθ_inv = data.surrogate_terms

    y = hcat(birth_death_deterministic(t, θ[1], θ[2], N0)...)
    μ_εgθ = mean_correction(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)

    y[:,1] .= y[:,1] .+ μ_εgθ[1:size(y,1)]
    y[:,2] .= y[:,2] .+ μ_εgθ[size(y,1)+1:end]

    return y 
end

@everywhere function errorfunction(predictions, θ, region, dist=data.surrogate_terms[5])
    THdelta = 1.0 - region
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    # find pointwise HDR - note each individual sample is itself a normal distribution 
    # take a large number of samples from the MvNormal distribution
    samples = permutedims(rand(dist, 10000))
    for j in axes(predictions, 2)
        for i in axes(predictions, 1)
            norm_dist = fit_mle(Normal, samples[:, Int(i + (j-1)*size(predictions, 1))])

            lq[i,j] = predictions[i,j] + quantile(norm_dist, THdelta / 2.0)
            uq[i,j] = predictions[i,j] + quantile(norm_dist, 1 - (THdelta / 2.0))
        end
    end

    return lq, uq
end

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θ_true, generator_args::NamedTuple)
    y_obs = vcat(birth_death_firstreact(generator_args.t_single, θ_true..., N0)...)
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

# Data setup ###########################################################################
using Random, Distributions, Statistics, LatinHypercubeSampling
data_location = joinpath("Experiments", "Models", "Data", "birth-death_stochastic_identifiable.csv")
surrogate_location = joinpath("Experiments", "Models", "Data", "surrogate_terms_identifiable.bson")
function data_setup(t, t_single, θ_true, N0, generate_new_data=false)
    new_data = false
    if !generate_new_data && isfile(data_location)
        t, y_obs = eachcol(CSV.read(data_location, DataFrame))
    else
        Random.seed!(12348)
        y_obs = vcat(birth_death_firstreact(t_single, θ_true..., N0)...)

        println(y_obs)
        data_df = DataFrame(t=t, y_obs=y_obs)
        display(data_df)
        CSV.write(data_location, data_df)
        new_data = true
    end
    return t, y_obs, new_data
end

function parameter_and_data_setup()
    # true parameters
    birth_rate = 0.5; death_rate = 0.4; 
    global N0 = 1000
    θ_true = [birth_rate, death_rate]
    t_single = LinRange(0.1, 3, 100)
    t = vcat(t_single, t_single)
    len_t = length(t)
    t, y_obs, new_data = data_setup(t, t_single, θ_true, N0)

    # surrogate arguments
    lb = [0.01, 0.01]
    # ub = [2.5, 2.5]
    ub = [2.0, 2.0]
    num_points = 100000
    num_dims=2

    if new_data || !isfile(surrogate_location)
        surrogate_terms = generate_surrogate(t_single, N0, lb, ub, num_points, num_dims, len_t)
    else
        st = BSON.load(surrogate_location, @__MODULE__)[:s]
        display(st)
        surrogate_terms = ([getfield(st, k) for k ∈ fieldnames(SurrogateTerms)]...,)
    end
    
    data = (t=t, t_single=t_single, y_obs=y_obs, surrogate_terms=surrogate_terms)

    # Named tuple of all data required within the log-likelihood function
    training_gen_args = (t=t, t_single=t_single, surrogate_terms=surrogate_terms, is_test_set=false)
    testing_gen_args  = (t=t, t_single=t_single, surrogate_terms=surrogate_terms, is_test_set=true)

    t_pred=LinRange(0.1, 3, 100)

    θG = [birth_rate, death_rate]
    θnames = [:β, :δ]
    par_magnitudes = [1,1]

    return data, training_gen_args, testing_gen_args, θ_true, t_pred, θnames, 
        θG, lb, ub, par_magnitudes
end

data, training_gen_args, testing_gen_args, θ_true, t_pred, θnames, 
    θG, lb, ub, par_magnitudes = parameter_and_data_setup()
