##############################################################################
######################## BIRTH-DEATH STOCHASTIC MODEL ########################
##############################################################################

using BSON

# Model functions
###########################################################################
function birth_death_firstreact(t_max, β, δ=1., N0=1, t_init=0.0)
    t = [t_init*1]
    N = [N0*1]

    N_total = N0*1

    while t[end] < t_max && N_total != 0
        h_i = [β*N_total, δ*N_total]

        # delta t's for each process
        delta_t_birth = rand(Exponential(1/h_i[1]))
        delta_t_death = rand(Exponential(1/h_i[2]))

        if delta_t_birth <= delta_t_death 
            N_total += 1
            delta_t = delta_t_birth
        else
            N_total -= 1
            delta_t = delta_t_death
        end

        push!(t, t[end] + delta_t)
        push!(N, N_total*1)
    end 

    return t, N
end

function stochastic_at_t(t_interest, t, N)
    N_interest = zeros(length(t_interest))

    reached_end_of_events = (false, 0)
    for (i, t_i) in enumerate(t_interest)
        j = findfirst(t_i .< t)

        if isnothing(j)
            reached_end_of_events = (true, i)
            break
        end
        if j == 1; j = 2 end
        N_interest[i] = N[j-1]
    end

    if reached_end_of_events[1]
        i = reached_end_of_events[2]
        N_interest[(i+1):end] .= N_interest[i]
    end

    return N_interest
end

struct SurrogateTerms
    μ_ε::Vector{<:Float64}
    μ_θ::Vector{<:Float64}
    Σ_εθ::Matrix{<:Float64}
    Σ_θθ_inv::Matrix{<:Float64}
    dist::MvNormal
end

function generate_surrogate(t, N0, lb, ub, num_points, num_dims, len_t)
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    y_stochastic = zeros(len_t, num_points)

    for i in 1:num_points
        y_stochastic[:, i] .= stochastic_at_t(t, birth_death_firstreact(t[end], grid[:,i]..., N0)...)
    end

    Y = vcat(y_stochastic, grid)
    μ_Nd = mean(Y, dims=2)
    μ_ε = μ_Nd[1:len_t]
    μ_θ = μ_Nd[(end-num_dims+1):end]

    Y .-= μ_Nd
    Σ_Nd =  (Y*Y') ./ (num_points-1)
    Σ_εε = Σ_Nd[1:len_t, 1:len_t]
    Σ_εθ = Σ_Nd[1:len_t, (end-num_dims+1):end]
    Σ_θθ = Σ_Nd[(end-num_dims+1):end, (end-num_dims+1):end]
    Σ_θθ_inv = inv(Σ_θθ)
    Σ_εgθ = Σ_εε - Σ_εθ*Σ_θθ_inv*Σ_εθ'
    Σ_εgθ .= (Σ_εgθ+Σ_εgθ') ./2.

    dist = MvNormal(zeros(length(μ_ε)), Σ_εgθ)

    surrogate_terms = (μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, dist)
    bson(surrogate_location, Dict(:s=>SurrogateTerms(surrogate_terms...)))
    return surrogate_terms
end

function birth_death_deterministic(t, β, δ=1., N0=1)
    return N0 .* exp.((β-δ) .* t)
end

@everywhere function model_mean(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)
    return μ_ε + Σ_εθ*Σ_θθ_inv*(θ - μ_θ) 
end

@everywhere function loglhood(θ, data)
    μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, dist = data.surrogate_terms

    μ_εgθ = model_mean(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)
    e = loglikelihood(dist, μ_εgθ .- data.y_obs)
    return e
end

# predicts the mean of the data distribution
@everywhere function predictFunc(θ, data, t=data.t)
    μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, _ = data.surrogate_terms
    return model_mean(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)
end

@everywhere function errorFunc(predictions, θ, bcl, dist=data.surrogate_terms[5])
    THalpha = 1.0 - bcl
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    # find pointwise HDR
    # take a large number of samples from the MvNormal distribution
    samples = permutedims(rand(dist, 10000))
    for i in eachindex(predictions)
        k = kde(@view samples[:,i])
        xgrid = LinRange(minimum(@view samples[:,i]), 
                        maximum(@view samples[:,i]), 
                        10000)
        ygrid = pdf.(Ref(k), xgrid)
        threshold = first(hdr_thresholds([THalpha], ygrid))

        lq[i] = predictions[i] + xgrid[findfirst(ygrid .> threshold)]
        uq[i] = predictions[i] + xgrid[findlast(ygrid .> threshold)]
    end

    return lq, uq
end

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = stochastic_at_t(t, birth_death_firstreact(t[end], θ_true..., N0)...)
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

# Data setup ###########################################################################
using CSV, DataFrames
using Random, Distributions, Statistics, LatinHypercubeSampling
data_location = joinpath("Experiments", "Models", "Data", "birth-death_stochastic.csv")
surrogate_location = joinpath("Experiments", "Models", "Data", "surrogate_terms.bson")
function data_setup(t, θ_true, N0, generate_new_data=false)
    new_data = false
    if !generate_new_data && isfile(data_location)
        t, y_obs = eachcol(CSV.read(data_location, DataFrame))
    else
        y_obs = stochastic_at_t(t, birth_death_firstreact(t[end], θ_true..., N0)...)

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
    t = LinRange(0.1, 3, 100)
    len_t = length(t)
    t, y_obs, new_data = data_setup(t, θ_true, N0)

    # surrogate arguments
    lb = [0.1, 0.1]
    ub = [1.6, 1.0]
    num_points = 10000
    num_dims=2

    if new_data || !isfile(surrogate_location)
        surrogate_terms = generate_surrogate(t, N0, lb, ub, num_points, num_dims, len_t)
    else
        st = BSON.load(surrogate_location, @__MODULE__)[:s]
        display(st)
        surrogate_terms = ([getfield(st, k) for k ∈ fieldnames(SurrogateTerms)]...,)
    end
    
    data = (t=t, y_obs=y_obs, surrogate_terms=surrogate_terms)

    # Named tuple of all data required within the log-likelihood function
    training_gen_args = (t=t, surrogate_terms=surrogate_terms, is_test_set=false)
    testing_gen_args = (t=t, surrogate_terms=surrogate_terms, is_test_set=true)

    t_pred=LinRange(0.1, 3, 100)

    θG = [birth_rate, death_rate]
    θnames = [:β, :δ]
    par_magnitudes = [1,1]

    return data, training_gen_args, testing_gen_args, θ_true, t_pred, θnames, 
        θG, lb, ub, par_magnitudes
end

data, training_gen_args, testing_gen_args, θ_true, t_pred, θnames, 
    θG, lb, ub, par_magnitudes = parameter_and_data_setup()
