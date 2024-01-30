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
    Σ_εθ::Matrix{<:Float64}
    Σ_θθ_inv::Matrix{<:Float64}
    dist::MvNormal
end

@everywhere function birth_death_deterministic(t, β, δ=1.0, N0=1)
    return N0 .* exp.((β - δ) .* t)
end

function generate_surrogate(t, N0, lb, ub, num_points, num_dims, len_t)
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    y_stochastic = zeros(len_t, num_points)
    y_deterministic = zeros(len_t, num_points)

    Threads.@threads for i in 1:num_points
        y_stochastic[:, i] .= birth_death_firstreact(t, grid[:, i]..., N0)[1]
        y_deterministic[:, i] .= birth_death_deterministic(t, grid[:, i]..., N0)[1]
    end

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
    Σ_εgθ = Σ_εε - Σ_εθ*Σ_θθ_inv*(Σ_εθ') # this term is fixed
    Σ_εgθ .= (Σ_εgθ+Σ_εgθ') ./2.
    # Σ_εgθ = Σ_εgθ - 1.5 * minimum(eigvals(Σ_εgθ)) * I

    dist = MvNormal(zeros(length(μ_ε)), Σ_εgθ)

    surrogate_terms = (μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, dist)
    bson(surrogate_location, Dict(:s=>SurrogateTerms(surrogate_terms...)))
    return surrogate_terms
end

@everywhere function mean_correction(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)
    return μ_ε + Σ_εθ*Σ_θθ_inv*(θ - μ_θ) 
end

@everywhere function loglhood(θ, data)
    μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, dist = data.surrogate_terms

    y = birth_death_deterministic(data.t, θ[1], θ[2], N0)

    μ_εgθ = mean_correction(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)
    e = loglikelihood(dist, y .+ μ_εgθ .- data.y_obs)
    return e
end

# predicts the mean of the stochastic data distribution
@everywhere function predictFunc(θ, data, t=data.t)
    μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, _ = data.surrogate_terms

    y = birth_death_deterministic(t, θ[1], θ[2], N0)

    return y .+ mean_correction(θ, μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv)
end

@everywhere function errorFunc(predictions, θ, region, dist=data.surrogate_terms[5])
    THdelta = 1.0 - region
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    # find pointwise HDR - note each individual sample is itself a normal distribution 
    # take a large number of samples from the MvNormal distribution
    samples = permutedims(rand(dist, 10000))
    for i in eachindex(predictions)
        norm_dist = fit_mle(Normal, samples[:,i])
        lq[i] = predictions[i] + quantile(norm_dist, THdelta / 2.0)
        uq[i] = predictions[i] + quantile(norm_dist, 1 - (THdelta / 2.0))
    end

    return lq, uq
end

# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θ_true, generator_args::NamedTuple)
    y_obs = birth_death_firstreact(generator_args.t, θ_true..., N0)[1]
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

##########################################################
# Well informed direction
xytoXY_sip(xy) = [xy[1]-xy[2]; xy[1]+xy[2]]
XYtoxy_sip(XY) = [(XY[1]+XY[2])/2.; (XY[2]-XY[1])/2.]

function loglhood_XYtoxy_sip(Θ,data); loglhood(XYtoxy_sip(Θ), data) end

function predictFunc_XYtoxy_sip(Θ,data, t=data.t); predictFunc(XYtoxy_sip(Θ), data, t) end

function errorFunc_XYtoxy_sip(predictions, Θ, region); errorFunc(predictions, XYtoxy_sip(Θ), region) end

function data_generator_XYtoxy_sip(Θ, generator_args::NamedTuple); data_generator(XYtoxy_sip(Θ), generator_args) end

# Data setup ###########################################################################
using Random, Distributions, Statistics, LatinHypercubeSampling
data_location = joinpath("Experiments", "Models", "Data", "birth-death_stochastic.csv")
surrogate_location = joinpath("Experiments", "Models", "Data", "surrogate_terms.bson")
function data_setup(t, θ_true, N0, generate_new_data=false)
    new_data = false
    if !generate_new_data && isfile(data_location)
        t, y_obs = eachcol(CSV.read(data_location, DataFrame))
    else
        Random.seed!(12348)
        y_obs, _ = birth_death_firstreact(t, θ_true..., N0)

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
    global N0 = 5000
    θ_true = [birth_rate, death_rate]
    t = LinRange(0.1, 2, 101)
    len_t = length(t)
    t, y_obs, new_data = data_setup(t, θ_true, N0)

    # surrogate arguments
    lb = [0.01, 0.01]
    ub = [2., 2.0]
    num_points = 100000
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
    testing_gen_args  = (t=t, surrogate_terms=surrogate_terms, is_test_set=true)

    t_pred=LinRange(0.1, 2, 101)

    θG = [birth_rate, death_rate]
    θnames = [:β, :δ]
    par_magnitudes = [1,1]

    θnames_sip = [:β_minus_δ, :β_plus_δ]
    θG_sip = xytoXY_sip(θG)
    lb_sip, ub_sip = transformbounds_NLopt(xytoXY_sip, lb, ub)

    return data, training_gen_args, testing_gen_args, θ_true, t_pred, θnames, 
        θG, lb, ub, par_magnitudes, θnames_sip, θG_sip, lb_sip, ub_sip
end

data, training_gen_args, testing_gen_args, θ_true, t_pred, θnames, 
    θG, lb, ub, par_magnitudes, θnames_sip, θG_sip, lb_sip, ub_sip = parameter_and_data_setup()
