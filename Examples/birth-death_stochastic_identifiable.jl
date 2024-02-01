using Random, Distributions
using LatinHypercubeSampling
using Statistics

using LikelihoodBasedProfileWiseAnalysis

function birth_death_firstreact(t_max, β, δ=1., N0=1, t_init=0.0)

    t = [t_init*1]
    N = [N0*1]
    δ_total = [0]

    N_total = N0*1

    while t[end] < t_max && N_total != 0
        h_i = [β*N_total, δ*N_total]

        # delta t's for each process
        delta_t_birth = rand(Exponential(1/h_i[1]))
        delta_t_death = rand(Exponential(1/h_i[2]))

        if delta_t_birth <= delta_t_death 
            N_total += 1
            delta_t = delta_t_birth
            push!(δ_total, δ_total[end])
        else
            N_total -= 1
            delta_t = delta_t_death
            push!(δ_total, δ_total[end]+1)
        end

        push!(t, t[end] + delta_t)
        push!(N, N_total*1)
    end 

    return t, N, δ_total
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

function stochastic_at_t(t_interest, t, N, δ_total)
    return vcat(stochastic_at_t(t_interest[1:Int(len_t/2)], t, N), stochastic_at_t(t_interest[Int(len_t/2)+1:end], t, δ_total))
end

function birth_death_deterministic(t, β, δ=1., N0=1)
    return N0 .* exp.((β-δ) .* t)
end


# Some kind of design distribution (also prior):
# a_dist = 0.5 + Beta(1.2,1.2) * 1.5
# plot(collect(0:0.01:2), pdf.(a_dist, 0:0.01:2))
# b_dist = 0.25 + Beta(1.2,1.2) * 1.5
# plot(collect(0:0.01:2), pdf.(b_dist, 0:0.01:2))
# a, b = birth_death_firstreact(t[end], birth_rate, death_rate, N0)
# scatter(t, stochastic_at_t(t, a, b), opacity=0.3, label="Census")
# scatter!(a, b, opacity=0.2, label="N")
# plot!(t, birth_death_deterministic(t, birth_rate, death_rate, N0), label="deterministic")

t = LinRange(0.1, 3, 100)
t = vcat(t, t)
len_t = length(t)
birth_rate = 0.5; death_rate = 0.4; N0 = 1000
yobs = stochastic_at_t(t, birth_death_firstreact(t[end], birth_rate, death_rate, N0)...)

θG = [birth_rate, death_rate]
θnames = [:β, :δ]
lb = [0.1, 0.1]
ub = [1.6, 1.0]
num_points = 10000
num_dims=2
# num_gens=10
function generate_surrogate()
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    y_stochastic = zeros(length(t), num_points)

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

    data = (t=t, yobs=yobs, surrogate_terms = (μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, Σ_εgθ))

    return data
end

function loglhood(θ, data)
    μ_ε, μ_θ, Σ_εθ, Σ_θθ_inv, Σ_εgθ = data.surrogate_terms

    μ_εgθ = μ_ε + Σ_εθ*Σ_θθ_inv*(θ - μ_θ) 
    dist = MvNormal(zeros(length(μ_εgθ)), Σ_εgθ)
    e = loglikelihood(dist, μ_εgθ .- data.yobs)
    return e
end

data = generate_surrogate()

model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub);

univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals!(model, profile_type=EllipseApprox())
univariate_confidenceintervals!(model)
get_points_in_intervals!(model, 30, additional_width =0.2)

using Plots; gr()

plots = plot_univariate_profiles_comparison(model, 0.2, 0.4, palette_to_use=:Spectral_8, opacity=0.5)
for i in eachindex(plots); display(plots[i]) end