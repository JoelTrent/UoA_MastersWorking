using InformationGeometry, Plots
using DifferentialEquations, Distributions
# DS = DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1])
# model(x::Real, θ::AbstractVector{<:Real}) = θ[1] * x + θ[2]
# DM = DataModel(DS, model)
# plot(DM)

# sols = ConfidenceRegions(DM, 1:2; tol=1e-9);
# VisualizeSols(DM, sols)


# model2(x::Real, θ::AbstractVector{<:Real}) = θ[1]^3 * x + exp(θ[1] + θ[2])
# DM2 = DataModel(DS, model2)
# sols2 = ConfidenceRegions(DM2, 1:2; tol=1e-9);
# VisualizeSols(DM2, sols2)

# plot(DM2)
# ConfidenceBands(DM2, sols2[2])

# DM3 = DataModel(DS, (x,θ)-> θ[1]^3 * x + exp(θ[1] + θ[2]) + θ[3] * sin(x))
# Planes, sols3 = ConfidenceRegion(DM3, 1; tol=1e-6, Dirs=(1,2,3), N=50);
# VisualizeSols(DM3, Planes, sols3)

# plot(DM3)
# ConfidenceBands(DM3, sols3)


# Logistic model example
function DE!(dC, C, p, t)
    λ,K=p
    dC[1]= λ * C[1] * (1.0 - C[1]/K)
end

function odesolver(t, λ, K, C0)
    p=(λ,K)
    tspan=(0.0, maximum(t))
    prob=ODEProblem(DE!, [C0], tspan, p)
    sol=solve(prob, saveat=t)
    return sol[1,:]
end

function model(t, a::AbstractVector{<:Real})
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]

λ_min, λ_max = (0.00, 0.05)
K_min, K_max = (50.0, 150.0)
C0_min, C0_max = (0.0001, 50.0)
lb = [λ_min, K_min, C0_min]
ub = [λ_max, K_max, C0_max]

# true data
data0 = model(t, a)

# noisy data
using Random; Random.seed!(12348)
data = data0 + σ*randn(length(t))

DS = DataSet(collect(t), data, σ*ones(length(data)))
modelIntegral(x::Real, θ::AbstractVector{<:Real}) = (θ[2] * θ[3]) / (((θ[2] - θ[3]) * exp(-θ[1] * x)) + θ[3])

DM = DataModel(DS, ModelMap(modelIntegral, nothing, HyperCube(lb, ub)), a)
plot(DM)

ProfileLikelihood(DM, 2.5, N=20);

Planes, sols4 = ConfidenceRegion(DM, 1, tol=1e-8, Dirs=(1, 2, 3), N=100);
VisualizeSols(DM, Planes, sols4)

# boundary = vcat(Deplanarize.(Planes, sols4)...)
# ys = zeros(size(boundary, 1), length(t))
# for i in 1:size(boundary,1)
#     ys[i, :] .= model(t, boundary[i,:])
# end

# plot(t, minimum(ys, dims=1)')
# plot!(t, maximum(ys, dims=1)')



# Lotka-Volterra
# using StaticArrays
# function lotka_static(C, p, t)
#     dC_1 = p[1] * C[1] - C[1] * C[2]
#     dC_2 = p[2] * C[1] * C[2] - C[2]
#     SA[dC_1, dC_2]
# end

# function odesolver(t, α, β, C01, C02)
#     p = SA[α, β]
#     C0 = SA[C01, C02]
#     tspan = (0.0, 21)
#     prob = ODEProblem(lotka_static, C0, tspan, p)
#     sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=t)
#     return sol[1, :], sol[2, :]
# end

# function ODEmodel(t, θ)
#     return odesolver(t, θ[1], θ[2], θ[3], θ[4])
# end

# α_true = 0.9;
# β_true = 1.1;
# x0_true = 0.8;
# y0_true = 0.3;
# t = LinRange(0, 10, 21)
# σ = 0.2

# θ_true = [α_true, β_true, x0_true, y0_true]

# # true data
# y_true = hcat(ODEmodel(t, θ_true)...)

# Random.seed!(12348)
# # noisy data
# data = y_true .+ σ * randn(length(t), 2)


# αmin, αmax = (0.7, 1.2)
# βmin, βmax = (0.7, 1.4)
# x0min, x0max = (0.5, 1.2)
# y0min, y0max = (0.1, 0.5)
# lb = [αmin, βmin, x0min, y0min]
# ub = [αmax, βmax, x0max, y0max]


# DS = DataSet(collect(t), data, σ * ones(length(data)))

# DM = DataModel(DS, ModelMap(ODEmodel, nothing, HyperCube(lb, ub)), θ_true)
# plot(DM)

# ProfileLikelihood(DM, 2.5, N=20);

# Planes, sols4 = ConfidenceRegion(DM, 1, tol=1e-8, Dirs=(1, 2, 3), N=100);
# VisualizeSols(DM, Planes, sols4)
