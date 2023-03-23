using InformationGeometry, Plots
using DifferentialEquations, Distributions
DS = DataSet([1,2,3,4], [4,5,6.5,9], [0.5,0.45,0.6,1])
model(x::Real, θ::AbstractVector{<:Real}) = θ[1] * x + θ[2]
DM = DataModel(DS, model)
plot(DM)

sols = ConfidenceRegions(DM, 1:2; tol=1e-9);
VisualizeSols(DM, sols)


model2(x::Real, θ::AbstractVector{<:Real}) = θ[1]^3 * x + exp(θ[1] + θ[2])
DM2 = DataModel(DS, model2)
sols2 = ConfidenceRegions(DM2, 1:2; tol=1e-9);
VisualizeSols(DM2, sols2)

plot(DM2)
ConfidenceBands(DM2, sols2[2])

DM3 = DataModel(DS, (x,θ)-> θ[1]^3 * x + exp(θ[1] + θ[2]) + θ[3] * sin(x))
Planes, sols3 = ConfidenceRegion(DM3, 1; tol=1e-6, Dirs=(1,2,3), N=50);
VisualizeSols(DM3, Planes, sols3)

plot(DM3)
ConfidenceBands(DM3, sols3)


# Logistic model example
# function DE!(dC, C, p, t)
#     λ,K=p
#     dC[1]= λ * C[1] * (1.0 - C[1]/K)
# end

# function odesolver(t, λ, K, C0)
#     p=(λ,K)
#     tspan=(0.0, maximum(t))
#     prob=ODEProblem(DE!, [C0], tspan, p)
#     sol=solve(prob, saveat=t)
#     return sol[1,:]
# end

# function model(t, a::AbstractVector{<:Real})
#     y=odesolver(t, a[1],a[2],a[3])
#     return y
# end

# λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
# tt=0:5:1000
# a=[λ, K, C0]

# # true data
# data0 = model(t, a)

# # noisy data
# data = data0 + σ*randn(length(t))

# DS = DataSet(collect(t), data, fill(σ, length(data)))

# modelIntegral(x::Real, θ::AbstractVector{<:Real}) = θ[2] * θ[3] / ((θ[2]-θ[3]) * exp(-θ[1]*x) + θ[3])
# # DM = DataModel(DS, model, [0.01, 100.0, 10.0], true)
# DM = DataModel(DS, modelIntegral)


# ProfileLikelihood(DM, 1)[1]

# sols4 = ConfidenceRegion(DM, 1, tol=1e-8, Dirs=(1,2,3), N=10);