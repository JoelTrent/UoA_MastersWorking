include(joinpath("..","JuLikelihood.jl"))

λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50., 150.)
C0min, C0max = (0.01, 50.)

lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]

function forward_parameter_transformKminusC0(θ::Vector{<:T}) where T<:Real
    Θ=zeros(T, length(θ))
    Θ .= θ
    Θ[2] = θ[2]-θ[3]
    return Θ
end

@time transformbounds_NLopt(forward_parameter_transformKminusC0, lb, ub)
@time transformbounds(forward_parameter_transformKminusC0, lb, ub, Int[1,3], Int[2])


λmin, λmax = (-1.0, 1.05)
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]

function forward_parameter_transformλsquared(θ::Vector{<:T}) where T<:Real
    Θ=zeros(T, length(θ))
    Θ .= θ
    Θ[1] = (θ[1])^2
    return Θ
end

# FAILS because function is not monotonic on the supplied bounds
newlb, newub = transformbounds(forward_parameter_transformλsquared, lb, ub, Int[2,3], Int[1])
-sqrt(newub[1])
sqrt(newlb[1])

# SUCCEEDS because it's able to find the combination of lb[i] and ub[i] that produces the correct minima
newlb, newub = transformbounds_NLopt(forward_parameter_transformλsquared, lb, ub)
-sqrt(newub[1])
sqrt(newlb[1])