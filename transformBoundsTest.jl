include("transformBounds.jl")

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

transformbounds_NLP(forward_parameter_transformKminusC0, lb, ub)
transformbounds(forward_parameter_transformKminusC0, lb, ub, Int[1,3], Int[2])