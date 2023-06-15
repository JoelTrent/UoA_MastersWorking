using Revise
using EllipseSampling
import Distributions
using Plots
gr()

N=8

a, b = 200.0, 1.0 
α = 0.240*π
Cx, Cy= 2.0, 2.0
points=generate_N_equally_spaced_points(N, a, b, α, Cx, Cy, start_point_shift=0.0)
plt=scatter(points[1,:], points[2,:], label=nothing, mw=0, ms=8)

Hw11 = (cos(α)^2 / a^2 + sin(α)^2 / b^2)
Hw22 = (sin(α)^2 / a^2 + cos(α)^2 / b^2)
Hw12 = cos(α)*sin(α)*(1/a^2 - 1/b^2)
Hw_norm = [Hw11 Hw12; Hw12 Hw22]

confidence_level=0.95
Hw = Hw_norm ./ (0.5 ./ (Distributions.quantile(Distributions.Chisq(2), confidence_level)*0.5))
Γ = convert.(Float64, inv(BigFloat.(Hw, precision=64)))

using BenchmarkTools
@btime EllipseSampling.calculate_ellipse_parameters(Γ, 1, 2, confidence_level)

a_calc, b_calc, _, _, α_calc = EllipseSampling.calculate_ellipse_parameters(Γ, 1, 2, confidence_level)

println("a=", a_calc)
println("b=", b_calc)
println("α/pi=", α_calc/pi)

points2 = generate_N_equally_spaced_points(N, Γ, [Cx, Cy], 1, 2, confidence_level=confidence_level, start_point_shift=0.0)
scatter!(points2[1, :], points2[2, :], label=nothing, markeralpha=0.8, mw=0, ms=4)

display(plt)
savefig(plt, joinpath("test","precision_issue_0.251pi.png"))

function ellipse_loglike(θ::Vector,
    mleTuple::@NamedTuple{θmle::Vector{T}, Hmle::Matrix{T}}) where {T<:Float64}
    return -0.5 * ((θ - mleTuple.θmle)' * mleTuple.Hmle * (θ - mleTuple.θmle))
end

# [ellipse_loglike(points[:,i], (θmle=[Cx, Cy], Hmle=Hw)) for i in 1:N]