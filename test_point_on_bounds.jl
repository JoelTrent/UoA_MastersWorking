include("JuLikelihood.jl")


internalpoint = [1.0,2.0]
direction = 1/3
ind1=1
ind2=2
lb = [-2, -1, 5, 7] .* 1.0
ub = [5, 4, 10, 12] .* 1.0


findpointonbounds(internalpoint, 1.7, lb, ub, ind1, ind2)

function find_m_spaced_radialdirections(num_directions::Int)
    radial_dirs = zeros(num_directions)

    radial_dirs .= rand() * 2.0 / convert(Float64, num_directions) .+ collect(LinRange(1e-12, 2.0, num_directions+1))[1:end-1]

    return radial_dirs
end

findmradialdirections(20)