
using EllipseSampling
using LinearAlgebra
using Plots
gr()

e = construct_ellipse(10., 1.0)
num_points=5

# place N equally spaced points on boundary of ellipse approximation at some small confidence level.
# generateN_equally_spaced_points(10, Γ, θmle, ind1, ind2)
points = generate_N_equally_spaced_points(num_points, e, start_point_shift=0.0)

# determine true loglikelihood function values at all N points
# ll_vals = .....
ll_vals = zeros(num_points) .- 0.03
# θmle = zeros(model.core.num_pars)
θmle = zeros(3)

gradient_i = zeros(num_points)
# for i in 1:num_points
    # estimate gradient of normal as function value change between mle and ellipse point ÷ euclidean distance
    

# end

# calculate normal at each point
# Also implement via forward diff - gives both magnitude and 
normal_vectors = zeros(2, num_points)

# NOTE: METHOD REQUIRES THERE TO BE AT LEAST 3 POINTS
function normal_vector!(normal_vectors, index, point1, point2)
    normal_vectors[:, index] .= [(point2[2]-point1[2]), -(point2[1]-point1[1])]
    normal_vectors[:, index] .= @view(normal_vectors[:, index]) / norm(@view(normal_vectors[:,index])) 
    return nothing
end

normal_vector!(normal_vectors, 1, @view(points[:, end]), @view(points[:, 2]))
normal_vector!(normal_vectors, num_points, @view(points[:, end-1]), @view(points[:, 1]))

for i in 2:num_points-1
   normal_vector!(normal_vectors, i, @view(points[:, i-1]), @view(points[:, i+1]))
end

boundary = scatter(points[1,:], points[2,:], aspect_ratio = :equal)
quiver!(points[1,:], points[2,:], quiver=(normal_vectors[1,:], normal_vectors[2,:]))