"""
Distorts uniformly spaced angles on a circle to angles on an ellipse representative of the relative magnitude of each parameter. If the magnitude of a parameter is a NaN value (i.e. either bound is Inf), then the relative magnitude is set to 1.0, as no information is known about its magnitude.

Angles are anticlockwise
"""
function findNpointpairs_radialMLE!(p::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    ind1::Int, 
                                    ind2::Int,
                                    bound_warning::Bool,
                                    radial_start_point_shift::Float64)

    mle_point = model.core.θmle[[ind1, ind2]]
    external = zeros(2,num_points)
    point_is_on_bounds = falses(num_points)
    
    if isnan(model.core.θmagnitudes[ind1]) || isnan(model.core.θmagnitudes[ind2]) 
        relative_magnitude = 1.0
    else
        relative_magnitude = model.core.θmagnitudes[ind1]/model.core.θmagnitudes[ind2]
    end

    radial_dirs = find_m_spaced_radialdirections(num_points, start_point_shift=radial_start_point_shift)

    for i in 1:num_points
        dir_vector = [relative_magnitude * cospi(radial_dirs[i]), sinpi(radial_dirs[i]) ]
        external[:,i], bound_ind, upper_or_lower = findpointonbounds(model, mle_point, dir_vector, ind1, ind2, true)

        # if bound point is a point inside the boundary, note that this is the case
        p.pointa .= external[:,i]
        g = bivariate_optimiser(0.0, p)
        if g ≥ 0
            point_is_on_bounds[i] = true

            if bound_warning
                @warn string("The ", upper_or_lower, " bound on variable ", model.core.θnames[bound_ind], " is inside the confidence boundary")
                bound_warning = false
            end
        else
            # make bracket a tiny bit smaller
            if isinf(g)
                v_bar = external[:,i] .- mle_point
                external[:,i] .= mle_point .+ ((1.0-1e-12) .* v_bar)
            end
        end
    end

    return external, point_is_on_bounds, bound_warning
end

"""
    edge_length(boundary, inds1, inds2, relative_magnitude)

Euclidean distance between two vertices (length of an edge), scaled by the relative magnitude of parameters, so that each dimension has roughly the same weight.
"""
function edge_length(boundary, inds1::Union{UnitRange, AbstractVector}, inds2::Union{UnitRange, AbstractVector}, relative_magnitude)
    return colwise(Euclidean(), 
    boundary[:, inds1] ./ SA[relative_magnitude, 1.0], 
    boundary[:, inds2] ./ SA[relative_magnitude, 1.0]) 
end

function edge_length(boundary, ind1::Int, ind2::Int, relative_magnitude)
    return evaluate(Euclidean(), 
    boundary[:, ind1] ./ SA[relative_magnitude, 1.0], 
    boundary[:, ind2] ./ SA[relative_magnitude, 1.0]) 
end

function edge_length(boundary, candidate_point, index::Int, relative_magnitude)
    return evaluate(Euclidean(), 
    candidate_point ./ SA[relative_magnitude, 1.0], 
    boundary[:, index] ./ SA[relative_magnitude, 1.0]) 
end

"""
internal_angle_from_pi!(vertex_internal_angle_objs, indexes, boundary, adjacent_vertices)

The magnitude the internal angle in radians between two adjacent edges is from pi radians - i.e. how far away the two edges are from representing a straight boundary. If a boundary is straight then the objective is 0.0 radians, whereas if the boundary has an internal angle of pi/4 radians (45 deg) the objective is pi*3/4 (135 deg). Computes this by considering the angle between the two vectors that can be used to represent the edges (using `AngleBetweenVectors.jl`).
"""
function internal_angle_from_pi!(vertex_internal_angle_objs, indexes::UnitRange, boundary, edge_clock, edge_anti, relative_magnitude)
    for i in indexes
        vertex_internal_angle_objs[i] = AngleBetweenVectors.angle((boundary[:,i] .- boundary[:, edge_clock[i]]) ./ SA[relative_magnitude, 1.0], 
                                                                    (boundary[:,edge_anti[i]] .- boundary[:,i])./ SA[relative_magnitude, 1.0])
    end
    return nothing
end

function internal_angle_from_pi(index::Int, boundary, edge_clock, edge_anti, relative_magnitude)
    if index == edge_clock[index] && index == edge_anti[index]
        return 0.0
    end 
    return AngleBetweenVectors.angle((boundary[:,index] .- boundary[:, edge_clock[index]])./ SA[relative_magnitude, 1.0],
                                        (boundary[:,edge_anti[index]] .- boundary[:, index])./ SA[relative_magnitude, 1.0]) 
end

function iterativeboundary_init(bivariate_optimiser::Function, 
                                model::LikelihoodModel, 
                                num_points::Int, 
                                p::NamedTuple, 
                                ind1::Int, 
                                ind2::Int,
                                initial_num_points::Int,
                                biv_opt_is_ellipse_analytical::Bool,
                                radial_start_point_shift::Float64,
                                ellipse_sqrt_distortion::Float64,
                                use_ellipse::Bool,
                                save_internal_points::Bool)

    boundary = zeros(2, num_points)
    boundary_all = zeros(model.core.num_pars, num_points)
    internal_all = zeros(model.core.num_pars, save_internal_points ? num_points : 0)
    ll_values = zeros(save_internal_points ? num_points : 0)
    internal_count = 0
    point_is_on_bounds = falses(num_points)
    # warn if bound prevents reaching boundary
    bound_warning=true

    if use_ellipse
        _, _, _, external, point_is_on_bounds_external, bound_warning = findNpointpairs_radialMLE!(p, bivariate_optimiser, model, 
                                                                initial_num_points, ind1, ind2, 
                                                                0.1, radial_start_point_shift, ellipse_sqrt_distortion)
    else
        external, point_is_on_bounds_external, bound_warning = findNpointpairs_radialMLE!(p, bivariate_optimiser, model, 
                                                                initial_num_points, ind1, ind2, bound_warning, radial_start_point_shift)
    end

    point_is_on_bounds[1:initial_num_points] .= point_is_on_bounds_external[:]

    mle_point = model.core.θmle[[ind1,ind2]]
    for i in 1:initial_num_points
        if point_is_on_bounds[i]
            p.pointa .= external[:,i]
            bivariate_optimiser(0.0, p)
            boundary[:,i] .= external[:,i]
            boundary_all[[ind1, ind2], i] .= external[:,i]
        else
            p.pointa .= mle_point .* 1.0
            v_bar = external[:,i] .- mle_point

            v_bar_norm = norm(v_bar, 2)
            p.uhat .= v_bar ./ v_bar_norm

            Ψ = find_zero(bivariate_optimiser, (0.0, v_bar_norm), Roots.Brent(); p=p)
            
            boundary[:,i] .= p.pointa + Ψ*p.uhat
            boundary_all[[ind1, ind2], i] .= boundary[:,i]
        end
        if !biv_opt_is_ellipse_analytical
            variablemapping2d!(@view(boundary_all[:, i]), p.λ_opt, p.θranges, p.λranges)
        end
    end

    if initial_num_points == num_points
        return true, boundary_all, PointsAndLogLikelihood(internal_all[:,1:internal_count], ll_values[1:internal_count])
    end 

    num_vertices = initial_num_points * 1

    if isnan(model.core.θmagnitudes[ind1]) || isnan(model.core.θmagnitudes[ind2]) 
        relative_magnitude = 1.0
    else
        relative_magnitude = model.core.θmagnitudes[ind1]/model.core.θmagnitudes[ind2]
    end

    # edge i belongs to vertex i, and gives the vertex it is connected to (clockwise)
    edge_clock = zeros(Int, num_points)
    edge_clock[2:num_vertices] .= 1:(num_vertices-1)
    edge_clock[1] = num_vertices

    # edge i belongs to vertex i, and gives the vertex it is connected to (anticlockwise)
    edge_anti = zeros(Int, num_points)
    edge_anti[1:num_vertices-1] .= 2:num_vertices
    edge_anti[num_vertices] = 1

    edge_anti_on_bounds = falses(num_points)
    edge_anti_on_bounds[1:num_vertices] .= @view(point_is_on_bounds[1:num_vertices]) .&& @view(point_is_on_bounds[@view(edge_anti[1:num_vertices])])
    
    # tracked heap for length of edges (anticlockwise)
    edge_lengths = zeros(num_points)
    edge_lengths[num_vertices+1:end] .= -Inf

    edge_lengths[1:num_vertices] .= edge_length(boundary, 1:num_vertices, 
                                                @view(edge_anti[1:num_vertices]), 
                                                relative_magnitude)

    # edge_vectors = zeros(2, num_points)
    # edge_vectors[:, 1:num_vertices] .= boundary[:, edges[2,1:num_vertices]] .- boundary[:, edges[1,1:num_vertices]]


    edge_heap = TrackingHeap(Float64, S=NoTrainingWheels, O=MaxHeapOrder, N = 2,
                                init_val_coll=edge_lengths)
    
    # internal angle function
    # tracked heap for internal angles between adjacent edges (more specifically, angle away from 180 deg - i.e. a straight boundary)

    vertex_internal_angle_objs = zeros(num_points)
    internal_angle_from_pi!(vertex_internal_angle_objs, 1:num_vertices, boundary, edge_clock, edge_anti, relative_magnitude)
  
    angle_heap = TrackingHeap(Float64, S=NoTrainingWheels, O=MaxHeapOrder, N = 2, init_val_coll=vertex_internal_angle_objs)


    return false, boundary, boundary_all, internal_all, ll_values, internal_count, point_is_on_bounds, edge_anti_on_bounds, bound_warning, mle_point, num_vertices, edge_clock, edge_anti, edge_heap, angle_heap, relative_magnitude
end


"""
newboundarypoint!(p::NamedTuple, point_is_on_bounds::BitVector, edge_anti_on_bounds::BitVector, boundary::Matrix{Float64}, 
    boundary_all::Matrix{Float64}, internal_all::Matrix{Float64}, ll_values::Vector{Float64}, internal_count::Int,
    bivariate_optimiser::Function, model::LikelihoodModel, edge_anti::Vector{Int}, num_vertices::Int, ind1::Int, ind2::Int,
    biv_opt_is_ellipse_analytical::Bool, ve1::Int, ve2::Int, relative_magnitude::Float64, bound_warning::Bool, save_internal_points::Bool)


    

"""
function newboundarypoint!(p::NamedTuple,
                            point_is_on_bounds::BitVector,
                            edge_anti_on_bounds::BitVector,
                            boundary::Matrix{Float64},
                            boundary_all::Matrix{Float64},
                            internal_all::Matrix{Float64},
                            ll_values::Vector{Float64},
                            internal_count::Int,
                            bivariate_optimiser::Function, 
                            model::LikelihoodModel, 
                            edge_anti::Vector{Int},
                            num_vertices::Int,
                            ind1::Int, 
                            ind2::Int,
                            biv_opt_is_ellipse_analytical::Bool,
                            ve1::Int,
                            ve2::Int,
                            relative_magnitude::Float64,
                            bound_warning::Bool,
                            save_internal_points::Bool)

    failure = false

    # candidate point - midpoint of edge calculation
    candidate_midpoint = boundary[:, ve1] .+ 0.5 .* (boundary[:, ve2] - boundary[:, ve1])

    # find new boundary point, given candidate midpoint, the vertexes that describe the edge
    # use candidate point to find new vertex

    if edge_anti_on_bounds[ve1] # accept the mid point and find nuisance parameters
        p.pointa .= candidate_midpoint
        bivariate_optimiser(0.0, p)
        boundary[:, num_vertices] .= candidate_midpoint
        boundary_all[[ind1, ind2], num_vertices] .= candidate_midpoint
        edge_anti_on_bounds[num_vertices] = true
        point_is_on_bounds[num_vertices] = true
    else

        p.pointa .= candidate_midpoint
        dir_vector = SA[(boundary[2,ve2] - boundary[2,ve1]), -(boundary[1,ve2] - boundary[1,ve1])] .* SA[relative_magnitude, 1.0]
        g = bivariate_optimiser(0.0, p)
        if g > 0.0 # internal - push out normal to edge

            if save_internal_points
                internal_count += 1
                ll_values[internal_count] = g * 1.0
                internal_all[[ind1, ind2], internal_count] .= candidate_midpoint
                if !biv_opt_is_ellipse_analytical
                    variablemapping2d!(@view(internal_all[:, internal_count]), p.λ_opt, p.θranges, p.λranges)
                end
            end

            boundpoint, bound_ind, upper_or_lower = findpointonbounds(model, candidate_midpoint, (dir_vector ./ norm(dir_vector, 2)), ind1, ind2, true)

            p.pointa .= boundpoint
            v_bar = candidate_midpoint .- boundpoint
            v_bar_norm = norm(v_bar, 2)
            p.uhat .= v_bar ./ v_bar_norm

            # if bound point and pointa bracket a boundary, search for the boundary
            # otherwise, the bound point is used as the level set boundary (i.e. it's inside the true level set boundary)
            g = bivariate_optimiser(0.0, p)
            if biv_opt_is_ellipse_analytical || g < 0.0
                # make bracket a tiny bit smaller
                
                lb = isinf(g) ? 1e-12 * v_bar_norm : 0.0

                Ψ = find_zero(bivariate_optimiser, (lb, v_bar_norm), Roots.Brent(); p=p)

                boundarypoint = p.pointa + Ψ*p.uhat
                boundary[:, num_vertices] .= boundarypoint
                boundary_all[[ind1, ind2], num_vertices] .= boundarypoint
            else
                point_is_on_bounds[num_vertices] = true
                boundary[:, num_vertices] .= boundpoint
                boundary_all[[ind1, ind2], num_vertices] .= boundpoint

                if bound_warning
                    @warn string("The ", upper_or_lower, " bound on variable ", model.core.θnames[bound_ind], " is inside the confidence boundary")
                    bound_warning = false
                end
            end

        else # external - push inwards
            
            candidate_line = Meshes.Line(Point(candidate_midpoint), Point(candidate_midpoint .+ dir_vector)) 
            # find edge that the line normal vector intersects 
            current_vertex = ve2 * 1
            while current_vertex != ve1
                edge_segment = Segment(Point(boundary[:,current_vertex]), Point(boundary[:, edge_anti[current_vertex]]))

                if intersection(candidate_line, edge_segment).type != IntersectionType(0)
                    break
                end
                current_vertex = edge_anti[current_vertex] * 1
            end

            # by construction/algorithm enforcement all polygons we search within must have at least three points so this is ok
            # don't want to choose an edge vertex that's on the candidate edge
            if edge_anti[current_vertex] == ve1 
                edge_vertex_index = 1
            elseif current_vertex == ve2
                edge_vertex_index = 2
            else
                edge_vertex_index = argmin(SA[edge_length(boundary, candidate_midpoint, current_vertex, relative_magnitude),
                    edge_length(boundary, candidate_midpoint, edge_anti[current_vertex], relative_magnitude)])
            end
            edge_vertex = edge_vertex_index == 1 ? current_vertex : edge_anti[current_vertex] * 1
            
            p.pointa .= boundary[:,edge_vertex] .* 1.0

            v_bar = candidate_midpoint .- p.pointa
            v_bar_norm = norm(v_bar, 2)
            p.uhat .= v_bar ./ v_bar_norm

            Ψ = solve(ZeroProblem(bivariate_optimiser, v_bar_norm), Roots.Order8(); p=p)

            boundarypoint = p.pointa + Ψ*p.uhat

            if isnan(Ψ) || isinf(Ψ) || isapprox(boundarypoint, p.pointa)
                # failure=true
                f(x) = bivariate_optimiser(x, p)
                Ψs = find_zeros(f, 0.0, v_bar_norm; p=p)
                if length(Ψs) == 0
                    failure=true
                elseif length(Ψs) == 1
                    boundarypoint = p.pointa + Ψs[1]*p.uhat
                    if isapprox(boundarypoint, p.pointa)
                        failure=true
                    end
                else
                    boundarypoint = p.pointa + Ψs[end]*p.uhat
                end
            end
                
            if failure
                return num_vertices, internal_count, failure, bound_warning, edge_vertex
            end

            boundary[:, num_vertices] .= boundarypoint
            boundary_all[[ind1, ind2], num_vertices] .= boundarypoint
            bivariate_optimiser(Ψ, p)
        end

        if !biv_opt_is_ellipse_analytical
            variablemapping2d!(@view(boundary_all[:, num_vertices]), p.λ_opt, p.θranges, p.λranges)
        end
    end

    return num_vertices, internal_count, failure, bound_warning, 0
end

function heapupdates_success!(edge_heap::TrackingHeap,
                        angle_heap::TrackingHeap, 
                        edge_clock::Vector{Int},
                        edge_anti::Vector{Int},
                        point_is_on_bounds::BitVector,
                        edge_anti_on_bounds::BitVector,
                        boundary::Matrix{Float64},
                        num_vertices::Int,
                        vi::Int, 
                        adj_vertex::Int,
                        relative_magnitude::Float64,
                        clockwise_from_vi=false)

    # perform required updates
    if clockwise_from_vi
        # adjacent vertex is clockwise from vi
        edge_clock[vi] = num_vertices
        edge_clock[num_vertices] = adj_vertex

        edge_anti[adj_vertex] = num_vertices
        edge_anti[num_vertices] = vi

        if point_is_on_bounds[num_vertices]
            edge_anti_on_bounds[num_vertices] = point_is_on_bounds[num_vertices] && point_is_on_bounds[vi]
            edge_anti_on_bounds[adj_vertex] = point_is_on_bounds[num_vertices] && point_is_on_bounds[adj_vertex]
        end

        # update edge length for adj_vertex and num_vertices
        for i in SA[adj_vertex, num_vertices]
            TrackingHeaps.update!(edge_heap, i, edge_length(boundary, i, edge_anti[i], relative_magnitude))
        end
    else
        # adjacent vertex is anticlockwise from vi
        edge_clock[adj_vertex] = num_vertices
        edge_clock[num_vertices] = vi

        edge_anti[vi] = num_vertices
        edge_anti[num_vertices] = adj_vertex

        if point_is_on_bounds[num_vertices]
            edge_anti_on_bounds[vi] = point_is_on_bounds[num_vertices] && point_is_on_bounds[vi]
            edge_anti_on_bounds[num_vertices] = point_is_on_bounds[num_vertices] && point_is_on_bounds[adj_vertex]
        end
        
        # update edge length for vi and num_vertices 
        for i in SA[vi, num_vertices]
            TrackingHeaps.update!(edge_heap, i, edge_length(boundary, i, edge_anti[i], relative_magnitude))
        end
    end

    # update angle obj for vi, adj_vertex and new vertex (num_vertices)
    for i in SA[vi, adj_vertex, num_vertices]
        if point_is_on_bounds[i]
            TrackingHeaps.update!(angle_heap, i, 0.0)
        else
            TrackingHeaps.update!(angle_heap, i, internal_angle_from_pi(i, boundary, edge_clock, edge_anti, relative_magnitude))
        end
    end

    return nothing
end

function polygon_break_and_rejoin!(edge_clock::Vector{Int},
                                    edge_anti::Vector{Int},
                                    ve1::Int,
                                    ve2::Int,
                                    opposite_edge_ve1::Int,
                                    opposite_edge_ve2::Int,
                                    model::LikelihoodModel,
                                    ind1::Int, 
                                    ind2::Int)

    edge_clock[ve2] = opposite_edge_ve1 * 1
    edge_clock[opposite_edge_ve2] = ve1 * 1

    # ve1 -> edge_anti[opposite_edge]
    # opposite_edge -> ve2
    edge_anti[ve1] = opposite_edge_ve2 * 1
    edge_anti[opposite_edge_ve1] = ve2 * 1

    # In the case we have a polygon with only one vertex
    if opposite_edge_ve2 == ve1 || ve2 == opposite_edge_ve1
        # 1==1
        @info string("there is likely to be multiple distinct level sets at this confidence level for parameters ", model.core.θnames[ind1]," and ", model.core.θnames[ind2], ". No additional points can be found on one of these level sets within this algorithm run.")
    end
    return nothing
end

"""
This function is used in the event that no boundary points are found using [`newboundarypoint`](@ref). Failure means it is likely that multiple level sets exist. If so, we break the edges of the candidate point and `e_intersect` and reconnect the vertexes such that we now have multiple boundary polygons.
		
If we only have one or two points on one of these boundary polygons we will display an info message as no additional points can be found from the method directly.
		
If we have three or more points on these boundary polygons, then there should be no problems finding other parts of these polygons.

If the largest polygon has less than two points the method will display a warning message and terminate, returning the boundary found up until then. 
"""
function heapupdates_failure!(edge_heap::TrackingHeap,
                                angle_heap::TrackingHeap, 
                                edge_clock::Vector{Int},
                                edge_anti::Vector{Int},
                                point_is_on_bounds::BitVector,
                                boundary::Matrix{Float64},
                                num_vertices::Int,
                                ve1::Int,
                                ve2::Int,
                                opposite_edge_ve1::Int,
                                model::LikelihoodModel,
                                ind1::Int, 
                                ind2::Int,
                                relative_magnitude::Float64)

    opposite_edge_ve2 = edge_anti[opposite_edge_ve1] * 1
    polygon_break_and_rejoin!(edge_clock, edge_anti, ve1, ve2, opposite_edge_ve1, opposite_edge_ve2, model, ind1, ind2)

    # In the case we have a new polygon with two vertices, simplest way to handle is to break this two vertex polygon into two 1 vertex polygons, so long as we have another polygon with at least 3 vertices. 
    if ve1 == edge_anti[opposite_edge_ve2]
        polygon_break_and_rejoin!(edge_clock, edge_anti, ve1, opposite_edge_ve2, opposite_edge_ve2, ve1, model, ind1, ind2)
    end
    if opposite_edge_ve1 == edge_anti[ve2]
        polygon_break_and_rejoin!(edge_clock, edge_anti, ve2, opposite_edge_ve1, opposite_edge_ve1, ve2, model, ind1, ind2)
    end

    # update edge length for ve1 and opposite_edge_ve1
    for i in SA[ve1, opposite_edge_ve1]
        TrackingHeaps.update!(edge_heap, i, edge_length(boundary, i, edge_anti[i], relative_magnitude))
    end

    # update angle obj for ve1, ve2, and opposite edge vertices
    for i in SA[ve1, ve2, opposite_edge_ve1, opposite_edge_ve2]
        if point_is_on_bounds[i]
            TrackingHeaps.update!(angle_heap, i, 0.0)
        else
            TrackingHeaps.update!(angle_heap, i, internal_angle_from_pi(i, boundary, edge_clock, edge_anti, relative_magnitude))
        end
    end

    if TrackingHeaps.top(edge_heap)[2] ≤ 0.0
        @warn(string("the number of vertices on the largest level set polygon is less than three at the current step. Algorithm aborting."))

        if save_internal_points
            ll_values = ll_values[1:internal_count] .+ mle_targetll
            if biv_opt_is_ellipse_analytical
                internal_all = get_λs_bivariate_ellipse_analytical!(internal_all[[ind1, ind2],1:internal_count], internal_count,
                                        p.consistent, ind1, ind2, 
                                        model.core.num_pars, p.initGuess,
                                        p.θranges, p.λranges)
            else
                internal_all = internal_all[:, 1:internal_count]
            end
        end

        return true, (boundary_all[:, 1:num_vertices], PointsAndLogLikelihood(internal_all, ll_values))
    end

    return false, nothing
end

function bivariate_confidenceprofile_iterativeboundary(bivariate_optimiser::Function, 
                                                model::LikelihoodModel, 
                                                num_points::Int, 
                                                consistent::NamedTuple, 
                                                ind1::Int, 
                                                ind2::Int,
                                                initial_num_points::Int,
                                                angle_points_per_iter::Int,
                                                edge_points_per_iter::Int,
                                                radial_start_point_shift::Float64,
                                                ellipse_sqrt_distortion::Float64,
                                                use_ellipse::Bool,
                                                mle_targetll::Float64,
                                                save_internal_points::Bool)

    num_points ≥ initial_num_points || throw(ArgumentError("num_points must be greater than or equal to initial_num_points"))
    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateΨ_ellipse_analytical

    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]
    if biv_opt_is_ellipse_analytical
        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                    θranges=θranges, λranges=λranges, consistent=consistent)
    else
        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                    θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-2))
    end

    return_tuple = iterativeboundary_init(bivariate_optimiser, model, num_points, p, ind1, ind2,
                                            initial_num_points, biv_opt_is_ellipse_analytical,
                                            radial_start_point_shift, ellipse_sqrt_distortion,
                                            use_ellipse, save_internal_points)

    if return_tuple[1]
        return return_tuple[2], return_tuple[3]
    end

    _, boundary, boundary_all, internal_all, ll_values, internal_count, point_is_on_bounds, edge_anti_on_bounds, bound_warning, mle_point, num_vertices, edge_clock, edge_anti, edge_heap, angle_heap, relative_magnitude = return_tuple

    while num_vertices < num_points

        iter_max = min(num_points, num_vertices+angle_points_per_iter)
        while num_vertices < iter_max
            num_vertices += 1

            # candidate vertex
            candidate = TrackingHeaps.top(angle_heap)
            vi = candidate[1] * 1
            adjacents = SA[edge_clock[vi], edge_anti[vi]]
            adjacent_index = argmax(getindex.(Ref(angle_heap), adjacents))
            adj_vertex = adjacents[adjacent_index] * 1 # choose adjacent vertex with the biggest obj angle as candidate edge
            ve1 = adjacent_index==1 ? adj_vertex : vi
            ve2 = edge_anti[ve1] * 1

            num_vertices, internal_count, failure, bound_warning, opposite_edge_ve1 = newboundarypoint!(p, point_is_on_bounds, edge_anti_on_bounds, 
                                                                    boundary, boundary_all,
                                                                    internal_all,
                                                                    ll_values,
                                                                    internal_count,
                                                                    bivariate_optimiser, 
                                                                    model, edge_anti, num_vertices, ind1, ind2, 
                                                                    biv_opt_is_ellipse_analytical, 
                                                                    ve1, ve2, relative_magnitude,
                                                                    bound_warning,
                                                                    save_internal_points)

            if failure # appears we have found two distinct level sets - break the edges and join to form two separate polygons
                num_vertices -= 1
                
                termination, return_args = heapupdates_failure!(edge_heap, angle_heap, edge_clock, edge_anti, point_is_on_bounds,
                                                                boundary, num_vertices, ve1, ve2, opposite_edge_ve1, model, ind1, ind2,
                                                                relative_magnitude)
                if termination; return return_args end
                continue
            end

            heapupdates_success!(edge_heap, angle_heap, edge_clock, edge_anti, point_is_on_bounds, edge_anti_on_bounds,
                            boundary, num_vertices, vi, adj_vertex, relative_magnitude, adjacent_index == 1)
            
        end
        if num_vertices == num_points; break end

        iter_max = min(num_points, num_vertices+edge_points_per_iter)
        while num_vertices < iter_max
            num_vertices += 1
            
            # candidate edge
            candidate = TrackingHeaps.top(edge_heap)
            vi = candidate[1] * 1
            adj_vertex = edge_anti[vi] * 1

            num_vertices, internal_count, failure, bound_warning, opposite_edge_ve1 = newboundarypoint!(p, point_is_on_bounds, edge_anti_on_bounds, 
                                                                    boundary, boundary_all, 
                                                                    internal_all,
                                                                    ll_values,
                                                                    internal_count,bivariate_optimiser, 
                                                                    model, edge_anti, num_vertices, ind1, ind2, 
                                                                    biv_opt_is_ellipse_analytical, 
                                                                    vi, adj_vertex, relative_magnitude,
                                                                    bound_warning,
                                                                    save_internal_points)
            
            if failure # appears we have found two distinct level sets - break the edges and join to form two separate polygons
                ve1 = vi
                ve2 = adj_vertex
                num_vertices -= 1
                
                termination, return_args = heapupdates_failure!(edge_heap, angle_heap, edge_clock, edge_anti, point_is_on_bounds,
                                                                boundary, num_vertices, ve1, ve2, opposite_edge_ve1, model, ind1, ind2,
                                                                relative_magnitude)
                if termination; return return_args end
                continue
            end
                        
            heapupdates_success!(edge_heap, angle_heap, edge_clock, edge_anti, point_is_on_bounds, edge_anti_on_bounds,
                            boundary, num_vertices, vi, adj_vertex, relative_magnitude)
        end
    end

    if biv_opt_is_ellipse_analytical
        return get_λs_bivariate_ellipse_analytical!(@view(boundary[[ind1, ind2], :]), num_points,
                                                    consistent, ind1, ind2, 
                                                    model.core.num_pars, initGuess,
                                                    θranges, λranges, boundary), PointsAndLogLikelihood(internal_all, ll_values)
    end

    if save_internal_points
        ll_values = ll_values[1:internal_count] .+ mle_targetll
        if biv_opt_is_ellipse_analytical
            internal_all = get_λs_bivariate_ellipse_analytical!(internal_all[[ind1, ind2],1:internal_count], internal_count,
                                    p.consistent, ind1, ind2, 
                                    model.core.num_pars, p.initGuess,
                                    p.θranges, p.λranges)
        else
            internal_all = internal_all[:, 1:internal_count]
        end
    end

    return boundary_all, PointsAndLogLikelihood(internal_all, ll_values)
end


# polygon = [0 3 2 2 -1; 0 1 2 4 2]*1.0

# edges = [1 2 3 4 5; 2 3 4 5 1]

# edge_vectors = polygon[:, edges[2,:]] .- polygon[:, edges[1,:]]


# @time angles = rad2deg.([AngleBetweenVectors.angle(edge_vectors[:, i], edge_vectors[:,j]) for (i,j) in [(1,2),(2,3),(3,4),(4,5),(5,1)]])

# @time rad2deg.([(abs(∠(Point(polygon[:,i]), Point(polygon[:,i+1]), Point(polygon[:,i+2])))) for i in 1:3])

# @time ∠(Point(polygon[:,1]), Point(polygon[:,2]), Point(polygon[:,3]))
# @time AngleBetweenVectors.angle(edge_vectors[:,1], edge_vectors[:,2])
# @time ∠(Vec(edge_vectors[:,1]), Vec(edge_vectors[:,2]))