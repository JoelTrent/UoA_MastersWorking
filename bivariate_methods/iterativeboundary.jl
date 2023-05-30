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
                                    bound_warning::Bool)

    external = zeros(2,num_points)
    point_is_on_bounds = falses(num_points)
    
    if isnan(model.core.θmagnitudes[ind1]) || isnan(model.core.θmagnitudes[ind2]) 
        relative_magnitude = 1.0
    else
        relative_magnitude = model.core.θmagnitudes[ind1]/model.core.θmagnitudes[ind2]
    end

    radial_dirs = find_m_spaced_radialdirections(num_points)

    x,y = model.core.θmle[[ind1,ind2]] .* 1.0
    for i in 1:num_points
        dir_vector = [relative_magnitude * cospi(radial_dirs[i]), sinpi(radial_dirs[i]) ]
        external[:,i], bound_ind, upper_or_lower = findpointonbounds(model, [x,y], dir_vector, ind1, ind2, true)

        # if bound point is a point inside the boundary, note that this is the case
        p.pointa .= external[:,i]

        if bivariate_optimiser(0.0, p) > 0
            point_is_on_bounds[i] = true

            if bound_warning
                @warn string("The ", upper_or_lower, " bound on variable ", model.core.θnames[bound_ind], " is inside the confidence boundary")
                bound_warning = false
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
                                biv_opt_is_ellipse_analytical::Bool)

    boundary = zeros(2, num_points)
    boundary_all = zeros(model.core.num_pars, num_points)
    internal_all = zeros(model.core.num_pars, 0)
    ll_values = zeros(0)
    point_is_on_bounds = falses(num_points)
    # warn if bound prevents reaching boundary
    bound_warning=true

    external, point_is_on_bounds_external, bound_warning = findNpointpairs_radialMLE!(p, bivariate_optimiser, model, 
                                                            initial_num_points, ind1, ind2, bound_warning)

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

            Ψ_y1 = find_zero(bivariate_optimiser, (0.0, v_bar_norm), Roots.Brent(); p=p)
            
            boundary[:,i] .= p.pointa + Ψ_y1*p.uhat
            boundary_all[[ind1, ind2], i] .= boundary[:,i]
        end
        if !biv_opt_is_ellipse_analytical
            variablemapping2d!(@view(boundary_all[:, i]), p.λ_opt, p.θranges, p.λranges)
        end
    end

    if initial_num_points == num_points
        return true, boundary_all, PointsAndLogLikelihood(internal_all, ll_values)
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

    # println(vertex_internal_angle_objs)
  
    angle_heap = TrackingHeap(Float64, S=NoTrainingWheels, O=MaxHeapOrder, N = 2, init_val_coll=vertex_internal_angle_objs)


    return false, boundary, boundary_all, internal_all, ll_values, point_is_on_bounds, edge_anti_on_bounds, bound_warning, mle_point, num_vertices, edge_clock, edge_anti, edge_heap, angle_heap, relative_magnitude
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

    return_tuple = iterativeboundary_init(bivariate_optimiser, model, num_points, p, ind1, ind2, initial_num_points, biv_opt_is_ellipse_analytical)

    if return_tuple[1]
        return return_tuple[2], return_tuple[3]
    end

    _, boundary, boundary_all, internal_all, ll_values, point_is_on_bounds, edge_anti_on_bounds, bound_warning, mle_point, num_vertices, edge_clock, edge_anti, edge_heap, angle_heap, relative_magnitude = return_tuple
   

    while num_vertices < num_points

        iter_max = min(num_points, num_vertices+angle_points_per_iter)
        while num_vertices < iter_max
            num_vertices += 1

            # candidate vertex
            candidate = TrackingHeaps.top(angle_heap)
            vi = candidate[1]
            adjacents = SA[edge_clock[vi], edge_anti[vi]]
            adjacent_index = argmax(getindex.(Ref(angle_heap), adjacents))
            adj_vertex = adjacents[adjacent_index] # choose adjacent vertex with the biggest obj angle as candidate edge
            ve1 = adjacent_index==1 ? adj_vertex : vi
            ve2 = edge_anti[ve1]

            # candidate point - midpoint of edge calculation
            candidate_midpoint = boundary[:, vi] .+ 0.5 .* (boundary[:, adj_vertex] - boundary[:, vi])

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

                # num_vertices += 1
                p.pointa .= candidate_midpoint
                dir_vector = SA[(boundary[2,ve2] - boundary[2,ve1]), -(boundary[1,ve2] - boundary[1,ve1])] .* SA[relative_magnitude, 1.0]
                g = bivariate_optimiser(0.0, p)
                if g > 0 # internal - push out normal to edge
                    p.uhat .= dir_vector ./ norm(dir_vector, 2)

                    boundpoint, bound_ind, upper_or_lower = findpointonbounds(model, candidate_midpoint, p.uhat, ind1, ind2, true)

                    v_bar_norm = (boundpoint[1] - p.pointa[1]) / p.uhat[1]

                    # if bound point and pointa bracket a boundary, search for the boundary
                    # otherwise, the bound point is used as the level set boundary (i.e. it's inside the true level set boundary)
                    if biv_opt_is_ellipse_analytical || bivariate_optimiser(v_bar_norm, p) < 0

                        Ψ_y1 = find_zero(bivariate_optimiser, (0.0, v_bar_norm), Roots.Brent(); p=p)

                        boundarypoint = p.pointa + Ψ_y1*p.uhat
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
                    println("ahhh")
                    p.uhat .= -dir_vector ./ norm(dir_vector, 2)
                    # num_vertices-=1
                end

                if !biv_opt_is_ellipse_analytical
                    variablemapping2d!(@view(boundary_all[:, num_vertices]), p.λ_opt, p.θranges, p.λranges)
                end
            end

            # perform required updates
            if adjacent_index == 1  
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
                TrackingHeaps.update!(angle_heap, i, internal_angle_from_pi(i, boundary, edge_clock, edge_anti, relative_magnitude))
            end
        end
        if num_vertices == num_points; break end

        iter_max = min(num_points, num_vertices+edge_points_per_iter)
        while num_vertices < iter_max
            
            # candidate edge
            candidate = TrackingHeaps.top(edge_heap)
            vi = candidate[1]
            adj_vertex = edge_anti[vi]
            
            # candidate point - midpoint of edge calculation
            candidate_midpoint = boundary[:, vi] .+ 0.5 .* (boundary[:, adj_vertex] - boundary[:, vi])
            
            # find new boundary point, given candidate midpoint, the vertexes that describe the edge and a set of valid cluster points to potentially search towards. 
            # use candidate point to find new vertex
            
            # if edge_anti_on_bounds[vi] # just accept the mid point and find nuisance parameters (if needed)
            
            boundary[:,num_vertices+1] .= [1.0,2.] # findpoint(candidate_midpoint)
            
            # perform required updates
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
            
            # update angle obj for vi, adj_vertex and new vertex (num_vertices)
            for i in SA[vi, adj_vertex, num_vertices]
                TrackingHeaps.update!(angle_heap, i, internal_angle_from_pi(i, boundary, edge_clock, edge_anti, relative_magnitude))
            end
            num_vertices += 1
        end
        if num_vertices == num_points; break end
    end


    if biv_opt_is_ellipse_analytical
        return get_λs_bivariate_ellipse_analytical!(@view(boundary[[ind1, ind2], :]), num_points,
                                                    consistent, ind1, ind2, 
                                                    model.core.num_pars, initGuess,
                                                    θranges, λranges, boundary), PointsAndLogLikelihood(internal_all, ll_values)
    end


    return boundary_all, PointsAndLogLikelihood(internal_all, ll_values)
end


polygon = [0 3 2 2 -1; 0 1 2 4 2]*1.0

edges = [1 2 3 4 5; 2 3 4 5 1]

edge_vectors = polygon[:, edges[2,:]] .- polygon[:, edges[1,:]]


@time angles = rad2deg.([AngleBetweenVectors.angle(edge_vectors[:, i], edge_vectors[:,j]) for (i,j) in [(1,2),(2,3),(3,4),(4,5),(5,1)]])

@time rad2deg.([(abs(∠(Point(polygon[:,i]), Point(polygon[:,i+1]), Point(polygon[:,i+2])))) for i in 1:3])

@time ∠(Point(polygon[:,1]), Point(polygon[:,2]), Point(polygon[:,3]))
@time AngleBetweenVectors.angle(edge_vectors[:,1], edge_vectors[:,2])
@time ∠(Vec(edge_vectors[:,1]), Vec(edge_vectors[:,2]))