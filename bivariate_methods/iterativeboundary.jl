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

        if !(bivariate_optimiser(0.0, p) > 0)
            point_is_on_bounds[i] = true

            if bound_warning
                @warn string("The ", upper_or_lower, " bound on variable ", model.core.θnames[bound_ind], " is inside the confidence boundary")
                bound_warning = false
            end
        end
    end

    return external, point_is_on_bounds
end

function bivariate_confidenceprofile_iterativeboundary(bivariate_optimiser::Function, 
                                                model::LikelihoodModel, 
                                                num_points::Int, 
                                                consistent::NamedTuple, 
                                                ind1::Int, 
                                                ind2::Int,
                                                initial_num_points::Int,

                                                mle_targetll::Float64,
                                                save_internal_points::Bool)

    num_points ≥ initial_num_points || throw(ArgumentError("num_points must be greater than or equal to initial_num_points"))
    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateΨ_ellipse_analytical

    boundary = zeros(2, num_points)
    boundary_all = zeros(model.core.num_pars, num_points)
    internal_all = zeros(model.core.num_pars, 0)
    ll_values = zeros(0)
    point_is_on_bounds = falses(num_points)
    # warn if bound prevents reaching boundary
    bound_warning=true

    if biv_opt_is_ellipse_analytical
        p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, Ψ_x=[0.0],
            θranges=θranges, λranges=λranges, consistent=consistent)
    else
        p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, Ψ_x=[0.0],
            θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-2))
    end

    external, point_is_on_bounds = findNpointpairs_radialMLE!(p, bivariate_optimiser, model, 
                                                            initial_num_points, ind1, ind2, bound_warning)

    mle_point = model.core.θmle[[ind1,ind2]]
    for i in 1:initial_num_points
        if point_is_on_bounds[i]
            p.pointa .= external[:,i]
            bivariate_optimiser(0, p)
            boundary[:,i] .= external[:,i]
            boundary[[ind1, ind2], i] .= external[:,i]
        else
            p.pointa .= mle_point .* 1.0
            v_bar = external[:,i] - mle_point

            v_bar_norm = norm(v_bar, 2)
            p.uhat .= v_bar / v_bar_norm

            Ψ_y1 = find_zero(bivariate_optimiser, (0.0, v_bar_norm), Roots.Brent(); p=p)
            
            boundary[:,i] .= p.pointa + Ψ_y1*p.uhat
            boundary[[ind1, ind2], i] .= boundary[:,i]
        end
        if !biv_opt_is_ellipse_analytical
            variablemapping2d!(@view(boundary_all[:, i]), p.λ_opt, θranges, λranges)
        end
    end

    if initial_num_points == num_points
        return boundary_all, PointsAndLogLikelihood(internal_all, ll_values)
    end 

    num_vertices = initial_num_points * 1

    # adjacent vertices of each vertex
    adjacent_vertices = zeros(Int, num_points, 2)
    adjacent_vertices[1,:] .= num_vertices, 2
    for i in 2:num_vertices-1; edges[i,:] .= i-1, i+1 end
    adjacent_vertices[num_vertices,:] .= num_vertices-1, 1

    # tracked heap for internal angles between adjacent edges 
    # WRITE CUSTOM STRUCTS FOR STORING EDGE INFORMATION BETWEEN VERTICES SO THAT IT IS EASY TO UPDATE CONNECTIONS
    # AND/OR FUNCTIONS THAT DO THIS
    vertex_internal_angle_objs = zeros(num_points)
    
    angle_heap = TrackingHeap(Float64, S=NoTrainingWheels, O=MaxHeapOrder, N = 2, init_val_coll=vertex_internal_angle_objs)

    # vertices edges are incident on
    edges = zeros(Int, num_points, 2)
    for i in 1:num_vertices-1; edges[i,:] .= i, i+1 end
    edges[num_vertices,:] .= num_vertices, 1
    
    # tracked heap for length of edges
    edge_lengths = zeros(num_points)
    edge_lengths[num_vertices+1:end] .= -Inf

    edge_lengths[1:num_vertices] .= colwise(Euclidean(), 
                                            @view(boundary[:, 1:num_vertices]), 
                                            @view(boundary[:, @view(edges[2,1:num_vertices])])) 

    edge_heap = TrackingHeap(Float64, S=NoTrainingWheels, O=MaxHeapOrder, N = 2, init_val_coll=edge_lengths)
    








    # for (i, j, N) in [[ind1, ind2, div(num_points,2)], [ind2, ind1, (div(num_points,2) + rem(num_points,2))]]

    #     


    #     x_vec, y_vec, internal, ll = findNpointpairs_fix1axis!(p, bivariate_optimiser, model,
    #                                                         N, i, j, mle_targetll, save_internal_points,
    #                                                         biv_opt_is_ellipse_analytical)
        
    #     for k in 1:N
    #         count +=1

    #         p.Ψ_x[1] = x_vec[k]

    #         Ψ_y1 = find_zero(bivariate_optimiser, (y_vec[1,k], y_vec[2,k]), Roots.Brent(); p=p)

    #         boundary[i, count] = x_vec[k]
    #         boundary[j, count] = Ψ_y1
            
    #         if !biv_opt_is_ellipse_analytical
    #             variablemapping2d!(@view(boundary[:, count]), p.λ_opt, θranges, λranges)
    #         end
    #     end

    #     if save_internal_points
    #         internal_all = hcat(internal_all, internal)
    #         ll_values = vcat(ll_values, ll) 
    #     end
    # end




    if biv_opt_is_ellipse_analytical
        return get_λs_bivariate_ellipse_analytical!(@view(boundary[[ind1, ind2], :]), num_points,
                                                    consistent, ind1, ind2, 
                                                    model.core.num_pars, initGuess,
                                                    θranges, λranges, boundary), PointsAndLogLikelihood(internal_all, ll_values)
    end

    return boundary, PointsAndLogLikelihood(internal_all, ll_values)
end