"""
    star_obj(centers, points)

centers have 2D points stored in columns.

points have 2D boundary points stored in columns and are in either clockwise or anticlockwise order of polygon connection.
"""
function star_obj(centers, points)
    n = size(points,2)
    obj = zeros(size(centers,2))
    for ci in axes(centers,2)
        c_point = centers[:,ci]
        for vi in 1:n
            intersects_polygon=false
            
            internal_segment = Segment(Point(c_point), Point(points[:,vi]))
            # println()
            # println("Vi=", vi)
            # println(internal_segment)

            # all vertex to vertex edges in the polygon that don't include vertex vi
            v1s = vcat(collect(1:vi-2), vcat(collect(vi+1:n-1), vi != 1 && vi != n ? [n] : Int[]))
            v2s = vcat(collect(2:vi-1), vcat(collect(vi+2:n), vi != 1 && vi != n ? [1] : Int[]))
            # println("v1s:", v1s)
            # println("v2s:", v2s)

            for ei in eachindex(v1s)
                edge_segment = Segment(Point(points[:,v1s[ei]]), Point(points[:,v2s[ei]])) 
                # println(edge_segment)
                # segment_plot(internal_segment, edge_segment, xlim=[-0.1,4.1], ylim=[-0.1, 6.1])
                if intersection(internal_segment, edge_segment).type != IntersectionType(0) 
                    intersects_polygon=true
                    continue
                end
            end
            if !intersects_polygon
                obj[ci] +=1
            end
        end
    end
    return obj
end

"""
    smooth_boundary!(points, point_is_on_bounds)

`points` has 2D boundary points stored in columns and are in either clockwise or anticlockwise order of polygon connection. `points` is edited in place.

Points that we know are on the provided bounds are not moved/smoothed, although their presence will impact what the smoother tries to do.

Smoother must attempt to ensure that smoothed points don't go outside the level set boundary???? - may need to evaluate ll for every smoothed point to guarantee this.
"""
function boundary_smoother!(points, point_is_on_bounds)
    mesh = SimpleMesh([(points[1,i], points[2,i]) for i in axes(points,2)], 
                        [connect(tuple(1:size(points,2)...))])
    mesh = mesh |> LaplaceSmoothing(30)
    # mesh = mesh |> TaubinSmoothing(300)

    for i in axes(points, 2)
        if !point_is_on_bounds[i]
            points[:,i] .= mesh.vertices[i].coords
        end
    end
    return nothing
end

"""
Use kmeans with say 9 clusters on either: the polygon boundary points or points sampled within the polygon boundary.

Define objectives for the cluster points: how close each point is to being a star point, e.g. by finding which cluster point can see the most boundary points (i.e. there isn’t a edge in the way) - this is a form of regularisation by discretisation; we’re assuming that the boundary defined by a polygon with straight lines between boundary points is relatively consistent with the true shape. Additionally, we assume that our boundary points are relatively well spaced out so that the objective isn’t biased by a ton of our known points being located in a specific portion of the boundary.

If we find a star point (or multiple) then we should use that point to push out radially (e.g. instead of the MLE point). Note: If boundary is convex, all points in our set are star points by definition. If concave if star point exists use that for continuation, else use kmeans points which are likely to be star points for their local sections

Second obj we can use to tie break is how close each cluster point is to the centre of the polygon boundary (e.g. just using a simple how many points of all points (either the boundary points OR points that are sampled in the boundary) are on either side in x and y axis of the cluster point (and how close these are to 50%).

If star point doesn't exist then... Presently will use each individual kmeans point as the point to push out from based on the cluster the boundary point belongs to. In future perhaps use point closest to being a star point for all vertices it is a star point for?? And then other points for the others???

Returns whether or not a star point was found (if not, cannot guarantee that the ordering of boundary points will stay the same, and require a TSP iteration after solving for the next level set).
"""
function refine_search_directions!(search_directions, points, point_is_on_bounds; k_means=9, sample_in_polygon=true, verbose=true)

    if sample_in_polygon
        mesh = SimpleMesh([(points[1,i], points[2,i]) for i in axes(points,2)], 
                            [connect(tuple(1:size(points,2)...))])
        internal_points = reduce(hcat,[point.coords for point in collect(sample(mesh, HomogeneousSampling(500)))])
        R = kmeans(internal_points, k_means)
    else
        internal_points=points
        R = kmeans(points, k_means)
    end

    obj = star_obj(R.centers, points)
    # println(obj)
    max_obj = maximum(obj)
    star_centers = R.centers[:, findall(max_obj .== obj)]

    point_not_on_bounds = .!point_is_on_bounds

    if max_obj == size(points,2)
        verbose && println("star point found")

        if size(star_centers, 2) == 1
            new_center = star_centers
        else
            new_center = star_centers[:, argmin([ sum(abs.( ([sum(internal_points[1,:] .< star_centers[1,i]) , sum(internal_points[2,:] .< star_centers[2,i]) ]./ size(internal_points,2)) .- 0.5))  for i in axes(star_centers, 2)])]
            # println(new_center)
        end
        search_directions[:, point_not_on_bounds] .= points[:, point_not_on_bounds] .- new_center

        return true
    else
        verbose && println("star point not found")

        if sample_in_polygon
            # for each boundary point, first find the cluster center it is closest to (Euclidean distance), assigning it to that cluster
            cluster_assignments = zeros(Int, size(points, 2))
            for i in axes(points, 2)
                if point_not_on_bounds[i]
                    cluster_assignments[i] = argmin(colwise(Euclidean(), R.centers, points[:,i]))
                end
            end
        else
            cluster_assignments = assignments(R)
        end

        for i in axes(points, 2)
            if point_not_on_bounds[i]
                search_directions[:,i] .= points[:,i] .- R.centers[:,cluster_assignments[i]]
            end
        end
        return false
    end
end