using PolygonInbounds
using DataStructures

function get_segment_distance_squared(px, py, a, b)
    x, y = a
    dx = b[1]-x
    dy = b[2]-y

    if dx !== 0 || dy !== 0
        t = ((px - x)*dx + (py - y)*dy) / (dx*dx + dy*dy)

        if t > 1
            x = b[1]
            y = b[2]
        elseif t > 0 
            x += dx * t
            y += dy * t
        end
    end

    dx = px-x
    dy = py-y
    return dx^2 + dy^2
end

function point_to_polygon_distance(x, y, node, edge)

    distance = Inf
    is_inside, is_boundary = inpoly2([x y], node, edge)
    is_boundary && return 0.0

    for i in axes(edge,1)
        distance = min(distance, get_segment_distance_squared(x, y, node[edge[i,1],:], node[edge[i,2],:]))
    end

    is_inside && return distance
    return -distance
end

struct Cell
    x::Real
    y::Real
    h::Real
    d::Real
    max::Real

    function Cell(x,y,h,node,edge)
        d = point_to_polygon_distance(x,y,node,edge)
        m = d + h * sqrt(2.0)
        return new(x,y,h,d,m)
    end
end

function get_centroid_cell(node, edge)
    area = 0.0
    x = 0.0
    y = 0.0

    for i in 1:(size(node, 1)-1)
        a = node[i,:]
        b = node[i+1,:]
        f = a[1]*b[2] - b[1]*a[2]
        x += (a[1] + b[1]) * f
        y += (a[2] + b[2]) * f
        area += f * 3
    end

    if area == 0.0; return Cell(node[1,1], node[1,2], 0, node, edge) end
    return Cell(x/area, y/area, 0, node, edge)
end

"""
    polylabel(node, edge, precision=1.0)

node is a n×2 array of polygon vertices - ideally ordered
edge is a n×2 array of edges between nodes that make up the polygon
"""
function polylabel(node, edge, precision=1.0; debug=false)

    xmin, xmax = extrema(node[:,1])
    ymin, ymax = extrema(node[:,2])

    width = xmax - xmin
    height = ymax - ymin
    cell_size = min(width, height)
    h = cell_size / 2.0

    if cell_size == 0.0; return [xmin, ymin], 0 end

    pq = PriorityQueue{Cell,Real}()
    x = xmin * 1.0
    while x < xmax
        y = ymin * 1.0
        while y < ymax
            new_cell = Cell(x+h, y+h, h, node, edge)
            enqueue!(pq, new_cell, new_cell.max)
            y += cell_size
        end
        x += cell_size
    end

    # take centroid as the first best guess
    best_cell = get_centroid_cell(node, edge)

    # second guess: bounding box centroid
    bbox_cell = Cell(xmin + width/2.0, ymin + height/2.0, 0, node, edge)
    if bbox_cell.d > best_cell.d; best_cell = bbox_cell end 

    num_probes = length(pq)

    while length(pq) > 0
        # pick the most promising cell from the queue
        cell = dequeue!(pq)

        # update the best cell if we found a better one
        if cell.d > best_cell.d
            best_cell = cell
            debug && println(string("found best ", round(cell.d; sigdigits=4), " after ", num_probes, " probes"))
        end

        # do not drill down further if there's no chance of a better solution
        if (cell.max - best_cell.d) <= precision; continue end

        # split the cell into four cells
        h = cell.h / 2.0;

        for x in (cell.x-h, cell.x+h), y in (cell.y-h, cell.y+h)
            new_cell = Cell(x, y, h, node, edge)
            enqueue!(pq, new_cell, new_cell.max)
        end
        num_probes += 4;
    end

    if debug
        println("num probes: ",  num_probes);
        println("best distance: ", best_cell.d);
    end

    return [best_cell.x, best_cell.y], best_cell.d
end

nodes =  [0.0 0; 0 10; 10 10; 10 0]
edges = [1 2; 2 3; 3 4; 4 1]
# n = 4
# edges = zeros(Int, n, 2)
# for i in 1:n-1; edges[i,:] .= i, i+1 end
# edges[end,:] .= n, 1
polylabel(nodes, edges)




# points = [0.05 0.0; 1 1; -1 1]

# tol = 1e-1

# function poly2_inside(points, nodes, edges, atol)
#     return inpoly2(points, nodes, edges, atol=tol)
# end

# @btime poly2_inside(points, nodes, edges, atol)

# ins, bnd = inpoly2(points[:,1], nodes, edges, atol=tol)

# using Meshes

# mesh = SimpleMesh([(nodes[i,1], nodes[i,2]) for i in 1:n], [connect(tuple(1:n...))])

# function mesh_inside(points, mesh)
#     inside = falses(3)
#     for i in 1:3
#         inside[i] = Point(points[i,:]) in mesh
#     end
#     return inside
# end
    
# @btime mesh_inside(points, mesh)