using UnivariateUnimodalHighestDensityRegion
using Luxor
using Colors

function logo(name::String, grey_outlines::Bool)
    Drawing(500, 500, name)
    origin()

    colors = (Luxor.julia_green, Luxor.julia_red, Luxor.julia_purple, Luxor.julia_blue)
    grey = "grey95"

    # x_radius, y_radius = 275.0, 275/2
    # cx, cy = 0.0, 0.0
    # N=7
    N=5000
    xmap(x) = (x * 500) - 250
    ymap(y) = -1 * ((y*490)-250)
        
    
    d = Beta(2.5, 5.0)
    interval = round.(Int, xmap.(univariate_unimodal_HDR(d, 0.9)))
    
    x = LinRange(0, 1, N)
    y = pdf.(d, x)
    x = xmap.(x)
    y = ymap.(y ./maximum(y))

    points = [Point(x[i], y[i]) for i in 1:N]

    println(interval)

    setline(16)

    poly([Point(interval[1], -250), Point(interval[2], -250), Point(interval[2], 250), Point(interval[1], 250)], action=:clip)
    # setcolor(colors[1])
    blend_interval = blend(Point(interval[1]+50, 150), Point(interval[2]-50, -0), colors[3], colors[1])
    setblend(blend_interval)
    # poly(points[[1,end]], action=:stroke)
    poly(points[[collect(1:end)..., 1]], action=:fill)
    clipreset()


    # gsave()
    # poly([Point(interval[1], -250), Point(interval[2], -250), Point(interval[2], 250), Point(interval[1], 250)], action=:clip)
    # # poly(points[[1,end]], action=:stroke)
    # poly(points[[collect(1:end)..., 1]], action=:clip)
    
    # rotate(deg2rad(45))
    # # (interval[2] - interval[1]) / -2

    # sethue(colors[4])
    # for i in 0:100:500
    #     # rect(Point(-250, 240-i), 470, 25, action=:fill)
    # end

    # grestore()
    # clipreset()


    blend_outer_left = blend(Point(interval[1]-10, 240), Point(-240, 224), colors[4], colors[2])
    # setblend(blend_outer_left)
    
    setcolor(colors[2])
    poly([Point(interval[1]+1, -250), Point(-250, -250), Point(-250, 250), Point(interval[1]+1, 250)], action=:clip)
    poly(points[[collect(1:end)..., 1]], action=:fill)
    clipreset()

    blend_outer_right = blend(Point(interval[2] + 10, 240), Point(83, 200), colors[4], colors[2])
    # setblend(blend_outer_right)

    poly([Point(interval[2]-1, -250), Point(250, -250), Point(250, 250), Point(interval[2]-1, 250)], action=:clip)
    poly(points[[collect(1:end)..., 1]], action=:fill)
    clipreset()


    poly(points, action=:clip)
    setcolor(grey)
    # setcolor(colors[2])
    for i in 1:2
        poly([Point(interval[i], 0), Point(interval[i], 250.1)], action=:stroke)
    end
    clipreset()


    setcolor(colors[3])
    setcolor(grey)
    poly(points, action=:stroke)
    # setline(16)
    # setcolor(grey)
    # poly(points, action=:stroke)


    finish()
    return preview()
end

logo("logo.png", false)
logo("logo.svg", false)
# logo("logo-dark.png", false)
# logo("logo-dark.svg", false)
