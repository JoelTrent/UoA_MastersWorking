using EllipseSampling
using Luxor
using Colors

function logo(name::String, grey_outlines::Bool)
    Drawing(500, 500, name)
    origin()

    colors = (Luxor.julia_green, Luxor.julia_red, Luxor.julia_purple, Luxor.julia_blue)
    grey = "grey90"

    x_radius, y_radius = 275.0, 275/2
    cx, cy = 0.0, 0.0
    N=7

    points = generate_N_equally_spaced_points(N, construct_ellipse(x_radius, y_radius, deg2rad(45)); start_point_shift=0.5)

    colors = colors[[1,3,2,4]]
    points=points[:, [5,6,7,1,2,3,4]]

    if grey_outlines
        gsave()
        rotate(deg2rad(45))
        setcolor(grey)
        setline(18)
        ellipse(cx, cy, x_radius*2, y_radius*2, action=:stroke)
        grestore()

        for i in 1:N
            ind2 = i==N ? 1 : i+1

            setline(10)
            setcolor(grey)
            poly([Point(points[:,i]...), Point(cx,cy)], action=:stroke)
        end
    end

    for i in 1:N
        ind2 = i==N ? 1 : i+1
        gsave()

        line_point = perpendicular(Point(points[:,i]...),  Point(points[:,ind2]...), Point(cx, cy))
        reflection = line_point + (line_point - Point(cx,cy))
        poly([Point(points[:,i]...) + 0.5*(Point(points[:,i]...) - Point(cx,cy)), reflection, Point(points[:,ind2]...) + 0.5*(Point(points[:,ind2]...) - Point(cx,cy)), Point(cx, cy)], action=:clip)

        rotate(deg2rad(45))
        setcolor(colors[mod1(ind2, end)])
        setline(16)
        ellipse(cx, cy, x_radius*2, y_radius*2, action=:stroke)
        clipreset()
        grestore()

        setline(8)
        setcolor(colors[mod1(i, end)])
        poly([Point(points[:,i]...), Point(cx,cy)], action=:stroke)
    end

    if grey_outlines
        setcolor(grey)
        circle(O, 22, action=:fill)
        for i in 1:N
            setcolor(grey)
            circle.(Point(points[:,i]...), 34, action = :fill)
        end
    end

    setcolor(colors[1])
    circle(O, 20, action=:fill)
    for i in 1:N
        setcolor(colors[mod1(i, end)])
        circle.(Point(points[:,i]...), 32, action = :fill)
    end

    finish()
    return preview()
end

logo("logo.png", false)
logo("logo.svg", false)
# logo("logo-dark.png", false)
# logo("logo-dark.svg", false)
