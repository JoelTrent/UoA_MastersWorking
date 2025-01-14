using EllipseSampling
using Plots; gr()
Plots.reset_defaults()
default(palette=:seaborn_colorblind6, msw=0, markeralpha=0.7, aspect_ratio=:equal, label=nothing, dpi=300, size=(500,400)./1.25)
# Plots.scalefontsizes()

output_location = joinpath("Bespoke graphics", "EllipseSampling_squashed")
function main()
    # # Equally Spaced Points
    # e=construct_ellipse(1.0, 0.5, 0.0, 2.0, 1.0)
    # points=generate_N_equally_spaced_points(9, e; start_point_shift=0.0) 
    # p = scatter(points[1,:], points[2,:])
    # savefig(p, joinpath(output_location, "equallyspaced1.pdf"))

    # points=generate_N_equally_spaced_points(9, 1.0, 0.5, 0.0, 2.0, 1.0; start_point_shift=0.0) 
    # p=scatter(points[1,:], points[2,:])
    # savefig(p, joinpath(output_location, "equallyspaced2.pdf"))

    # ## Rotated Ellipses
    # e=construct_ellipse(1.0, 0.5, pi/3.0, 2.0, 1.0)
    # points=generate_N_equally_spaced_points(9, e; start_point_shift=0.0) 
    # p=scatter(points[1,:], points[2,:])
    # savefig(p, joinpath(output_location, "equallyspaced3.pdf"))

    # Clustered Points
    e=construct_ellipse(1.0, 0.1, 0.0, 2.0, 1.0)
    points=generate_N_clustered_points(30, e; start_point_shift=0.0, sqrt_distortion=0.0) 
    p=scatter(points[1,:], points[2,:])
    savefig(p, joinpath(output_location, "clustered1.pdf"))

    p=plot()
    e=construct_ellipse(1.0, 0.1, 0.0, 2.0, 1.0)
    for sqrt_distortion in 0.0:0.2:1.0
        points=generate_N_clustered_points(10, e; start_point_shift=0.0, sqrt_distortion=sqrt_distortion) 
        scatter!(p, points[1,:], points[2,:], label=string("sqrt_distortion=",sqrt_distortion))
    end
    savefig(p, joinpath(output_location, "clustered2.pdf"))

    p=plot(palette=:Paired_6)
    e=construct_ellipse(1.0, 1.0, 0.0, 2.0, 1.0)
    for sqrt_distortion in 0.0:0.5:1.0
        points=generate_N_clustered_points(10, e; start_point_shift=0.0, sqrt_distortion=sqrt_distortion) 
        scatter!(p, points[1,:], points[2,:], label=string("sqrt_distortion=",sqrt_distortion),
                markersize=7-sqrt_distortion*4, markeralpha=0.8)
    end
    savefig(p, joinpath(output_location, "clustered3.pdf"))

    # # Custom Sampling Method
    # e=construct_ellipse(1.0, 0.5, 0.0, 2.0, 1.0)
    # N = 100
    # samples = rand(N)

    # # wrap e in Ref so that the function correctly broadcasts across samples
    # points = generate_perimeter_point.(samples, Ref(e)) 
    # points = reduce(hcat, points)

    # p=scatter(points[1,:], points[2,:])
    # savefig(p, joinpath(output_location, "customsampling.pdf"))
end

main()