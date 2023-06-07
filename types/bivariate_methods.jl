

"""
    AbstractBivariateMethod

Supertype for bivariate boundary finding methods. Use `bivariate_methods()` for a list of available methods (see [`bivariate_methods`](@ref)).

# Subtypes

[`AbstractBivariateVectorMethod`](@ref)

[`Fix1AxisMethod`](@ref)

[`ContinuationMethod`](@ref)

[`AnalyticalEllipseMethod`](@ref)
"""
abstract type AbstractBivariateMethod end

"""
    AbstractBivariateVectorMethod <: AbstractBivariateMethod

Supertype for bivariate boundary finding methods that search between two distinct points. 

# Subtypes

[`IterativeBoundaryMethod`](@ref)

[`RadialMLEMethod`](@ref)

[`RadialRandomMethod`](@ref)

[`SimultaneousMethod`](@ref)

# Supertype Hiearachy

AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any
"""
abstract type AbstractBivariateVectorMethod <: AbstractBivariateMethod end

struct IterativeBoundaryMethod <: AbstractBivariateVectorMethod
    initial_num_points::Int
    angle_points_per_iter::Int
    edge_points_per_iter::Int
    radial_start_point_shift::Float64
    ellipse_sqrt_distortion::Float64
    use_ellipse::Bool
    function IterativeBoundaryMethod(w, x, y, z=rand(), a=1.0; use_ellipse::Bool=false)
        w > 0 || throw(DomainError("initial_num_points must be greater than zero"))
        x ≥ 0 || throw(DomainError("angle_points_per_iter must be greater than or equal to zero"))
        y ≥ 0 || throw(DomainError("edge_points_per_iter must be greater than or equal zero"))
        x > 0 || y > 0 || throw(DomainError("at least one of angle_points_per_iter and edge_points_per_iter must be greater than zero"))
        (0.0 <= z && z <= 1.0) || throw(DomainError("radial_start_point_shift must be in the closed interval [0.0,1.0]"))
        (0.0 <= a && a <= 1.0) || throw(DomainError("ellipse_sqrt_distortion must be in the closed interval [0.0,1.0]"))
        return new(w, x, y, z, a, use_ellipse)
    end
end


"""
    RadialMLEMethod(ellipse_start_point_shift::Float64, ellipse_sqrt_distortion::Float64)

Method for finding the bivariate boundary of a confidence profile by bracketing between the MLE point and points on the provided bounds in directions given by points found on the boundary of a ellipse approximation of the loglikelihood function around the MLE, `e`, using [EllipseSampling.jl](https://github.com/JoelTrent/EllipseSampling.jl) (see [`findNpointpairs_radialMLE!`](@ref) and [`bivariate_confidenceprofile_vectorsearch`](@ref))..

# Arguments
- `ellipse_start_point_shift`: a number ∈ [0.0,1.0]. Default is `rand()` (defined on [0.0,1.0]), meaning that, by default, every time this function is called a different set of points will be generated.
- `ellipse_sqrt_distortion`: a number ∈ [0.0,1.0]. Default is `0.01`. 

# Details

For additional information on arguments see the keyword arguments for `generate_N_clustered_points` in [EllipseSampling.jl](https://github.com/JoelTrent/EllipseSampling.jl).

Recommended for use with the [`EllipseApprox`](@ref) profile type. Will produce reasonable results for the [`LogLikelihood`](@ref) profile type when bivariate profile boundaries are convex. Otherwise, [`IterativeBoundaryMethod`](@ref), which has an option to use a starting solution from [`RadialMLEMethod`](@ref), is preferred as it iteratively improves the quality of the boundary and can discover regions not explored by this method. This method is unlikely to find boundaries that do not contain the MLE point (if they exist).

# Internal Points

Finds no internal points.

# Supertype Hiearachy

RadialMLEMethod <: AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any
"""
struct RadialMLEMethod <: AbstractBivariateVectorMethod
    ellipse_start_point_shift::Float64
    ellipse_sqrt_distortion::Float64

    function RadialMLEMethod(ellipse_start_point_shift=rand(), ellipse_sqrt_distortion=0.01) 
        (0.0 <= ellipse_start_point_shift && ellipse_start_point_shift <= 1.0) || throw(DomainError("ellipse_start_point_shift must be in the closed interval [0.0,1.0]"))
        (0.0 <= ellipse_sqrt_distortion && ellipse_sqrt_distortion <= 1.0) || throw(DomainError("ellipse_sqrt_distortion must be in the closed interval [0.0,1.0]"))
        return new(ellipse_start_point_shift, ellipse_sqrt_distortion)
    end
end

"""
    RadialRandomMethod(num_radial_directions::Int)

Method for finding the bivariate boundary of a confidence profile by finding internal boundary points using a uniform random distribution on provided bounds and choosing `num_radial_directions` to explore from these points (see [`findNpointpairs_radialRandom!`](@ref) and [`bivariate_confidenceprofile_vectorsearch`](@ref)).

# Arguments
- `num_radial_directions`: an integer greater than zero. 

# Details

Recommended for use with the [`LogLikelihood`](@ref) profile type. Radial directions are rescaled by the relative magnitude/scale of the two interest parameters so that the directions we explore and boundary found as a result are not dominated by the parameter with the larger magnitude.

The method first uniformly samples the region specified by the bounds for the two interest parameters until a point within the boundary is found. Then it chooses `num_radial_directions`, spaced uniformly around a circle, and rotates these directions by a random phase shift ∈ `[0.0, 2π/num_directions]` radians. These directions are then distorted by the relative magnitude/scale of the two interest parameters. Then for each of these directions, until it runs out of directions or finds the desired number of points, it finds the closest point on the bounds from the internal point in this direction. If the point on the bounds is outside the boundary, it accepts the point pair. A bracketing method is then used to find a boundary point between the point pair (the bound point and the internal point). The method continues searching for internal points and generating directions until the desired number of boundary points is found.
    
Given a fixed number of desired boundary points, we can decrease the cost of finding internal points by increasing the number of radial directions to explore, `num_radial_directions`, at each internal point. However, it is important to balance `num_radial_directions` with the desired number of boundary points. If `num_radial_directions` is large relative to the number of boundary points, then the boundary the method finds may constitute more of a local search around found internal points. Resultantly, there may be regions were the boundary is not well explored. This will be less of an issue for more 'circular' boundary regions.   

[`IterativeBoundaryMethod`](@ref) may be preferred over this method if evaluating the loglikelihood function is expensive or the bounds provided for the interest parameters are much larger than the boundary, as the uniform random sampling for internal points will become very expensive. 

This method can find multiple boundaries (if they exist).

# Internal Points

Finds a minimum of `div(num_points, num_radial_directions, RoundUp )` internal points.

# Supertype Hiearachy

RadialRandomMethod <: AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any
"""
struct RadialRandomMethod <: AbstractBivariateVectorMethod
    num_radial_directions::Int
    function RadialRandomMethod(num_radial_directions) 
        num_radial_directions > 0 ||  throw(DomainError("num_radial_directions must be greater than zero")) 
        return new(num_radial_directions)
    end
end

"""
    SimultaneousMethod(num_radial_directions::Int)

Method for finding the bivariate boundary of a confidence profile by finding internal and external boundary points using a uniform random distribution on provided bounds, pairing these points in the order they're found and bracketing between each pair (see [`findNpointpairs_simultaneous!`](@ref) and [`bivariate_confidenceprofile_vectorsearch`](@ref)).

# Arguments
- `num_radial_directions`: an integer greater than zero. 

# Details

Recommended for use with the [`LogLikelihood`](@ref) profile type. 

The method first uniformly samples the region specified by the bounds for the two interest parameters until a point within the boundary is found. Then it chooses `num_radial_directions`, spaced uniformly around a circle, and rotates these directions by a random phase shift ∈ `[0.0, 2π/num_directions]` radians. These directions are then distorted by the relative magnitude/scale of the two interest parameters. Then for each of these directions, until it runs out of directions or finds the desired number of points, it finds the closest point on the bounds from the internal point in this direction. If the point on the bounds is outside the boundary, it accepts the point pair. A bracketing method is then used to find a boundary point between the point pair (the bound point and the internal point). The method continues searching for internal points and generating directions until the desired number of boundary points is found.

[`RadialRandomMethod`](@ref) and [`IterativeBoundaryMethod`](@ref) are preferred over this method from a computational performance standpoint as they require fewer loglikelihood evalutions (when [`RadialRandomMethod`](@ref) has parameter `num_radial_directions` > 1). 

This method can find multiple boundaries (if they exist).

# Internal Points

Finds `num_points` internal points.

# Supertype Hiearachy

SimultaneousMethod <: AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any
"""
struct SimultaneousMethod <: AbstractBivariateVectorMethod end


"""
    Fix1AxisMethod

Method for finding the bivariate boundary of a confidence profile by using uniform random distributions on provided bounds to draw a value for one interest parameter, fix it, and draw two values for the other interest parameter, using the pair to find a boundary point using a bracketing method if the pair are a valid bracket (see [`findNpointpairs_fix1axis!`](@ref) and [`bivariate_confidenceprofile_fix1axis`](@ref)).

# Details

Recommended for use with the [`LogLikelihood`](@ref) profile type. 

The method finds the desired number of boundary points by fixing the first and second interest parameters for half of these points each. It first draws a value for the fixed parameter using a uniform random distribution on provided bounds (e.g. Ψ_x). Then it draws two values for the unfixed parameter in the same fashion (e.g. Ψ_y1 and Ψ_y2]). If one of these points ([Ψ_x, Ψ_y1] and [Ψ_x, Ψ_y2]) is an internal point and the other an external point, the point pair is accepted as they are a valid bracket. A bracketing method is then used to find a boundary point between the point pair (the internal and external point). The method continues searching for valid point pairs until the desired number of boundary points is found.

[`RadialRandomMethod`](@ref) and [`IterativeBoundaryMethod`](@ref) are preferred over this method from a computational performance standpoint as they require fewer loglikelihood evalutions (when [`RadialRandomMethod`](@ref) has parameter `num_radial_directions` > 1). [`SimultaneousMethod`](@ref) will also likely be more efficient, even though it uses four random numbers vs three, as it doesn't unneccesarily throw away points.  

This method can find multiple boundaries (if they exist).

# Internal Points

Finds `num_points` internal points.

# Supertype Hiearachy

Fix1AxisMethod <: AbstractBivariateMethod <: Any
"""
struct Fix1AxisMethod <: AbstractBivariateMethod end

"""
`ellipse_confidence_level` is the confidence level at which to construct the initial ellipse.
`num_level_sets` the number of level sets used to get to the highest confidence level set specified in target_confidence_levels. `num_level_sets` ≥ length(target_confidence_levels)
"""
struct ContinuationMethod <: AbstractBivariateMethod 
    num_level_sets::Int
    ellipse_confidence_level::Float64
    # target_confidence_level::Float64
    # target_confidence_levels::Union{Float64, Vector{<:Float64}}
    ellipse_start_point_shift::Float64
    level_set_spacing::Symbol

    function ContinuationMethod(x,y,z=rand(),spacing=:loglikelihood)
        x > 0 || throw(DomainError("num_level_sets must be greater than zero"))

        (0.0 < y && y < 1.0) || throw(DomainError("ellipse_confidence_level must be in the open interval (0.0,1.0)"))

        # (0.0 < y && y < 1.0) || throw(DomainError("target_confidence_level must be in the interval (0.0,1.0)"))
        # if y isa Float64

        (0.0 <= z && z <= 1.0) || throw(DomainError("ellipse_start_point_shift must be in the closed interval [0.0,1.0]"))
        spacing ∈ [:confidence, :loglikelihood] || throw(ArgumentError("level_set_spacing must be either :confidence or :loglikelihood"))

        return new(x,y,z,spacing)
    end
end

"""

"""
struct AnalyticalEllipseMethod <: AbstractBivariateMethod end

"""
    bivariate_methods()

Prints a list of available bivariate methods. Available bivariate methods include [`IterativeBoundaryMethod`](@ref), [`RadialRandomMethod`](@ref), [`RadialMLEMethod`](@ref), [`SimultaneousMethod`](@ref), [`Fix1AxisMethod`](@ref), [`ContinuationMethod`](@ref) and [`AnalyticalEllipseMethod`](@ref).
"""
function bivariate_methods()
    methods = [IterativeBoundaryMethod, RadialRandomMethod, RadialRandomMethod, RadialMLEMethod, SimultaneousMethod, Fix1AxisMethod, ContinuationMethod, AnalyticalEllipseMethod]
    println(string("Available bivariate methods are: ", [i != length(methods) ? string(method, ", ") : string(method) for (i,method) in enumerate(methods)]...))
    return nothing
end