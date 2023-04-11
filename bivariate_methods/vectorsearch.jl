# functions for the BracketingMethodRadial and BracketingMethodSimultaneous methods

function generatepoint(model::LikelihoodModel, ind1::Int, ind2::Int)
    return rand(Uniform(model.core.θlb[ind1], model.core.θub[ind1])), rand(Uniform(model.core.θlb[ind2], model.core.θub[ind2]))
end

# function findNpointpairs(p, N, lb, ub, ind1, ind2; maxIters)
"""
At later stage, implement with maxIters in the event that either the user-specified bounds don't contain internal points (or similarly external points) for a given confidence region (i.e. bounds either don't contain the 2D boundary at a confidence level OR the 2D boundary at a confidence level contains the bounds)
"""
function findNpointpairs_simultaneous!(p::NamedTuple, 
                                        bivariate_optimiser::Function, 
                                        model::LikelihoodModel, 
                                        num_points::Int, 
                                        ind1::Int, 
                                        ind2::Int)

    insidePoints  = zeros(2,num_points)
    outsidePoints = zeros(2,num_points)

    Ninside=0; Noutside=0
    iters=0
    while Noutside<num_points && Ninside<num_points

        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        if bivariate_optimiser(0.0, p) > 0
            Ninside+=1
            insidePoints[:,Ninside] .= [x,y]
        else
            Noutside+=1
            outsidePoints[:,Noutside] .= [x,y]
        end
        iters+=1
    end

    # while Ninside < N && iters < maxIters
    while Ninside < num_points
        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        if bivariate_optimiser(0.0, p) > 0
            Ninside+=1
            insidePoints[:,Ninside] .= [x,y]
        end
        iters+=1
    end

    # while Noutside < N && iters < maxIters
    while Noutside < num_points
        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        if bivariate_optimiser(0.0, p) < 0
            Noutside+=1
            outsidePoints[:,Noutside] .= [x,y]
        end
        iters+=1
    end

    return insidePoints, outsidePoints
end

function find_m_spaced_radialdirections(num_directions::Int)
    radial_dirs = zeros(num_directions)
    radial_dirs .= rand() * 2.0 / convert(Float64, num_directions) .+ collect(LinRange(1e-12, 2.0, num_directions+1))[1:end-1]
    return radial_dirs
end

function findNpointpairs_radial!(p::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    num_directions::Int, 
                                    ind1::Int, 
                                    ind2::Int)

    insidePoints  = zeros(2,num_points)
    outsidePoints = zeros(2,num_points)

    count = 0
    while count < num_points
        x, y = 0.0,0.0
        # find an internal point
        while true
            x, y = generatepoint(model, ind1, ind2)
            p.pointa .= [x,y]
            (bivariate_optimiser(0.0, p) < 0) || break
        end
        
        radial_dirs = find_m_spaced_radialdirections(num_directions)

        for i in 1:num_directions
            boundpoint = findpointonbounds(model, [x, y], radial_dirs[i], ind1, ind2)

            # if bound point is a point outside the boundary, accept the point combination
            p.pointa .= boundpoint
            if (bivariate_optimiser(0.0, p) < 0)
                count +=1
                insidePoints[:, count] .= x, y 
                outsidePoints[:, count] .= boundpoint
            end

            if count == num_points
                break
            end
        end
    end
    return insidePoints, outsidePoints
end

function bivariate_confidenceprofile_vectorsearch(bivariate_optimiser::Function, 
                                                    model::LikelihoodModel, 
                                                    num_points::Int, 
                                                    consistent::NamedTuple, 
                                                    ind1::Int, 
                                                    ind2::Int; 
                                                    num_radial_directions::Int=0)


    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateΨ_ellipse_analytical_vectorsearch
    
    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]

    if biv_opt_is_ellipse_analytical
        boundarySamples = zeros(2, num_points)
        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                    θranges=θranges, λranges=λranges, consistent=consistent)
    else
        boundarySamples = zeros(model.core.num_pars, num_points)
        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                    θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-2))
    end

    if num_radial_directions == 0
        insidePoints, outsidePoints = findNpointpairs_simultaneous!(p, bivariate_optimiser, model, num_points, ind1, ind2)
    else
        insidePoints, outsidePoints = findNpointpairs_radial!(p, bivariate_optimiser, model, num_points, num_radial_directions, ind1, ind2)
    end

    ϵ=1e-8
    for i in 1:num_points
        p.pointa .= insidePoints[:,i]
        v_bar = outsidePoints[:,i] - insidePoints[:,i]

        v_bar_norm = norm(v_bar, 2)
        p.uhat .= v_bar / v_bar_norm

        Ψ_y1 = find_zero(bivariate_optimiser, (0.0, v_bar_norm), atol=ϵ, Roots.Brent(); p=p)
        
        if biv_opt_is_ellipse_analytical
            boundarySamples[:, i] .= p.pointa + Ψ_y1*p.uhat
        else
            boundarySamples[[ind1, ind2], i] .= p.pointa + Ψ_y1*p.uhat
            variablemapping2d!(@view(boundarySamples[:, i]), p.λ_opt, θranges, λranges)
        end
    end

    return boundarySamples
end