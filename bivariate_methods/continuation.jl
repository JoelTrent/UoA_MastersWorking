function update_targetll!(p, target_confidence_ll)
    return merge(p, (targetll=target_confidence_ll,))
end

function normal_vector_i_2d!(gradient_i, index, points)
    if index == 1
        gradient_i .= [(points[2,end]-points[2,2]), -(points[1,end]-points[1,2])]
    elseif index == size(points, 2)
        gradient_i .= [(points[2,end-1]-points[2,1]), -(points[1,end-1]-points[1,1])]
    else
        gradient_i .= [(points[2,index-1]-points[2,index+1]), -(points[1,index-1]-points[1,index+1])]
    end
    return nothing
end

"""
start_level_set only has two rows. Within this function we keep track of both the 2d points on the boundary - as this is all that needs to be known to get to the next point

For a given level set to get to, that's larger than all points in current level set. 
* Find normal direction at each point (either using ForwardDiff or a first order approximation)
* Check where this normal direction intersects bounds and ensure that the next level set is bracketed by current point and point on boundary
* If a bracket between current point and point on boundary does not exist the level set point recorded will be the point on the boundary
* init guess for nuisance parameters should be their value at current point
* using vector search bivariate function as input to find_zero(), using Order0 method, with starting guess of Ψ=0
* record point at that level set


LATER: Update to allow specification of alternate method to find_zero(), e.g. Order8 OR alternate method using NLsolve (and LineSearches) in "2DGradientLinesearchTests.jl" which is gradient based.
"""
function continuation_line_search!(p::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    bivariate_optimiser_gradient::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int,
                                    ind1::Int, 
                                    ind2::Int,
                                    atol::Float64,
                                    target_confidence_ll::Float64, 
                                    start_level_set_2D::Matrix{Float64}, 
                                    start_level_set_all::Matrix{Float64}=Matrix{Float64}(undef,0,0))

    f_gradient(x) = bivariate_optimiser_gradient(x, p)
    
    start_have_all_pars = !isempty(start_level_set_all) 
    
    target_level_set_2D = zeros(2, num_points)
    target_level_set_all = zeros(model.core.num_pars, num_points)
    
    gradient_i = [0.0,0.0]
    boundpoint = [0.0,0.0]
    boundarypoint = [0.0,0.0]
    p = update_targetll!(p, target_confidence_ll)
    
    for i in 1:num_points
        p.pointa .= start_level_set_2D[:,i]

        # if know the optimised values of nuisance parameters at a given start point,
        # pass them to the optimiser
        if start_have_all_pars
            boundsmapping2d!(p.initGuess, @view(start_level_set_all[:,i]), ind1, ind2)
        end

        # calculate gradient at point i; want to go in downhill direction
        # FORWARDDIFF NOT WORKING WITH ANON FUNCTION JUST YET - likely because it is contains a mutating function, i.e. zeros()
        # 
        # gradient_i .= -ForwardDiff.gradient(f_gradient, p.pointa)

        normal_vector_i_2d!(gradient_i, i, start_level_set_2D)

        p.uhat .= (gradient_i / (norm(gradient_i, 2))) 

        boundpoint .= findpointonbounds(model, p.pointa, p.uhat, ind1, ind2)

        # if bound point and pointa bracket a boundary, search for the boundary
        # otherwise, the bound point is used as the level set boundary (i.e. it's inside the true level set boundary)
        if bivariate_optimiser_gradient(boundpoint, p) < 0
            Ψ_y1 = solve(ZeroProblem(bivariate_optimiser, 0.0), atol=atol, Roots.Order8(); p=p)

            # in event Roots.Order8 fails to converge, switch to bracketing method
            if isnan(Ψ_y1)
                # value of v_bar_norm that satisfies the equation boundpoint = p.pointa + Ψ_y1*p.uhat
                v_bar_norm = (boundpoint[1] - p.pointa[1]) / p.uhat[1] 
                Ψ_y1 = find_zero(bivariate_optimiser, (0.0, v_bar_norm), atol=atol, Roots.Brent(); p=p)
            end

            boundarypoint .= p.pointa + Ψ_y1*p.uhat
            target_level_set_2D[:, i] .= boundarypoint
            target_level_set_all[[ind1, ind2], i] .= boundarypoint
        else
            target_level_set_2D[:, i] .= boundpoint
            target_level_set_all[[ind1, ind2], i] .= boundpoint
        end
        variablemapping2d!(@view(target_level_set_all[:, i]), p.λ_opt, p.θranges, p.λranges)
    end

    return target_level_set_2D, target_level_set_all
end

function continuation_inwards_radial_search!(p::NamedTuple, 
                                                bivariate_optimiser::Function, 
                                                model::LikelihoodModel, 
                                                num_points::Int, 
                                                ind1::Int, 
                                                ind2::Int, 
                                                atol::Float64,
                                                target_confidence_ll::Float64, 
                                                start_level_set_2D::Matrix{Float64})

    mle_point = model.core.θmle[[ind1, ind2]]
    
    target_level_set_2D = zeros(2, num_points)
    target_level_set_all = zeros(model.core.num_pars, num_points)
    
    boundarypoint = [0.0, 0.0]
    p = update_targetll!(p, target_confidence_ll)

    p.pointa .= mle_point
    # effectively equivalent code to vector search code 
    for i in 1:num_points
        v_bar = start_level_set_2D[:,i] - mle_point

        v_bar_norm = norm(v_bar, 2)
        p.uhat .= v_bar / v_bar_norm

        # there's a chance that the start_level_set_2D[:, i] is exactly on the boundary of interest, so define the bracket as slightly larger than that point
        Ψ_y1 = find_zero(bivariate_optimiser, (0.0, v_bar_norm*1.2), atol=atol, Roots.Brent(); p=p)
        
        boundarypoint .= p.pointa + Ψ_y1*p.uhat
        target_level_set_2D[:, i] .= boundarypoint
        target_level_set_all[[ind1, ind2], i] .= boundarypoint
        variablemapping2d!(@view(target_level_set_all[:, i]), p.λ_opt, p.θranges, p.λranges)
    end

    return target_level_set_2D, target_level_set_all
end

"""

The initial ellipse solution also should be in feasible region: contained within bounds specified for interest parameters. WARN if it is not - may cause some unexpected behaviour if the parameter is meant to be ≥ 0, yet is allowed to start there in the initial ellipse solution.

Find extrema of true log likelihoods of initial ellipse solution
3 cases: 
Case 1 is preferred. Warnings will be raised for both Case 2 and 3
Case 1: If min ll > than target ll of smallest target confidence level (preferred approach atm). Then line search from initial ellipse to ll boundary defined by min ll and this is the starting continuation solution. Line search in direction specified by forward diff at point

Case 2: If max ll < than target ll of smallest target confidence level. Line search radially towards the mle solution from the ellipse to the smallest target confidence level boundary and this is the starting continuation solution. In case two, this counts as a 'level set'.

Case 3: If min ll and max ll bracket the smallest target confidence level. Then line search radially towards the mle solution from initial ellipse to ll boundary defined by max ll and this is the starting continuation solution.

For both Case 2 and 3, could use gradient at initial elipse point rather than going radially towards the mle, but in my opinion may not be able to find the log likelihood boundary.
"""
function initial_continuation_solution!(p::NamedTuple, 
                                        bivariate_optimiser::Function, 
                                        bivariate_optimiser_gradient::Function, 
                                        model::LikelihoodModel, 
                                        num_points::Int, 
                                        ind1::Int, 
                                        ind2::Int,
                                        atol::Float64,
                                        profile_type::AbstractProfileType,
                                        ellipse_confidence_level::Float64, 
                                        target_confidence_ll::Float64)
    # get initial continuation starting solution
    # internal boundary - preferably very small
    ellipse_points = generate_N_equally_spaced_points(num_points, model.ellipse_MLE_approx.Γmle,
                                                        model.core.θmle, ind1, ind2,
                                                        confidence_level=ellipse_confidence_level, 
                                                        start_point_shift=0.0)

    for i in 1:num_points
        if model.core.θlb[ind1] > ellipse_points[1,i] || model.core.θub[ind1] < ellipse_points[1,i]
            @warn string("initial ellipse starting solution for 2D continuation method with variables ", model.core.θnames[ind1], " and ", model.core.θnames[ind2]," contains solutions outside specified bounds for ", model.core.θnames[ind1], ". This may cause unexpected behaviour - a smaller ellipse confidence level is recommended.")
            break
        end
        if model.core.θlb[ind2] > ellipse_points[2,i] || model.core.θub[ind2] < ellipse_points[2,i]
            @warn string("initial ellipse starting solution for 2D continuation method with variables ", model.core.θnames[ind1], " and ", model.core.θnames[ind2]," contains solutions outside specified bounds for ", model.core.θnames[ind2], ". This may cause unexpected behaviour - a smaller ellipse confidence level is recommended.")
            break
        end
    end

    # calculate true log likelihood at each point on ellipse approx
    ellipse_true_lls = zeros(num_points)
    update_targetll!(p, 0.0)

    for i in 1:num_points
        p.pointa .= ellipse_points[:,i]
        ellipse_true_lls[i] = bivariate_optimiser(0.0, p)
    end

    if profile_type isa LogLikelihood
        ellipse_true_lls .= ellipse_true_lls .- model.core.maximisedmle
    end

    min_ll, max_ll = extrema(ellipse_true_lls)

    if target_confidence_ll < min_ll # case 1
        corrected_ll = ll_correction(model, profile_type, min_ll)
        a, b = continuation_line_search!(p, bivariate_optimiser, 
                                            bivariate_optimiser_gradient, model, 
                                            num_points, ind1, ind2, atol,
                                            corrected_ll, ellipse_points)
        return a, b, min_ll

    elseif max_ll < target_confidence_ll # case 2
        corrected_ll = ll_correction(model, profile_type, target_confidence_ll)

        @warn string("ellipse starting point for continuation contains the smallest target confidence level set. Using a smaller ellipse confidence level is recommended")
        a, b = continuation_inwards_radial_search!(p, bivariate_optimiser, model, 
                                                    num_points, ind1, ind2, atol,
                                                    corrected_ll, ellipse_points)
        return a, b, target_confidence_ll
    end

    # else # case 3
    corrected_ll = ll_correction(model, profile_type, max_ll)
    println(corrected_ll)
    println(min_ll)
    println(target_confidence_ll)
    println(max_ll)

    @warn string("ellipse starting point for continuation intersects the smallest target confidence level set. Using a smaller ellipse confidence level is recommended")
    a, b = continuation_inwards_radial_search!(p, bivariate_optimiser, model, num_points, 
                                                ind1, ind2, atol, corrected_ll, ellipse_points)
    return a, b, max_ll
end

"""
bivariate_optimiser is the optimiser to use with find_zero
bivariate_optimiser_gradient is the optimiser to use with ForwardDiff.gradient and to evaluate the obj value for the initial ellipse solution.
"""
function bivariate_confidenceprofile_continuation(bivariate_optimiser::Function, 
                                                    bivariate_optimiser_gradient::Function, 
                                                    model::LikelihoodModel, 
                                                    num_points::Int, 
                                                    consistent::NamedTuple, 
                                                    ind1::Int, 
                                                    ind2::Int,
                                                    atol::Float64,
                                                    profile_type::AbstractProfileType,
                                                    ellipse_confidence_level::Float64, 
                                                    target_confidence_level::Float64, 
                                                    num_level_sets::Int)

    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)
    
    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]

    if profile_type isa EllipseApproxAnalytical
        boundarySamples = zeros(2, num_points)
        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                    θranges=θranges, λranges=λranges, consistent=consistent, targetll=0.0)
    else
        boundarySamples = zeros(model.core.num_pars, num_points)
        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                    θranges=θranges, λranges=λranges, consistent=consistent, targetll=0.0, 
                    λ_opt=zeros(model.core.num_pars-2))
    end

    # PERHAPS HAVE DIFFERENT VERSIONS DEPENDING ON WHETHER ForwardDiff or 1st Order Approx IS USED FOR NORMAL DIRECTIONS? ForwardDiff version can easily parallelise across points, if consider each point separately and find next point on level set. Vs 1st order approx requires knowledge of the points on either side of it on the level set, so would parallelise across points within a single level set, which is probs not as good. 

    # Specify x number of level sets to pass through.
    # Specify target level set(s) to reach (pass through) and/or record.
    # Specify confidence level of initial ellipse guess.

    initial_target_ll = get_target_loglikelihood(model, target_confidence_level,
                                                 EllipseApproxAnalytical(), 2)

    # find initial solution
    current_level_set_2D, current_level_set_all, initial_ll =
        initial_continuation_solution!(p, bivariate_optimiser, bivariate_optimiser_gradient, 
                                        model, num_points, ind1, ind2, atol, profile_type,
                                        ellipse_confidence_level, initial_target_ll)

    if initial_ll == initial_target_ll
        return current_level_set_all
    end

    initial_confidence_level = cdf(Chisq(2), -initial_ll*2.0)

    conf_level_sets = collect(LinRange(initial_confidence_level, target_confidence_level, num_level_sets+1)[2:end])

    level_set_lls = [get_target_loglikelihood(model, conf_level_sets[i], 
                        profile_type, 2) for i in 1:num_level_sets]

    for level_set_ll in level_set_lls
        current_level_set_2D, current_level_set_all = 
            continuation_line_search!(p, bivariate_optimiser, bivariate_optimiser_gradient,
                                        model, num_points, ind1, ind2, atol, level_set_ll,
                                        current_level_set_2D, current_level_set_all)
    end

    return current_level_set_all
end