# functions for the BracketingMethodRadialRandom, BracketingMethodRadialMLE and BracketingMethodSimultaneous methods

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
                                        ind2::Int,
                                        mle_targetll::Float64,
                                        save_internal_points::Bool,
                                        biv_opt_is_ellipse_analytical::Bool)

    internal  = zeros(2,num_points)
    internal_all = zeros(model.core.num_pars, save_internal_points ? num_points : 0)
    ll_values = zeros(save_internal_points ? num_points : 0)
    external = zeros(2,num_points)

    Ninside=0; Noutside=0
    iters=0
    while Noutside<num_points && Ninside<num_points

        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        g = bivariate_optimiser(0.0, p)
        if g > 0
            Ninside+=1
            internal[:,Ninside] .= [x,y]

            if save_internal_points
                ll_values[Ninside] = g * 1.0
                if !biv_opt_is_ellipse_analytical
                    internal_all[[ind1, ind2], Ninside] .= x, y
                    variablemapping2d!(@view(internal_all[:, Ninside]), p.λ_opt, p.θranges, p.λranges)
                end
            end
        else
            Noutside+=1
            external[:,Noutside] .= [x,y]
        end
        iters+=1
    end

    # while Ninside < N && iters < maxIters
    while Ninside < num_points
        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        g = bivariate_optimiser(0.0, p)
        if g > 0
            Ninside+=1
            internal[:,Ninside] .= [x,y]

            if save_internal_points
                ll_values[Ninside] = g * 1.0
                if !biv_opt_is_ellipse_analytical
                    internal_all[[ind1, ind2], Ninside] .= x, y
                    variablemapping2d!(@view(internal_all[:, Ninside]), p.λ_opt, p.θranges, p.λranges)
                end
            end
        end
        iters+=1
    end

    # while Noutside < N && iters < maxIters
    while Noutside < num_points
        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        if bivariate_optimiser(0.0, p) < 0
            Noutside+=1
            external[:,Noutside] .= [x,y]
        end
        iters+=1
    end

    if save_internal_points && biv_opt_is_ellipse_analytical
        get_λs_bivariate_ellipse_analytical!(internal_all, num_points,
                                                    p.consistent, ind1, ind2, 
                                                    model.core.num_pars, p.initGuess,
                                                    p.θranges, p.λranges)
    end

    if save_internal_points; ll_values .= ll_values .+ mle_targetll end

    return internal, internal_all, ll_values, external
end

function find_m_spaced_radialdirections(num_directions::Int; start_point_shift::Float64=rand())
    radial_dirs = zeros(num_directions)
    radial_dirs .= (start_point_shift * 2.0 / convert(Float64, num_directions)) .+ collect(LinRange(1e-12, 2.0, num_directions+1))[1:end-1]
    return radial_dirs
end

"""
Distorts uniformly spaced angles on a circle to angles on an ellipse representative of the relative magnitude of each parameter. If the magnitude of a parameter is a NaN value (i.e. either bound is Inf), then the relative magnitude is set to 1.0, as no information is known about its magnitude.
"""
function findNpointpairs_radialrandom!(p::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    num_directions::Int, 
                                    ind1::Int, 
                                    ind2::Int,
                                    mle_targetll::Float64,
                                    save_internal_points::Bool,
                                    biv_opt_is_ellipse_analytical::Bool)

    internal  = zeros(2,num_points)
    internal_all = zeros(model.core.num_pars, save_internal_points ? num_points : 0)
    ll_values = zeros(save_internal_points ? num_points : 0)
    internal_unique = trues(num_points)
    external = zeros(2,num_points)

    save_λs = save_internal_points && !biv_opt_is_ellipse_analytical

    count = 0
    internal_count=0
    λ_opt = zeros(model.core.num_pars-2)
    g_ll = 0.0
    
    if isnan(model.core.θmagnitudes[ind1]) || isnan(model.core.θmagnitudes[ind2]) 
        relative_magnitude = 1.0
    else
        relative_magnitude = model.core.θmagnitudes[ind1]/model.core.θmagnitudes[ind2]
    end
    while count < num_points
        x, y = 0.0, 0.0
        # find an internal point
        while true
            x, y = generatepoint(model, ind1, ind2)
            p.pointa .= [x,y]
            g_gen = bivariate_optimiser(0.0, p)
            if g_gen > 0 
                if save_internal_points; g_ll = g_gen end
                if save_λs; λ_opt .= p.λ_opt end
                break
            end
        end

        radial_dirs = find_m_spaced_radialdirections(num_directions)

        count_accepted=0
        for i in 1:num_directions
            dir_vector = [relative_magnitude * cospi(radial_dirs[i]), sinpi(radial_dirs[i]) ]
            boundpoint = findpointonbounds(model, [x, y], dir_vector, ind1, ind2)
            # boundpoint = findpointonbounds(model, [x, y], radial_dirs[i], ind1, ind2)

            # if bound point is a point outside the boundary, accept the point combination
            p.pointa .= boundpoint
            g = bivariate_optimiser(0.0, p)
            if g < 0
                count += 1
                count_accepted += 1
                internal[:, count] .= x, y 

                # make bracket a tiny bit smaller
                if isinf(g)
                    v_bar = boundpoint .- internal[:, count]
                    boundpoint .= internal[:, count] .+ ((1.0-1e-8) .* v_bar)
                end

                external[:, count] .= boundpoint
                internal_unique[count] = count_accepted == 1

                if save_internal_points && count_accepted == 1
                    internal_count += 1
                    ll_values[internal_count] = g_ll * 1.0
                    if !biv_opt_is_ellipse_analytical
                        internal_all[[ind1, ind2], internal_count] .= x, y
                        variablemapping2d!(@view(internal_all[:, internal_count]), λ_opt, p.θranges, p.λranges)
                    end
                end
            end

            if count == num_points
                break
            end
        end
    end

    if save_internal_points 
        ll_values = ll_values[1:internal_count] .+ mle_targetll
        
        if biv_opt_is_ellipse_analytical
            internal_all = get_λs_bivariate_ellipse_analytical!(internal[:, internal_unique], sum(internal_unique),
                                                                p.consistent, ind1, ind2, 
                                                                model.core.num_pars, p.initGuess,
                                                                p.θranges, p.λranges)
        else
            internal_all = internal_all[:, 1:internal_count]
        end
    end

    return internal, internal_all, ll_values, external
end

function findNpointpairs_radialMLE!(p::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    ind1::Int, 
                                    ind2::Int,
                                    ellipse_confidence_level::Float64,
                                    ellipse_start_point_shift::Float64,
                                    ellipse_sqrt_distortion::Float64)

    mle_point = model.core.θmle[[ind1, ind2]]
    internal = zeros(2,num_points) .= mle_point
    external = zeros(2,num_points)
    point_is_on_bounds = falses(num_points)
    # warn if bound prevents reaching boundary
    bound_warning=true

    check_ellipse_approx_exists!(model)
    ellipse_points = generate_N_clustered_points(num_points, model.ellipse_MLE_approx.Γmle,
                                                        model.core.θmle, ind1, ind2,
                                                        confidence_level=ellipse_confidence_level, 
                                                        start_point_shift=ellipse_start_point_shift, 
                                                        sqrt_distortion=ellipse_sqrt_distortion)

    bound_ind=0
    for i in 1:num_points
        dir_vector = ellipse_points[:,i] .- mle_point
        external[:,i], bound_ind, upper_or_lower = findpointonbounds(model, mle_point, dir_vector, ind1, ind2, true)

        p.pointa .= external[:,i]
        g = bivariate_optimiser(0.0, p)
        if g ≥ 0
            point_is_on_bounds[i] = true

            if bound_warning
                @warn string("The ", upper_or_lower, " bound on variable ", model.core.θnames[bound_ind], " is inside the confidence boundary")
                bound_warning = false
            end
        else
            # make bracket a tiny bit smaller
            if isinf(g)
                v_bar = external[:,i] .- mle_point
                external[:,i] .= mle_point .+ ((1.0-1e-8) .* v_bar)
            end
        end
    end

    internal_all = zeros(model.core.num_pars, 0)
    ll_values = zeros(0)
    return internal, internal_all, ll_values, external, point_is_on_bounds, bound_warning
end

function bivariate_confidenceprofile_vectorsearch(bivariate_optimiser::Function, 
                                                    model::LikelihoodModel, 
                                                    num_points::Int, 
                                                    consistent::NamedTuple, 
                                                    ind1::Int, 
                                                    ind2::Int,
                                                    mle_targetll::Float64,
                                                    save_internal_points::Bool;
                                                    num_radial_directions::Int=0,
                                                    ellipse_confidence_level::Float64=-1.0,
                                                    ellipse_start_point_shift::Float64=0.0,
                                                    ellipse_sqrt_distortion::Float64=0.0)

    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateΨ_ellipse_analytical_vectorsearch
    
    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]
    boundary = zeros(model.core.num_pars, num_points)

    if biv_opt_is_ellipse_analytical
        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                    θranges=θranges, λranges=λranges, consistent=consistent)
    else
        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                    θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-2))
    end

    if ellipse_confidence_level !== -1.0
        internal, internal_all, ll_values, external, point_is_on_bounds, _ = findNpointpairs_radialMLE!(p, bivariate_optimiser, model, num_points, ind1, ind2, 
                                                                                            ellipse_confidence_level, ellipse_start_point_shift, ellipse_sqrt_distortion)

    else
        if num_radial_directions == 0
            internal, internal_all, ll_values, external = findNpointpairs_simultaneous!(p, bivariate_optimiser, model, num_points, ind1, ind2,
                                                                                mle_targetll, save_internal_points, biv_opt_is_ellipse_analytical)
        else
            internal, internal_all, ll_values, external = findNpointpairs_radialrandom!(p, bivariate_optimiser, model, num_points, 
                                                                                num_radial_directions, ind1, ind2,
                                                                                mle_targetll,
                                                                                save_internal_points, biv_opt_is_ellipse_analytical)
        end
        point_is_on_bounds = falses(num_points)
    end

    for i in 1:num_points
        if point_is_on_bounds[i]
            p.pointa .= external[:,i]
            bivariate_optimiser(0, p)
            boundary[[ind1, ind2], i] .= external[:,i]
        else
            p.pointa .= internal[:,i]
            v_bar = external[:,i] .- internal[:,i]

            v_bar_norm = norm(v_bar, 2)
            p.uhat .= v_bar ./ v_bar_norm

            Ψ = find_zero(bivariate_optimiser, (0.0, v_bar_norm), Roots.Brent(); p=p)
            
            boundary[[ind1, ind2], i] .= p.pointa + Ψ*p.uhat
        end
        if !biv_opt_is_ellipse_analytical
            variablemapping2d!(@view(boundary[:, i]), p.λ_opt, θranges, λranges)
        end
    end

    if biv_opt_is_ellipse_analytical
        return get_λs_bivariate_ellipse_analytical!(@view(boundary[[ind1, ind2], :]), num_points,
                                                    consistent, ind1, ind2, 
                                                    model.core.num_pars, initGuess,
                                                    θranges, λranges, boundary), PointsAndLogLikelihood(internal_all, ll_values)
    end

    return boundary, PointsAndLogLikelihood(internal_all, ll_values)
end