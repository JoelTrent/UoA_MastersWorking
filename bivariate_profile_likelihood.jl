function variablemapping2dranges(num_pars::T, index1::T, index2::T) where T <: Int
    θranges = (1:(index1-1), (index1+1):(index2-1), (index2+1):num_pars)
    λranges = (1:(index1-1), index1:(index2-2), (index2-1):(num_pars-2) )
    return θranges, λranges
end

function variablemapping2d!(θ, λ, θranges, λranges)
    θ[θranges[1]] .= @view(λ[λranges[1]])
    θ[θranges[2]] .= @view(λ[λranges[2]])
    θ[θranges[3]] .= @view(λ[λranges[3]])
    return θ
end

# we know index1 < index2 by construction. If index1 and index2 are user provided, enforce this relationship 
function boundsmapping2d!(newbounds::Vector{<:Float64}, bounds::Vector{<:Float64}, index1::Int, index2::Int)
    newbounds[1:(index1-1)]      .= @view(bounds[1:(index1-1)])
    newbounds[index1:(index2-2)] .= @view(bounds[(index1+1):(index2-1)])
    newbounds[(index2-1):end]    .= @view(bounds[(index2+1):end])
    return nothing
end

function normaliseduhat(v_bar); v_bar/norm(vbar, 2) end

function generatepoint(model::LikelihoodModel, ind1::Int, ind2::Int)
    return rand(Uniform(model.core.θlb[ind1], model.core.θub[ind1])), rand(Uniform(model.core.θlb[ind2], model.core.θub[ind2]))
end

# function findNpointpairs(p, N, lb, ub, ind1, ind2; maxIters)
function findNpointpairs_simultaneous(g::Function, model::LikelihoodModel, p::NamedTuple, num_points::Int, ind1::Int, ind2::Int)

    insidePoints  = zeros(2,num_points)
    outsidePoints = zeros(2,num_points)

    Ninside = 0; Noutside=0
    iters=0
    while Noutside<num_points && Ninside<num_points

        x, y = generatepoint(model, ind1, ind2)
        p=merge(p, (pointa=[x,y],))
        if g(0.0, p) > 0
            Ninside+=1
            insidePoints[:,Ninside] .= [x,y]
        else
            Noutside+=1
            outsidePoints[:,Noutside] .= [x,y]
        end
        iters+=1
        # println(iters)
    end

    # while Ninside < N && iters < maxIters
    while Ninside < num_points
        x, y = generatepoint(model, ind1, ind2)
        p=merge(p, (pointa=[x,y],))
        if g(0.0, p) > 0
            Ninside+=1
            insidePoints[:,Ninside] .= [x,y]
        end
        iters+=1
        # println(iters)
    end

    # while Noutside < N && iters < maxIters
    while Noutside < num_points
        x, y = generatepoint(model, ind1, ind2)
        p=merge(p, (pointa=[x,y],))
        if g(0.0, p) < 0
            Noutside+=1
            outsidePoints[:,Noutside] .= [x,y]
        end
        iters+=1
        # println(iters)
    end

    return insidePoints, outsidePoints
end

function findpointonbounds(model::LikelihoodModel, internalpoint::Tuple{Float64, Float64}, direction_πradians::Float64, ind1::Int, ind2::Int)

    # by construction 0 < direction_πradians < 2, i.e. direction_radians ∈ [1e-10, 2 - 1e-10]

    quadrant = convert(Int, div(direction_πradians, 0.5, RoundUp))

    if quadrant == 1
        xlim, ylim = model.core.θub[ind1], model.core.θub[ind2]
    elseif quadrant == 2
        xlim, ylim = model.core.θlb[ind1], model.core.θub[ind2]
    elseif quadrant == 3
        xlim, ylim = model.core.θlb[ind1], model.core.θlb[ind2]
    else
        xlim, ylim = model.core.θub[ind1], model.core.θlb[ind2]
    end

    cosDir = cospi(direction_πradians)
    sinDir = sinpi(direction_πradians)
    
    r_vals = abs.([(xlim-internalpoint[1]) / cosDir , (ylim-internalpoint[2]) / sinDir])

    r = minimum(r_vals)
    r_pos = argmin(r_vals)

    boundpoint = [0.0, 0.0]

    if r_pos == 1
        boundpoint[1] = xlim
        boundpoint[2] = internalpoint[2] + r * sinDir
    else
        boundpoint[1] = internalpoint[1] + r * cosDir
        boundpoint[2] = ylim
    end

    return boundpoint
end

function find_m_spaced_radialdirections(num_directions::Int)
    radial_dirs = zeros(num_directions)

    radial_dirs .= rand() * 2.0 / convert(Float64, num_directions) .+ collect(LinRange(1e-12, 2.0, num_directions+1))[1:end-1]

    return radial_dirs
end

function findNpointpairs_radial(g::Function, model::LikelihoodModel, p::NamedTuple, num_points::Int, num_directions::Int, ind1::Int, ind2::Int)

    insidePoints  = zeros(2,num_points)
    outsidePoints = zeros(2,num_points)

    count = 0

    while count < num_points
        x, y = 0.0,0.0
        # find an internal point
        while true
            x, y = generatepoint(model, ind1, ind2)
            p=merge(p, (pointa=[x,y],))
            (g(0.0, p) < 0) || break
        end
        
        radial_dirs = find_m_spaced_radialdirections(num_directions)

        for i in 1:num_directions
            boundpoint = findpointonbounds(model, (x, y), radial_dirs[i], ind1, ind2)

            # if bound point is a point outside the boundary, accept the point combination
            p=merge(p, (pointa=boundpoint,))
            if (g(0.0, p) < 0)
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

# function bivariateΨ_unsafe(Ψ, p)
#     θs=zeros(p.consistent.num_pars)
#     θs[p.ind1] = p.Ψ_x
#     θs[p.ind2] = Ψ
    
#     function fun(λ); return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data) end

#     (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
#     llb=fopt-p.consistent.targetll
#     return llb, xopt
# end

function bivariateΨ(Ψ::Real, p)
    θs=zeros(p.consistent.num_pars)
    
    function fun(λ)
        θs[p.ind1] = p.Ψ_x
        θs[p.ind2] = Ψ
        return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    return llb, xopt
end

# function bivariateΨ_vectorsearch_unsafe(Ψ, p)
#     θs=zeros(p.consistent.num_pars)
#     θs[p.ind1], θs[p.ind2] = p.pointa + Ψ*p.uhat
    
#     function fun(λ); return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data) end

#     (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
#     llb=fopt-p.consistent.targetll
#     return llb, xopt
# end

function bivariateΨ_vectorsearch(Ψ, p)
    θs=zeros(p.consistent.num_pars)
    Ψxy = p.pointa + Ψ*p.uhat
    
    function fun(λ)
        θs[p.ind1], θs[p.ind2] = Ψxy
        return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    return llb, xopt
end

function bivariateΨ_ellipse_analytical(Ψ, p)
    return analytic_ellipse_loglike([p.Ψ_x, Ψ], [p.ind1, p.ind2], p.consistent.data) - p.consistent.targetll
end

function bivariateΨ_ellipse_analytical_vectorsearch(Ψ, p)
    return analytic_ellipse_loglike(p.pointa + Ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data) - p.consistent.targetll
end


function init_bivariate_parameters(model::LikelihoodModel, ind1::Int, ind2::Int)
    newLb     = zeros(model.core.num_pars-2) 
    newUb     = zeros(model.core.num_pars-2)
    initGuess = zeros(model.core.num_pars-2)

    boundsmapping2d!(newLb, model.core.θlb, ind1, ind2)
    boundsmapping2d!(newUb, model.core.θub, ind1, ind2)
    boundsmapping2d!(initGuess, model.core.θmle, ind1, ind2)

    θranges, λranges = variablemapping2dranges(model.core.num_pars, ind1, ind2)

    return newLb, newUb, initGuess, θranges, λranges
end

function get_bivariate_opt_func(profile_type::Symbol, method::AbstractBivariateMethod)
    if method isa BracketingMethodFix1Axis
        if profile_type == :EllipseApproxAnalytical
            return bivariateΨ_ellipse_analytical
        elseif profile_type == :LogLikelihood || profile_type == :EllipseApprox
            return bivariateΨ
        end

    elseif method isa BracketingMethodRadial || method isa BracketingMethodSimultaneous
        if profile_type == :EllipseApproxAnalytical
            return bivariateΨ_ellipse_analytical_vectorsearch
        elseif profile_type == :LogLikelihood || profile_type == :EllipseApprox
            return bivariateΨ_vectorsearch
        end
    end

    return (missing)
end

function bivariate_confidenceprofile_fix1axis(bivariate_optimiser::Function, model::LikelihoodModel, num_points::Int, consistent::NamedTuple, ind1::Int, ind2::Int)

    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)
    g(x,p) = bivariate_optimiser(x,p)[1]

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateΨ_ellipse_analytical

    if biv_opt_is_ellipse_analytical
        boundarySamples = zeros(2, num_points)
    else
        boundarySamples = zeros(model.core.num_pars, num_points)
    end

    count=0
    for (i, j, N) in [[ind1, ind2, div(num_points,2)], [ind2, ind1, (div(num_points,2) + rem(num_points,2))]]

        ϵ=(model.core.θub[i]-model.core.θlb[i])/10^6
        indexesSorted = i < j

        for k in 1:N
            count +=1
            Ψ_x, Ψ_y0, Ψ_y1 = 0.0, 0.0, 0.0

            # do-while loop
            while true
                Ψ_x = rand(Uniform(lb[i], ub[i]))
                Ψ_y0 = rand(Uniform(lb[j], ub[j]))
                Ψ_y1 = rand(Uniform(lb[j], ub[j]))

                p0=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, Ψ_x=Ψ_x,
                    θranges=θranges, λranges=λranges, consistent=consistent)

                (( g(Ψ_y0, p0) * g(Ψ_y1, p0) ) ≥ 0) || break
            end

            p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, Ψ_x=Ψ_x,
                θranges=θranges, λranges=λranges, consistent=consistent)

            # println(count)

            Ψ_y1 = find_zero(g, (Ψ_y0, Ψ_y1), atol=ϵ, Roots.Brent(); p=p)


            if biv_opt_is_ellipse_analytical
                if indexesSorted
                    boundarySamples[:, count] .= p[:Ψ_x], Ψ_y1
                else
                    boundarySamples[:, count] .= Ψ_y1, p[:Ψ_x]
                end
            else
                boundarySamples[i, count] = p[:Ψ_x]
                boundarySamples[j, count] = Ψ_y1

                variablemapping2d!(@view(boundarySamples[:, count]), bivariate_optimiser(Ψ_y1, p)[2], θranges, λranges)
            end
        end
    end
    return boundarySamples
end

function bivariate_confidenceprofile_vectorsearch(bivariate_optimiser::Function, model::LikelihoodModel, num_points::Int, consistent::NamedTuple, ind1::Int, ind2::Int; num_radial_directions::Int=0)


    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)
    g(x,p) = bivariate_optimiser(x,p)[1]

    boundarySamples = zeros(model.core.num_pars, num_points)

    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]

    p0=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                θranges=θranges, λranges=λranges, consistent=consistent)

    if num_radial_directions == 0
        insidePoints, outsidePoints = findNpointpairs_simultaneous(g, model, p0, num_points, ind1, ind2)
    else
        insidePoints, outsidePoints = findNpointpairs_radial(g, model, p0, num_points, num_radial_directions, ind1, ind2)
    end

    for i in 1:num_points
        pointa .= insidePoints[:,i]
        v_bar = outsidePoints[:,i] - insidePoints[:,i]

        v_bar_norm = norm(v_bar, 2)
        uhat .= v_bar / v_bar_norm

        p=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                θranges=θranges, λranges=λranges, consistent=consistent)

        ϵ=v_bar_norm/10^6

        Ψ_y1 = find_zero(g, (0.0, v_bar_norm), atol=ϵ, Roots.Brent(); p=p)
        
        boundarySamples[[ind1, ind2], i] .= pointa + Ψ_y1*uhat
        variablemapping2d!(@view(boundarySamples[:, i]), bivariate_optimiser(Ψ_y1, p)[2], θranges, λranges)
    end

    return boundarySamples
end


# num_points is the number of points to compute for a given method, that are on the boundary and/or inside the boundary.
function bivariate_confidenceprofiles(model::LikelihoodModel, θcombinations::Vector{Vector{Int64}}, num_points::Int; 
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood,
    method::AbstractBivariateMethod=BracketingMethodFix1Axis())

    valid_profile_type = profile_type in [:EllipseApprox, :EllipseApproxAnalytical, :LogLikelihood]
    @assert valid_profile_type "Specified `profile_type` is invalid. Allowed values are :EllipseApprox, :EllipseApproxAnalytical, :LogLikelihood."

    @assert method isa AbstractBivariateMethod "Specified `method` is not a AbstractBivariateMethod. Allowed methods are BracketingMethodFix1Axis, BracketingMethodSimultaneous and BracketingMethodRadial."

    if profile_type in [:EllipseApprox, :EllipseApproxAnalytical]
        check_ellipse_approx_exists!(model)
    end

    bivariate_optimiser = get_bivariate_opt_func(profile_type, method)

    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Tuple{Symbol,Symbol}, BivariateConfidenceStruct}()

    for (ind1, ind2) in θcombinations

        if ind2 < ind1
            ind1, ind2 = ind2, ind1
        end
            
        if method isa BracketingMethodFix1Axis
            boundarySamples =  bivariate_confidenceprofile_fix1axis(
                        bivariate_optimiser, model, 
                        num_points, consistent, ind1, ind2)
            
        elseif method isa BracketingMethodSimultaneous
            boundarySamples = bivariate_confidenceprofile_vectorsearch(
                        bivariate_optimiser, model, 
                        num_points, consistent, ind1, ind2)
        elseif method isa BracketingMethodRadial
            boundarySamples = bivariate_confidenceprofile_vectorsearch(
                        bivariate_optimiser, model, 
                        num_points, consistent, ind1, ind2, num_radial_directions=method.num_radial_directions)
        elseif method isa ContinuationMethod
            boundarySamples = bivariate_confidenceprofile_continuation(
                        bivariate_optimiser, model, 
                        num_points, consistent, ind1, ind2)
        end
        
        confidenceDict[(model.core.θnames[ind1], model.core.θnames[ind2])] = BivariateConfidenceStruct((model.core.θmle[ind1], model.core.θmle[ind2]), boundarySamples, model.core.θlb[[ind1, ind2]], model.core.θub[[ind1, ind2]])
    end

    return confidenceDict
end

# profile just provided θcombinations_symbols
function bivariate_confidenceprofiles(model::LikelihoodModel, θcombinations_symbols::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}}, num_points::Int;
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood,
    method::AbstractBivariateMethod=BracketingMethodFix1Axis())

    θcombinations = convertθnames_toindices(model, θcombinations_symbols)

    return bivariate_confidenceprofiles(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method)
end

# profile m random combinations of parameters (sampling without replacement), where 0 < m ≤ binomial(model.core.num_pars,2)
function bivariate_confidenceprofiles(model::LikelihoodModel, profile_m_random_combinations::Int, num_points::Int;
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood,
    method::AbstractBivariateMethod=BracketingMethodFix1Axis())

    profile_m_random_combinations = max(0, min(profile_m_random_combinations, binomial(model.core.num_pars,2)))

    if profile_m_random_combinations == 0
        @error "`profile_m_random_combinations` must be a strictly positive integer."
        return nothing
    end

    θcombinations = sample(collect(combinations(1:model.core.num_pars, 2)), profile_m_random_combinations, replace=false)

    return bivariate_confidenceprofiles(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method)
end

# profile all combinations
function bivariate_confidenceprofiles(model::LikelihoodModel, num_points::Int; 
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood,
    method::AbstractBivariateMethod=BracketingMethodFix1Axis())

    θcombinations = collect(combinations(1:model.core.num_pars, 2))

    return bivariate_confidenceprofiles(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            method=method)
end

