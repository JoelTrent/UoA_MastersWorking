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
function findNpointpairs_vectorsearch(g::Function, model::LikelihoodModel, p::NamedTuple, num_points::Int, ind1::Int, ind2::Int)

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

function bivariateΨ_unsafe(Ψ, p)
    θs=zeros(p.consistent.num_pars)
    θs[p.ind1] = p.Ψ_x
    θs[p.ind2] = Ψ
    
    function fun(λ); return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data) end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    return llb, xopt
end

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

function bivariateΨ_vectorsearch_unsafe(Ψ, p)
    θs=zeros(p.consistent.num_pars)
    θs[p.ind1], θs[p.ind2] = p.pointa + Ψ*p.uhat
    
    function fun(λ); return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data) end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    return llb, xopt
end

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

function get_bivariate_opt_func(profile_type::Symbol, use_unsafe_optimiser::Bool, method)


    if method[1]==:Bracketing
        if method[2]==:Fix1Axis
            if profile_type == :EllipseApproxAnalytical
                return bivariateΨ_ellipse_analytical
            elseif profile_type == :LogLikelihood || profile_type == :EllipseApprox
                if use_unsafe_optimiser 
                    return bivariateΨ_unsafe
                else
                    return bivariateΨ
                end
            end

        elseif method[2]==:VectorSearch
            if profile_type == :EllipseApproxAnalytical
                return bivariateΨ_ellipse_analytical_vectorsearch
            elseif profile_type == :LogLikelihood || profile_type == :EllipseApprox
                if use_unsafe_optimiser 
                    return bivariateΨ_vectorsearch_unsafe
                else
                    return bivariateΨ_vectorsearch
                end
            end
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

            println(count)

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

function bivariate_confidenceprofile_vectorsearch(bivariate_optimiser::Function, model::LikelihoodModel, num_points::Int, consistent::NamedTuple, ind1::Int, ind2::Int)


    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)
    g(x,p) = bivariate_optimiser(x,p)[1]

    boundarySamples = zeros(model.core.num_pars, num_points)

    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]

    p0=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                θranges=θranges, λranges=λranges, consistent=consistent)

    insidePoints, outsidePoints = findNpointpairs_vectorsearch(g, model, p0, num_points, ind1, ind2)

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
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood, use_unsafe_optimiser::Bool=false,
     method=(:Bracketing, :Fix1Axis))



    valid_profile_type = profile_type in [:EllipseApprox, :EllipseApproxAnalytical, :LogLikelihood]
    @assert valid_profile_type "Specified `profile_type` is invalid. Allowed values are :EllipseApprox, :EllipseApproxAnalytical, :LogLikelihood."

    valid_method = method[1] in [:Bracketing] && method[2] in [:Fix1Axis, :VectorSearch]

    @assert valid_method "Specified `method` is invalid. Allowed values are (:Bracketing, Fix1Axis), (:Bracketing, :VectorSearch)."


    if profile_type in [:EllipseApprox, :EllipseApproxAnalytical]
        check_ellipse_approx_exists!(model)
    end

    bivariate_optimiser = get_bivariate_opt_func(profile_type, use_unsafe_optimiser, method)

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

        if method[1] == :Bracketing
            
            if method[2] == :Fix1Axis
                boundarySamples =  bivariate_confidenceprofile_fix1axis(
                            bivariate_optimiser, model, 
                            num_points, consistent, ind1, ind2)
                

            elseif method[2] == :VectorSearch
                boundarySamples = bivariate_confidenceprofile_vectorsearch(
                            bivariate_optimiser, model, 
                            num_points, consistent, ind1, ind2)
            end
        end
        
        confidenceDict[(model.core.θnames[ind1], model.core.θnames[ind2])] = BivariateConfidenceStruct((model.core.θmle[ind1], model.core.θmle[ind2]), boundarySamples, model.core.θlb[[ind1, ind2]], model.core.θub[[ind1, ind2]])
    end

    return confidenceDict
end


function bivariate_confidenceprofiles(model::LikelihoodModel, num_points::Int; 
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood, use_unsafe_optimiser::Bool=false,
     method=(:Bracketing,:Fix1Axis))


    θcombinations = collect(combinations(1:model.core.num_pars, 2))

    return bivariate_confidenceprofiles(model, θcombinations, num_points, 
            confidence_level=confidence_level, profile_type=profile_type, 
            use_unsafe_optimiser=use_unsafe_optimiser, method=method)

end