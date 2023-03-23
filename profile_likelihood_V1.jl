# Dictionaries used in these functions are intended to be replaced with (mutable) structs,
# which will have better performance when reading and writing

function normaliseduhat(v_bar); v_bar/norm(vbar, 2) end

function generatepoint(lb::Vector{<:Float64}, ub::Vector{<:Float64}, ind1::Int, ind2::Int)
    return rand(Uniform(lb[ind1], ub[ind1])), rand(Uniform(lb[ind2], ub[ind2]))
end

# function findNpointpairs(p, N, lb, ub, ind1, ind2; maxIters)
function findNpointpairs_vectorsearch(p, N, lb, ub, ind1, ind2)

    insidePoints  = zeros(2,N)
    outsidePoints = zeros(2,N)

    Ninside = 0; Noutside=0
    iters=0
    while Noutside<N && Ninside<N

        x, y = generatepoint(lb, ub, ind1, ind2)
        if bivariateΨ((x,y), p)[1] > 0
            Ninside+=1
            insidePoints[:,Ninside] .= [x,y]
        else
            Noutside+=1
            outsidePoints[:,Noutside] .= [x,y]
        end
        iters+=1
        println(iters)
    end

    # while Ninside < N && iters < maxIters
    while Ninside < N
        x, y = generatepoint(lb, ub, ind1, ind2)
        if bivariateΨ((x,y), p)[1] > 0
            Ninside+=1
            insidePoints[:,Ninside] .= [x,y]
        end
        iters+=1
        println(iters)
    end

    # while Noutside < N && iters < maxIters
    while Noutside < N
        x, y = generatepoint(lb, ub, ind1, ind2)
        if bivariateΨ((x,y), p)[1] < 0
            Noutside+=1
            outsidePoints[:,Noutside] .= [x,y]
        end
        iters+=1
        println(iters)
    end

    return insidePoints, outsidePoints
end


# function univariateprofiles_constrained(likelihoodFunc, data, fmle, θnames, θmle, lb, ub; confLevel=0.95)

#     df = 1
#     llstar = -quantile(Chisq(df), confLevel)/2
#     num_vars = length(θnames)

#     m = Model(Ipopt.Optimizer)
#     set_silent(m)

#     function b(θ...); likelihoodFunc(data[1], θ, data[2])-fmle end
#     register(model, :my_obj, num_vars, b; autodiff = true)
#     @variable(model, θ[i=1:3], lower_bound=lb[i], upper_bound=ub[i], start=θmle[i])
#     @NLobjective(model, Max, my_obj(θ...))

#     confidenceDict = Dict{Symbol, UnivariateConfidenceStruct}()

#     for (i, θname) in enumerate(θnames)

#         if is_valid(m, :myConstraint)
#             delete(m, :myConstraint)
#             unregister(m, :myConstraint)
#         end

#         function myConstraint(θ...); θ[i] end
#         register(model, :myConstraint, length(num_vars), myConstraint; autodiff = true)
#         @NLconstraint(model, myConstraint(θ...)==θmle[i])

#         function univariateΨ(Ψ)
#             set_normalized_rhs(:myConstraint, Ψ)
#             JuMP.optimize!(model)
#             return objective_value(model)            
#         end

#         interval=zeros(2)

#         interval[1] = find_zero(univariateΨ, (lb[i], θmle[i]), atol=ϵ, Roots.Brent())
#         interval[2] = find_zero(univariateΨ, (θmle[i], ub[i]), atol=ϵ, Roots.Brent())

#         confidenceDict[θname] = UnivariateConfidenceStruct(θmle[i], interval, [lb[i], ub[i]])
#     end

#     return confidenceDict
# end

function variablemapping1dranges(num_vars::T, index::T) where T <: Int
    θranges = (1:(index-1), (index+1):num_vars)
    λranges = (1:(index-1), index:(num_vars-1))
    return θranges, λranges
end

function variablemapping2dranges(num_vars::T, index1::T, index2::T) where T <: Int
    θranges = (1:(index1-1), (index1+1):(index2-1), (index2+1):num_vars)
    λranges = (1:(index1-1), index1:(index2-2), (index2-1):(num_vars-2) )
    return θranges, λranges
end

# function variablemapping1d!(θ, λ, index)
#     θ[1:(index-1)]   .= λ[1:(index-1)]
#     θ[(index+1):end] .= λ[index:end]
#     return θ
# end

function variablemapping1d!(θ, λ, θranges, λranges)
    θ[θranges[1]] .= @view(λ[λranges[1]])
    θ[θranges[2]] .= @view(λ[λranges[2]])
    return θ
end

function variablemapping2d!(θ, λ, θranges, λranges)
    θ[θranges[1]] .= @view(λ[λranges[1]])
    θ[θranges[2]] .= @view(λ[λranges[2]])
    θ[θranges[3]] .= @view(λ[λranges[3]])
    return θ
end

function boundsmapping1d!(newbounds::Vector{<:Float64}, bounds::Vector{<:Float64}, index::Int)
    newbounds[1:(index-1)] .= @view(bounds[1:(index-1)])
    newbounds[index:end]   .= @view(bounds[(index+1):end])
    return nothing
end

# we know index1 < index2 by construction. If index1 and index2 are user provided, enforce this relationship 
function boundsmapping2d!(newbounds::Vector{<:Float64}, bounds::Vector{<:Float64}, index1::Int, index2::Int)
    newbounds[1:(index1-1)]      .= @view(bounds[1:(index1-1)])
    newbounds[index1:(index2-2)] .= @view(bounds[(index1+1):(index2-1)])
    newbounds[(index2-1):end]    .= @view(bounds[(index2+1):end])
    return nothing
end

# THIS VERSION COULD BE USED if the likelihood function is not using a reparameterisation (i.e. overwriting values of
# θs when it is called) - perhaps make other the default, and let the user specify that a reparameterisation is not being
# used and/or that the reparameterisation does not edit thetas (θs). Should be slightly more performant.
function univariateΨ_unsafe(Ψ, p)
    θs=zeros(p[:num_vars])
    θs[p[:ind]] = Ψ
    
    function fun(λ); return p[:likelihoodFunc](variablemapping1d!(θs, λ, p[:θranges], p[:λranges]), p[:data]) end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function univariateΨ(Ψ, p)
    θs=zeros(p[:num_vars])

    function fun(λ)
        θs[p[:ind]] = Ψ
        return p[:likelihoodFunc](variablemapping1d!(θs, λ, p[:θranges], p[:λranges]), p[:data]) 
    end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function univariateΨ_ellipse_unsafe(Ψ, p)
    θs=zeros(p[:num_vars])
    θs[p[:ind]] = Ψ

    function fun(λ); return ellipse_loglike(variablemapping1d!(θs, λ, p[:θranges], p[:λranges]), p[:θmle], p[:H]) end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function univariateΨ_ellipse(Ψ, p)
    θs=zeros(p[:num_vars])
    
    function fun(λ)
        θs[p[:ind]] = Ψ
        return ellipse_loglike(variablemapping1d!(θs, λ, p[:θranges], p[:λranges]), p[:θmle], p[:H]) 
    end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function univariateΨ_ellipse_analytical(Ψ, p)
    return analytic_ellipse_loglike([Ψ], p[:ind], p[:θmle], p[:Γ]) - p[:targetll]
end


function bivariateΨ_unsafe(Ψ, p)
    θs=zeros(p[:num_vars])
    θs[p[:ind1]] = p[:Ψ_x]
    θs[p[:ind2]] = Ψ
    
    function fun(λ); return p[:likelihoodFunc](variablemapping2d!(θs, λ, p[:θranges], p[:λranges]), p[:data]) end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function bivariateΨ(Ψ::Real, p)
    θs=zeros(p[:num_vars])
    
    function fun(λ)
        θs[p[:ind1]] = p[:Ψ_x]
        θs[p[:ind2]] = Ψ
        return p[:likelihoodFunc](variablemapping2d!(θs, λ, p[:θranges], p[:λranges]), p[:data]) 
    end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function bivariateΨ(Ψ::Tuple{Real,Real}, p)
    θs=zeros(p[:num_vars])
    
    function fun(λ)
        θs[p[:ind1]] = Ψ[1]
        θs[p[:ind2]] = Ψ[2]
        return p[:likelihoodFunc](variablemapping2d!(θs, λ, p[:θranges], p[:λranges]), p[:data]) 
    end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function bivariateΨ_vectorsearch_unsafe(Ψ, p)
    θs=zeros(p[:num_vars])
    θs[p[:ind1]], θs[p[:ind2]] = p[:pointa] + Ψ*p[:uhat]
    
    function fun(λ); return p[:likelihoodFunc](variablemapping2d!(θs, λ, p[:θranges], p[:λranges]), p[:data]) end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function bivariateΨ_vectorsearch(Ψ, p)
    θs=zeros(p[:num_vars])
    Ψxy = p[:pointa] + Ψ*p[:uhat]
    
    function fun(λ)
        θs[p[:ind1]], θs[p[:ind2]] = Ψxy
        return p[:likelihoodFunc](variablemapping2d!(θs, λ, p[:θranges], p[:λranges]), p[:data]) 
    end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function univariateΨrelationship(Ψ, p)
    θs=zeros(p[:num_vars])
    # θs[p[:ind]] = Ψ
    
    function fun(λ)
        resolve_relationship!(θs, Ψ, λ, p[:ind], p[:λind], p[:relationship])
        return p[:likelihoodFunc](variablemapping1d!(θs, λ, p[:θranges], p[:λranges]), p[:data]) 
    end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end



function univariateΨ_ellipse_relationship(Ψ, p)
    θs=zeros(p[:num_vars])
    # θs[p[:ind]] = Ψ

    function fun(λ)
        resolve_relationship!(θs, Ψ, λ, p[:ind], p[:λind], p[:relationship])
        return ellipse_loglike(variablemapping1d!(θs, λ, p[:θranges], p[:λranges]), p[:θmle], p[:H]) 
    end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end



function univariateprofiles_ellipse_analytical(θnames, θmle, lb, ub, H, Γ; confLevel=0.95)

    df = 1
    llstar = -quantile(Chisq(df), confLevel)/2
    num_vars = length(θnames)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Symbol, UnivariateConfidenceStruct}()
    p = Dict(:ind=>[1], 
            :θmle=>θmle,
            :H=>H,
            :Γ=>Γ,
            :num_vars=>num_vars,
            :targetll=>(llstar))

    for (i, θname) in enumerate(θnames)
        interval = zeros(2)

        p[:ind]=[i]
        
        ϵ=(ub[i]-lb[i])/10^6
    
        g(x,p) = univariateΨ_ellipse_analytical(x,p)

        # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
        interval[1] = g(lb[i], p) <= 0.0 ? find_zero(g, (lb[i], θmle[i]), atol=ϵ, Roots.Brent(), p=p) : NaN
        interval[2] = g(ub[i], p) <= 0.0 ? find_zero(g, (θmle[i], ub[i]), atol=ϵ, Roots.Brent(), p=p) : NaN

        confidenceDict[θname] = UnivariateConfidenceStruct(θmle[i], interval, lb[i], ub[i], confLevel)
    end

    return confidenceDict, p
end

function univariateprofile_ellipse_providedrelationship(relationship::AbstractRelationship, θnames, θmle, lb, ub, H, Γ; confLevel=0.95)

    df = 1
    llstar = -quantile(Chisq(df), confLevel)/2
    num_vars = length(θnames)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Symbol, UnivariateConfidenceStruct}()
    p = Dict(:ind=>1,   
            :θmle=>θmle,
            :H=>H,
            :Γ=>Γ,
            :newLb=>zeros(num_vars-1), 
            :newUb=>zeros(num_vars-1),
            :initGuess=>zeros(num_vars-1),
            :num_vars=>num_vars,
            :targetll=>(llstar),
            :θranges=>(0:0, 0:0),
            :λranges=>(0:0, 0:0),
            :relationship=>relationship)

    interval = zeros(2)

    p[:ind] = findfirst(isequal(relationship.a), θnames)
    indexB  = findfirst(isequal(relationship.b), θnames)
    p[:λind] = p[:ind] > indexB ? indexB : indexB-1 

    Ψmle = Ψmlegivenrelationship(θmle, p[:ind], indexB, relationship)

    boundsmapping1d!(p[:newLb], lb, p[:ind])
    boundsmapping1d!(p[:newUb], ub, p[:ind])
    boundsmapping1d!(p[:initGuess], θmle, p[:ind])
    p[:θranges], p[:λranges] = variablemapping1dranges(p[:num_vars], p[:ind])
    
    ϵ=(relationship.ub-relationship.lb)/10^6

    g(x,p) = univariateΨ_ellipse_relationship(x,p)[1]

    # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
    interval[1] = g(relationship.lb, p) <= 0.0 ? find_zero(g, (relationship.lb, Ψmle), atol=ϵ, Roots.Brent(), p=p) : NaN
    interval[2] = g(relationship.ub, p) <= 0.0 ? find_zero(g, (Ψmle, relationship.ub), atol=ϵ, Roots.Brent(), p=p) : NaN

    confidenceDict[relationship.name] = UnivariateConfidenceStruct(Ψmle, interval, relationship.lb, relationship.ub, confLevel)
    

    return confidenceDict, p
end

function univariateprofiles_ellipse(θnames, θmle, lb, ub, H, Γ; confLevel=0.95)

    df = 1
    llstar = -quantile(Chisq(df), confLevel)/2
    num_vars = length(θnames)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Symbol, UnivariateConfidenceStruct}()
    p = Dict(:ind=>1,   
            :θmle=>θmle,
            :H=>H,
            :Γ=>Γ,
            :newLb=>zeros(num_vars-1), 
            :newUb=>zeros(num_vars-1),
            :initGuess=>zeros(num_vars-1),
            :num_vars=>num_vars,
            :targetll=>(llstar),
            :θranges=>(0:0, 0:0),
            :λranges=>(0:0, 0:0))

    for (i, θname) in enumerate(θnames)
        interval = zeros(2)

        p[:ind]=i
        boundsmapping1d!(p[:newLb], lb, i)
        boundsmapping1d!(p[:newUb], ub, i)
        boundsmapping1d!(p[:initGuess], θmle, i)
        p[:θranges], p[:λranges] = variablemapping1dranges(p[:num_vars], p[:ind])
        
        ϵ=(ub[i]-lb[i])/10^6
    
        g(x,p) = univariateΨ_ellipse(x,p)[1]

        # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
        interval[1] = g(lb[i], p) <= 0.0 ? find_zero(g, (lb[i], θmle[i]), atol=ϵ, Roots.Brent(), p=p) : NaN
        interval[2] = g(ub[i], p) <= 0.0 ? find_zero(g, (θmle[i], ub[i]), atol=ϵ, Roots.Brent(), p=p) : NaN

        confidenceDict[θname] = UnivariateConfidenceStruct(θmle[i], interval, lb[i], ub[i], confLevel)
    end

    return confidenceDict, p
end

# ellipse profile wise method should work for a provided relationship too
function univariateprofile_providedrelationship(relationship::AbstractRelationship, likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)

    df = 1
    llstar = -quantile(Chisq(df), confLevel)/2
    num_vars = length(θnames)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Symbol, UnivariateConfidenceStruct}()
    p = Dict(:ind=>1, 
            :λind=>1,
            :data=>data, 
            :newLb=>zeros(num_vars-1), 
            :newUb=>zeros(num_vars-1),
            :initGuess=>zeros(num_vars-1),
            :num_vars=>num_vars,
            :targetll=>(fmle+llstar),
            :likelihoodFunc=>likelihoodFunc,
            :θranges=>(0:0, 0:0),
            :λranges=>(0:0, 0:0),
            :relationship=>relationship)

    interval = zeros(2)

    p[:ind] = findfirst(isequal(relationship.a), θnames)
    indexB  = findfirst(isequal(relationship.b), θnames)
    p[:λind] = p[:ind] > indexB ? indexB : indexB-1 

    Ψmle = Ψmlegivenrelationship(θmle, p[:ind], indexB, relationship)

    boundsmapping1d!(p[:newLb], lb, p[:ind])
    boundsmapping1d!(p[:newUb], ub, p[:ind])
    boundsmapping1d!(p[:initGuess], θmle, p[:ind])
    p[:θranges], p[:λranges] = variablemapping1dranges(p[:num_vars], p[:ind])
    
    ϵ=(relationship.ub-relationship.lb)/10^6

    g(x,p) = univariateΨrelationship(x,p)[1]

    # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
    interval[1] = g(relationship.lb, p) <= 0.0 ? find_zero(g, (relationship.lb, Ψmle), atol=ϵ, Roots.Brent(), p=p) : NaN
    interval[2] = g(relationship.ub, p) <= 0.0 ? find_zero(g, (Ψmle, relationship.ub), atol=ϵ, Roots.Brent(), p=p) : NaN

    confidenceDict[relationship.name] = UnivariateConfidenceStruct(Ψmle, interval, relationship.lb, relationship.ub, confLevel)

    # a = [g(x,p) for x in LinRange(relationship.lb, relationship.ub, 10)]
    # return confidenceDict, p, a

    return confidenceDict, p
end

function univariateprofiles(likelihoodFunc, fmle, data, θnames, θmle, lb, ub; confLevel=0.95)

    df = 1
    llstar = -quantile(Chisq(df), confLevel)/2
    num_vars = length(θnames)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Symbol, UnivariateConfidenceStruct}()
    p = Dict(:ind=>1, 
            :data=>data, 
            :newLb=>zeros(num_vars-1), 
            :newUb=>zeros(num_vars-1),
            :initGuess=>zeros(num_vars-1),
            :num_vars=>num_vars,
            :targetll=>(fmle+llstar),
            :likelihoodFunc=>likelihoodFunc,
            :θranges=>(0:0, 0:0),
            :λranges=>(0:0, 0:0))

    for (i, θname) in enumerate(θnames)
        interval = zeros(2)
        p[:ind]=i

        boundsmapping1d!(p[:newLb], lb, i)
        boundsmapping1d!(p[:newUb], ub, i)
        boundsmapping1d!(p[:initGuess], θmle, i)
        p[:θranges], p[:λranges] = variablemapping1dranges(p[:num_vars], p[:ind])

        ϵ=(ub[i]-lb[i])/10^6
    
        g(x,p) = univariateΨ(x,p)[1]

        # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
        interval[1] = g(lb[i], p) <= 0.0 ? find_zero(g, (lb[i], θmle[i]), atol=ϵ, Roots.Brent(), p=p) : NaN
        interval[2] = g(ub[i], p) <= 0.0 ? find_zero(g, (θmle[i], ub[i]), atol=ϵ, Roots.Brent(), p=p) : NaN

        confidenceDict[θname] = UnivariateConfidenceStruct(θmle[i], interval, lb[i], ub[i], confLevel)
    end

    return confidenceDict, p
end

function bivariateprofiles(likelihoodFunc, fmle, data, θnames, θmle, lb, ub, num_points::Int; confLevel::Float64=0.95, method=(:Brent,:fix1axis))

    df = 2
    llstar = -quantile(Chisq(df), confLevel)/2
    num_vars = length(θnames)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Tuple{Symbol,Symbol}, BivariateConfidenceStruct}()
    p = Dict(:ind1=>1,
            :ind2=>2, 
            :data=>data, 
            :newLb=>zeros(num_vars-2), 
            :newUb=>zeros(num_vars-2),
            :initGuess=>zeros(num_vars-2),
            :num_vars=>num_vars,
            :targetll=>(fmle+llstar),
            :likelihoodFunc=>likelihoodFunc,
            :θranges=>(0:0, 0:0, 0:0),
            :λranges=>(0:0, 0:0, 0:0),
            :Ψ_x=>0.0)

    combinationsOfInterest = combinations(1:num_vars, 2)

    if method[1] == :Brent && method[2]==:fix1axis
        g(x,p) = bivariateΨ(x,p)[1]
    else
        g(x,p) = bivariateΨ_vectorsearch(x,p)[1]
    end


    for (ind1, ind2) in combinationsOfInterest

        println(ind1)
        println(ind2)

        boundsmapping2d!(p[:newLb], lb, ind1, ind2)
        boundsmapping2d!(p[:newUb], ub, ind1, ind2)
        boundsmapping2d!(p[:initGuess], θmle, ind1, ind2)
        p[:θranges], p[:λranges] = variablemapping2dranges(p[:num_vars], ind1, ind2)

        boundarySamples = zeros(num_vars, 2*num_points)

        
        # find 2*num_points using some procedure 
        
        if method[1] == :Brent
            
            if method[2] == :fix1axis
                
                count=0
                for (i, j, N) in [[ind1, ind2, div(num_points,2)], [ind2, ind1, num_points]]

                    p[:ind1]=i
                    p[:ind2]=j
                    ϵ=(ub[i]-lb[i])/10^6

                    while count < N

                        p[:Ψ_x] = rand(Uniform(lb[i], ub[i]))
                        Ψ_y0 = rand(Uniform(lb[j], ub[j]))
                        Ψ_y1 = rand(Uniform(lb[j], ub[j]))
                        
                        if ( g(Ψ_y0, p) * g(Ψ_y1, p) ) < 0 

                            count+=1
                            println(count)

                            Ψ_y1 = find_zero(g, (Ψ_y0, Ψ_y1), atol=ϵ, Roots.Brent(); p=p)

                            boundarySamples[i, count] = p[:Ψ_x]
                            boundarySamples[j, count] = Ψ_y1

                            variablemapping2d!(@view(boundarySamples[:, count]), bivariateΨ(Ψ_y1, p)[2], p[:θranges], p[:λranges])
                        end
                    end
                end

            elseif method[2] == :vectorsearch

                insidePoints, outsidePoints = findNpointpairs_vectorsearch(p, num_points, lb, ub, ind1, ind2)

                p[:pointa] = [0.0,0.0]
                p[:uhat]   = [0.0,0.0]

                for i in 1:num_points
                    p[:pointa] .= insidePoints[:,i]
                    v_bar = outsidePoints[:,i] - insidePoints[:,i]

                    v_bar_norm = norm(v_bar, 2)
                    p[:uhat] = v_bar / v_bar_norm

                    ϵ=v_bar_norm/10^6

                    Ψ_y1 = find_zero(g, (0.0, v_bar_norm), atol=ϵ, Roots.Brent(); p=p)
                    
                    boundarySamples[[ind1, ind2], i] .= p[:pointa] + Ψ_y1*p[:uhat]
                    variablemapping2d!(@view(boundarySamples[:, i]), bivariateΨ_vectorsearch(Ψ_y1, p)[2], p[:θranges], p[:λranges])
                end
            end
        end
        
        confidenceDict[(θnames[ind1], θnames[ind2])] = BivariateConfidenceStruct((θmle[ind1],θmle[ind2]), boundarySamples, lb[[ind1, ind2]], ub[[ind1, ind2]], confLevel)
    end

    return confidenceDict, p
end