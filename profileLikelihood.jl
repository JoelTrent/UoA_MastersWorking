# Section 7: Unconstrained numerical optimisation 
function optimise(fun, θ₀, lb, ub;
    dv = false,
    method = dv ? :LD_LBFGS : :LN_BOBYQA,
    )

    if dv || String(method)[2] == 'D'
        tomax = fun
    else
        tomax = (θ,∂θ) -> fun(θ)
    end

    opt = Opt(method,length(θ₀))
    opt.max_objective = tomax
    opt.lower_bounds = lb       # Lower bound
    opt.upper_bounds = ub       # Upper bound
    opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
    res = optimize(opt, θ₀)
    return res[[2,1]]
end

struct ConfidenceStruct
    mle::Float64
    confidence_interval::Vector{<:Float64}
    bounds::Vector{<:Float64}
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

#     confidenceDict = Dict{Symbol, ConfidenceStruct}()

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

#         confidenceDict[θname] = ConfidenceStruct(θmle[i], interval, [lb[i], ub[i]])
#     end

#     return confidenceDict
# end

function variablemapping1dranges(num_vars, index)
    θranges = (1:(index-1), (index+1):num_vars)
    λranges = (1:(index-1), index:(num_vars-1))
    return θranges, λranges
end

# function variablemapping1d!(θ, λ, index)
#     θ[1:(index-1)]   .= λ[1:(index-1)]
#     θ[(index+1):end] .= λ[index:end]
#     return θ
# end

function variablemapping1d!(θ, λ, θranges, λranges)
    θ[θranges[1]] .= λ[λranges[1]]
    θ[θranges[2]] .= λ[λranges[2]]
    return θ
end

function boundsmapping1d!(newbounds::Vector{<:Float64}, bounds::Vector{<:Float64}, index::Int)
    newbounds[1:(index-1)] .= bounds[1:(index-1)]
    newbounds[index:end]   .= bounds[(index+1):end]
    return nothing
end

function univariateΨ(Ψ, p)
    θs=zeros(p[:num_vars])
    θs[p[:ind]] = Ψ
    θranges, λranges = variablemapping1dranges(p[:num_vars], p[:ind])

    function fun(λ); return p[:likelihoodFunc](p[:data], variablemapping1d!(θs, λ, θranges, λranges) ) end

    (xopt,fopt)=optimise(fun, p[:initGuess], p[:newLb], p[:newUb])
    llb=fopt-p[:targetll]
    return llb, xopt
end

function univariateΨ_ellipse_analytical(Ψ, p)
    return analytic_ellipse_loglike([Ψ], p[:ind], p[:θmle], p[:Γ]) - p[:targetll]
end

function univariateΨ_ellipse(Ψ, p)
    θs=zeros(p[:num_vars])
    θs[p[:ind]] = Ψ
    θranges, λranges = variablemapping1dranges(p[:num_vars], p[:ind])

    function fun(λ); return ellipse_loglike(variablemapping1d!(θs, λ, θranges, λranges), p[:θmle], p[:H]) end

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

    confidenceDict = Dict{Symbol, ConfidenceStruct}()
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

        confidenceDict[θname] = ConfidenceStruct(θmle[i], interval, [lb[i], ub[i]])
    end

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

    confidenceDict = Dict{Symbol, ConfidenceStruct}()
    p = Dict(:ind=>1,   
            :θmle=>θmle,
            :H=>H,
            :Γ=>Γ,
            :newLb=>zeros(num_vars-1), 
            :newUb=>zeros(num_vars-1),
            :initGuess=>zeros(num_vars-1),
            :num_vars=>num_vars,
            :targetll=>(llstar))

    for (i, θname) in enumerate(θnames)
        interval = zeros(2)

        boundsmapping1d!(p[:newLb], lb, i)
        boundsmapping1d!(p[:newUb], ub, i)
        boundsmapping1d!(p[:initGuess], θmle, i)
        p[:ind]=i
        
        ϵ=(ub[i]-lb[i])/10^6
    
        g(x,p) = univariateΨ_ellipse(x,p)[1]

        # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
        interval[1] = g(lb[i], p) <= 0.0 ? find_zero(g, (lb[i], θmle[i]), atol=ϵ, Roots.Brent(), p=p) : NaN
        interval[2] = g(ub[i], p) <= 0.0 ? find_zero(g, (θmle[i], ub[i]), atol=ϵ, Roots.Brent(), p=p) : NaN

        confidenceDict[θname] = ConfidenceStruct(θmle[i], interval, [lb[i], ub[i]])
    end

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

    confidenceDict = Dict{Symbol, ConfidenceStruct}()
    p = Dict(:ind=>1, 
            :data=>data, 
            :newLb=>zeros(num_vars-1), 
            :newUb=>zeros(num_vars-1),
            :initGuess=>zeros(num_vars-1),
            :num_vars=>num_vars,
            :targetll=>(fmle+llstar),
            :likelihoodFunc=>likelihoodFunc)

    for (i, θname) in enumerate(θnames)
        interval = zeros(2)

        boundsmapping1d!(p[:newLb], lb, i)
        boundsmapping1d!(p[:newUb], ub, i)
        boundsmapping1d!(p[:initGuess], θmle, i)
        p[:ind]=i
        
        ϵ=(ub[i]-lb[i])/10^6
    
        g(x,p) = univariateΨ(x,p)[1]

        # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
        interval[1] = g(lb[i], p) <= 0.0 ? find_zero(g, (lb[i], θmle[i]), atol=ϵ, Roots.Brent(), p=p) : NaN
        interval[2] = g(ub[i], p) <= 0.0 ? find_zero(g, (θmle[i], ub[i]), atol=ϵ, Roots.Brent(), p=p) : NaN

        confidenceDict[θname] = ConfidenceStruct(θmle[i], interval, [lb[i], ub[i]])
    end

    return confidenceDict, p
end