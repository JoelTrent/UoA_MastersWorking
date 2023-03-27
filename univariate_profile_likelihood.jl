function variablemapping1dranges(num_pars::T, index::T) where T <: Int
    θranges = (1:(index-1), (index+1):num_pars)
    λranges = (1:(index-1), index:(num_pars-1))
    return θranges, λranges
end

function variablemapping1d!(θ, λ, θranges, λranges)
    θ[θranges[1]] .= @view(λ[λranges[1]])
    θ[θranges[2]] .= @view(λ[λranges[2]])
    return θ
end

function boundsmapping1d!(newbounds::Vector{<:Float64}, bounds::Vector{<:Float64}, index::Int)
    newbounds[1:(index-1)] .= @view(bounds[1:(index-1)])
    newbounds[index:end]   .= @view(bounds[(index+1):end])
    return nothing
end

function univariateΨ_ellipse_analytical(Ψ, p)
    return analytic_ellipse_loglike([Ψ], [p.ind], p.consistent.data) - p.consistent.targetll
end

# function univariateΨ_unsafe(Ψ, p)
#     θs=zeros(p.consistent.num_pars)
#     θs[p.ind] = Ψ
    
#     function fun(λ); p.consistent.loglikefunction(variablemapping1d!(θs, λ, p.θranges, p.λranges), p.consistent.data) end

#     (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
#     llb=fopt-p.consistent.targetll
#     return llb, xopt
# end

function univariateΨ(Ψ, p)
    θs=zeros(p.consistent.num_pars)

    function fun(λ)
        θs[p.ind] = Ψ
        return p.consistent.loglikefunction(variablemapping1d!(θs, λ, p.θranges, p.λranges), p.consistent.data) 
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    return llb, xopt
end

function get_univariate_opt_func(profile_type::Symbol)

    if profile_type == :LogLikelihood || profile_type == :EllipseApprox
        return univariateΨ
    elseif profile_type == :EllipseApproxAnalytical
        return univariateΨ_ellipse_analytical
    end

    return (missing)
end

function univariate_confidenceinterval(univariate_optimiser::Function, model::LikelihoodModel, consistent::NamedTuple, θi::Int)

    interval = zeros(2)

    newLb=zeros(model.core.num_pars-1) 
    newUb=zeros(model.core.num_pars-1)
    initGuess=zeros(model.core.num_pars-1)

    boundsmapping1d!(newLb, model.core.θlb, θi)
    boundsmapping1d!(newUb, model.core.θub, θi)
    boundsmapping1d!(initGuess, model.core.θmle, θi)

    θranges, λranges = variablemapping1dranges(model.core.num_pars, θi)

    p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, λranges=λranges, consistent=consistent)

    ϵ=(model.core.θub[θi]-model.core.θlb[θi])/10^6

    g(x,p) = univariate_optimiser(x,p)[1]

    # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
    interval[1] = g(model.core.θlb[θi], p) <= 0.0 ? find_zero(g, (model.core.θlb[θi], model.core.θmle[θi]), atol=ϵ, Roots.Brent(), p=p) : NaN
    interval[2] = g(model.core.θub[θi], p) <= 0.0 ? find_zero(g, (model.core.θmle[θi], model.core.θub[θi]), atol=ϵ, Roots.Brent(), p=p) : NaN

    return UnivariateConfidenceStruct(model.core.θmle[θi], interval, model.core.θlb[θi], model.core.θub[θi])
end

# profile provided θ indices
function univariate_confidenceintervals(model::LikelihoodModel, θs_to_profile::Vector{<:Int64}; 
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood)
    
    valid_profile_type = profile_type in [:EllipseApprox, :EllipseApproxAnalytical, :LogLikelihood]
    @assert valid_profile_type "Specified `profile_type` is invalid. Allowed values are :EllipseApprox, :EllipseApproxAnalytical, :LogLikelihood."

    if profile_type in [:EllipseApprox, :EllipseApproxAnalytical]
        check_ellipse_approx_exists!(model)
    end

    confidenceDict = Dict{Symbol, UnivariateConfidenceStruct}()

    univariate_optimiser = get_univariate_opt_func(profile_type)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 1)

    # unique!(θs_to_profile)

    # at a later date, want to check if a this interval has already been evaluated for a given parameter
    # and/or if a wider/thinner confidence level has been evaluated yet (can use that knowledge to decrease the search bounds in 1D)
    for θi in θs_to_profile
        confidenceDict[model.core.θnames[θi]] = univariate_confidenceinterval(univariate_optimiser, model, consistent, θi)
    end

    return confidenceDict
end

# profile just provided θnames
function univariate_confidenceintervals(model::LikelihoodModel, θs_to_profile::Vector{<:Symbol}; 
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood)

    indices_to_profile = convertθnames_toindices(model, θs_to_profile)
    return univariate_confidenceintervals(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type)
end

# profile m random parameters (sampling without replacement), where 0 < m ≤ model.core.num_pars
function univariate_confidenceintervals(model::LikelihoodModel, profile_m_random_parameters::Int; 
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood)

    profile_m_random_parameters = max(0, min(profile_m_random_parameters, model.core.num_pars))

    if profile_m_random_parameters == 0
        @error "`profile_m_random_parameters` must be a strictly positive integer."
        return nothing
    end

    indices_to_profile = sample(1:model.core.num_pars, profile_m_random_parameters, replace=false)

    return univariate_confidenceintervals(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type)
end

# profile all 
function univariate_confidenceintervals(model::LikelihoodModel; 
        confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood)
    return univariate_confidenceintervals(model, collect(1:model.core.num_pars), confidence_level=confidence_level,
                            profile_type=profile_type)
end