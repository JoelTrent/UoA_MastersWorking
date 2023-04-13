function get_univariate_opt_func(profile_type::AbstractProfileType=LogLikelihood())

    if profile_type isa LogLikelihood || profile_type isa EllipseApprox
        return univariateΨ
    elseif profile_type isa EllipseApproxAnalytical
        return univariateΨ_ellipse_analytical
    end

    return (missing)
end

function univariate_confidenceinterval(univariate_optimiser::Function, 
                                        model::LikelihoodModel, 
                                        consistent::NamedTuple, 
                                        θi::Int, 
                                        atol::Float64)

    univ_opt_is_ellipse_analytical = univariate_optimiser == univariateΨ_ellipse_analytical

    interval = zeros(2)
    boundarySamples = zeros(model.core.num_pars, 2)

    newLb=zeros(model.core.num_pars-1) 
    newUb=zeros(model.core.num_pars-1)
    initGuess=zeros(model.core.num_pars-1)

    boundsmapping1d!(newLb, model.core.θlb, θi)
    boundsmapping1d!(newUb, model.core.θub, θi)
    boundsmapping1d!(initGuess, model.core.θmle, θi)

    θranges, λranges = variablemapping1dranges(model.core.num_pars, θi)

    if univ_opt_is_ellipse_analytical
        # p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        #     θranges=θranges, λranges=λranges, consistent=consistent)

        # interval[1] = univariate_optimiser(model.core.θlb[θi], p) <= 0.0 ? find_zero(univariate_optimiser, (model.core.θlb[θi], model.core.θmle[θi]), atol=ϵ, Roots.Brent(), p=p) : NaN
        # interval[2] = univariate_optimiser(model.core.θub[θi], p) <= 0.0 ? find_zero(univariate_optimiser, (model.core.θmle[θi], model.core.θub[θi]), atol=ϵ, Roots.Brent(), p=p) : NaN

        interval .= analytic_ellipse_loglike_1D_soln(θi, consistent.data, consistent.targetll)

        if interval[1] < model.core.θlb[θi]; interval[1]=NaN end
        if interval[2] > model.core.θub[θi]; interval[2]=NaN end

        return UnivariateConfidenceStructAnalytical(model.core.θmle[θi], interval, model.core.θlb[θi], model.core.θub[θi])
    end

    # else
    p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-1))

    # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
    if univariate_optimiser(model.core.θlb[θi], p) <= 0.0
        interval[1] =  find_zero(univariate_optimiser, (model.core.θlb[θi], model.core.θmle[θi]), atol=atol, Roots.Brent(), p=p) 
        boundarySamples[θi,1] = interval[1]
        variablemapping1d!(@view(boundarySamples[:, 1]), p.λ_opt, θranges, λranges)
    else
        interval[1] =  NaN
        boundarySamples[:, 1] .= NaN
    end

    if univariate_optimiser(model.core.θub[θi], p) <= 0.0
        interval[2] =  find_zero(univariate_optimiser, (model.core.θmle[θi], model.core.θub[θi]), atol=atol, Roots.Brent(), p=p)
        boundarySamples[θi,2] = interval[2]
        variablemapping1d!(@view(boundarySamples[:,2]), p.λ_opt, θranges, λranges)
    else
        interval[1] =  NaN
        boundarySamples[:, 1] .= NaN
    end

    return UnivariateConfidenceStruct(model.core.θmle[θi], interval, boundarySamples, model.core.θlb[θi], model.core.θub[θi])
end

# profile provided θ indices
"""
atol is the absolute tolerance that decides if f(x) ≈ 0.0. I.e. if the loglikelihood function is approximately at the boundary of interest.
"""
function univariate_confidenceintervals(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Int64}; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol=1e-8)

    if profile_type isa AbstractEllipseProfileType
        check_ellipse_approx_exists!(model)
    end

    confidenceDict = Dict{Symbol, Union{UnivariateConfidenceStruct, UnivariateConfidenceStructAnalytical}}()

    univariate_optimiser = get_univariate_opt_func(profile_type)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 1)

    # unique!(θs_to_profile)

    # at a later date, want to check if a this interval has already been evaluated for a given parameter
    # and/or if a wider/thinner confidence level has been evaluated yet (can use that knowledge to decrease the search bounds in 1D)
    for θi in θs_to_profile
        confidenceDict[model.core.θnames[θi]] = univariate_confidenceinterval(univariate_optimiser, model, consistent, θi, atol)
    end

    return confidenceDict
end

# profile just provided θnames
function univariate_confidenceintervals(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Symbol}; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol=1e-8)

    indices_to_profile = convertθnames_toindices(model, θs_to_profile)
    return univariate_confidenceintervals(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type, atol=atol)
end

# profile m random parameters (sampling without replacement), where 0 < m ≤ model.core.num_pars
function univariate_confidenceintervals(model::LikelihoodModel, 
                                        profile_m_random_parameters::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol=1e-8)

    profile_m_random_parameters = max(0, min(profile_m_random_parameters, model.core.num_pars))

    profile_m_random_parameters > 0 || throw(DomainError("profile_m_random_parameters must be a strictly positive integer"))

    indices_to_profile = sample(1:model.core.num_pars, profile_m_random_parameters, replace=false)

    return univariate_confidenceintervals(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type, atol=atol)
end

# profile all 
function univariate_confidenceintervals(model::LikelihoodModel; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol=1e-8)
    return univariate_confidenceintervals(model, collect(1:model.core.num_pars), confidence_level=confidence_level,
                            profile_type=profile_type, atol=atol)
end