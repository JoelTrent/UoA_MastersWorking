"""
mle_targetll means that the variable is created such that the ll at the mle is 0.0
"""
function univariate_confidenceinterval(univariate_optimiser::Function, 
                                        model::LikelihoodModel, 
                                        consistent::NamedTuple, 
                                        θi::Int, 
                                        profile_type::AbstractProfileType,
                                        atol::Float64,
                                        mle_targetll::Float64,
                                        ll_shift::Float64,
                                        num_points_in_interval::Int; 
                                        bracket_l::Vector{<:Float64}=Float64[],
                                        bracket_r::Vector{<:Float64}=Float64[])

    interval = zeros(2)
    ll = zeros(2)
    newLb, newUb, initGuess, θranges, λranges = init_univariate_parameters(model, θi)

    if univariate_optimiser == univariateΨ_ellipse_analytical

        p=(ind=θi, initGuess=initGuess, θranges=θranges, λranges=λranges, consistent=consistent)

        # p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        #     θranges=θranges, λranges=λranges, consistent=consistent)

        # interval[1] = univariate_optimiser(model.core.θlb[θi], p) <= 0.0 ? find_zero(univariate_optimiser, (model.core.θlb[θi], model.core.θmle[θi]), atol=ϵ, Roots.Brent(), p=p) : NaN
        # interval[2] = univariate_optimiser(model.core.θub[θi], p) <= 0.0 ? find_zero(univariate_optimiser, (model.core.θmle[θi], model.core.θub[θi]), atol=ϵ, Roots.Brent(), p=p) : NaN

        interval .= analytic_ellipse_loglike_1D_soln(θi, consistent.data, mle_targetll)

        if interval[1] < model.core.θlb[θi]; interval[1]=NaN end
        if interval[2] > model.core.θub[θi]; interval[2]=NaN end

        interval_points = zeros(2)
        
        if isnan(interval[1])
            interval_points[1] = model.core.θlb[θi] * 1.0
            ll[1] = univariate_optimiser(interval_points[1], p) + p.consistent.targetll
        else 
            interval_points[1] = interval[1] * 1.0
            ll[1] = mle_targetll
        end

        if isnan(interval[2])
            interval_points[2] = model.core.θub[θi] * 1.0   
            ll[2] = univariate_optimiser(interval_points[2], p) + p.consistent.targetll
        else
            interval_points[2] = interval[2] * 1.0
            ll[2] = mle_targetll
        end

        points = PointsAndLogLikelihood(interval_points, ll)

        if num_points_in_interval > 0
            points = get_points_in_interval_single_row(univariate_optimiser, model,
                                                        num_points_in_interval, θi,
                                                        profile_type, points)
        end

        return UnivariateConfidenceStructAnalytical(interval, points)
    end

    if isempty(bracket_l)
        bracket_l = [model.core.θlb[θi], model.core.θmle[θi]]
    end
    if isempty(bracket_r)
        bracket_r = [model.core.θmle[θi], model.core.θub[θi]]
    end

    # else
    interval_points = zeros(model.core.num_pars, 2)

    p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-1))

    # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
    if univariate_optimiser(bracket_l[1], p) <= 0.0
        interval[1] = find_zero(univariate_optimiser, bracket_l, atol=atol, Roots.Brent(), p=p) 
        interval_points[θi,1] = interval[1]
        variablemapping1d!(@view(interval_points[:, 1]), p.λ_opt, θranges, λranges)
        ll[1] = mle_targetll
    else
        interval[1] =  NaN
        interval_points[θi,1] = bracket_l[1] * 1.0
        ll[1] = univariate_optimiser(bracket_l[1], p) + p.consistent.targetll - ll_shift
        variablemapping1d!(@view(interval_points[:, 1]), p.λ_opt, θranges, λranges)
    end

    if univariate_optimiser(bracket_r[2], p) <= 0.0
        interval[2] =  find_zero(univariate_optimiser, bracket_r, atol=atol, Roots.Brent(), p=p)
        interval_points[θi,2] = interval[2]
        variablemapping1d!(@view(interval_points[:,2]), p.λ_opt, θranges, λranges)
        ll[2] = mle_targetll
    else
        interval[2] =  NaN        
        interval_points[θi,2] = bracket_r[2] * 1.0
        ll[2] = univariate_optimiser(bracket_r[2], p) + p.consistent.targetll - ll_shift
        variablemapping1d!(@view(interval_points[:, 2]), p.λ_opt, θranges, λranges)
    end

    points = PointsAndLogLikelihood(interval_points, ll)

    if num_points_in_interval > 0
        points = get_points_in_interval_single_row(univariate_optimiser, model,
                                                    num_points_in_interval, θi,
                                                    profile_type, points)
    end

    return UnivariateConfidenceStruct(interval, points)
end