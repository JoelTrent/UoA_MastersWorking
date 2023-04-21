function get_uni_confidenceinterval(model::LikelihoodModel,
                                    uni_row_number::Int)
    return model.uni_profiles_dict[uni_row_number].confidence_interval
end

function get_interval_brackets(model::LikelihoodModel, 
                                θi::Int,
                                confidence_level::Float64,
                                profile_type::AbstractProfileType)

    prof_keys = collect(keys(model.uni_profile_row_exists[(θi, profile_type)]))
    len_keys = length(prof_keys)
    if len_keys > 1
        sort!(prof_keys)
        conf_ind = findfirst(isequal(confidence_level), prof_keys)

        bracket_l, bracket_r = [0.0, 0.0], [0.0, 0.0]

        if conf_ind>1
            bracket_l[2], bracket_r[1] = get_uni_confidenceinterval(model, model.uni_profile_row_exists[(θi, profile_type)][prof_keys[conf_ind-1]])
        else
            bracket_l[2], bracket_r[1] = model.core.θmle[θi], model.core.θmle[θi]
        end
        
        if conf_ind<len_keys
            bracket_l[1], bracket_r[2] = get_uni_confidenceinterval(model, model.uni_profile_row_exists[(θi, profile_type)][prof_keys[conf_ind+1]])
        else
            bracket_l[1], bracket_r[2] = model.core.θlb[θi], model.core.θub[θi]
        end
    else
        bracket_l, bracket_r = Float64[], Float64[]
    end
    return bracket_l, bracket_r
end

function add_uni_profiles_rows!(model::LikelihoodModel, 
                                num_rows_to_add::Int)
    new_rows = init_uni_profiles_df(num_rows_to_add, 
                                    existing_largest_row=nrow(model.uni_profiles_df))

    model.uni_profiles_df = vcat(model.uni_profiles_df, new_rows)
    return nothing
end

function set_uni_profiles_row!(model::LikelihoodModel, 
                                    θi::Int,
                                    not_evaluated_internal_points::Bool,
                                    not_evaluated_predictions::Bool,
                                    confidence_level::Float64,
                                    profile_type::AbstractProfileType,
                                    num_points::Int,
                                    additional_width::Float64)
    model.uni_profiles_df[model.num_uni_profiles, 2:end] .= θi*1, 
                                                            not_evaluated_internal_points,
                                                            not_evaluated_predictions,
                                                            confidence_level,
                                                            profile_type,
                                                            num_points,
                                                            additional_width
    return nothing
end

function get_univariate_opt_func(profile_type::AbstractProfileType=LogLikelihood())

    if profile_type isa LogLikelihood || profile_type isa EllipseApprox
        return univariateΨ
    elseif profile_type isa EllipseApproxAnalytical
        return univariateΨ_ellipse_unbounded #univariateΨ_ellipse_analytical
    end

    return (missing)
end

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
                                        num_points_in_interval::Int,
                                        additional_width::Float64; 
                                        bracket_l::Vector{<:Float64}=Float64[],
                                        bracket_r::Vector{<:Float64}=Float64[])

    interval = zeros(2)
    ll = zeros(2)
    interval_points = zeros(model.core.num_pars, 2)
    newLb, newUb, initGuess, θranges, λranges = init_univariate_parameters(model, θi)

    p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, λranges=λranges, consistent=consistent, 
        λ_opt=zeros(model.core.num_pars-1))

    if isempty(bracket_l)
        bracket_l = [model.core.θlb[θi], model.core.θmle[θi]]
    end
    if isempty(bracket_r)
        bracket_r = [model.core.θmle[θi], model.core.θub[θi]]
    end

    if univariate_optimiser == univariateΨ_ellipse_unbounded

        interval .= analytic_ellipse_loglike_1D_soln(θi, consistent.data_analytic, mle_targetll)
        
        if interval[1] >= bracket_l[1]
            interval_points[θi,1] = interval[1]
            univariate_optimiser(interval[1], p)
            variablemapping1d!(@view(interval_points[:, 1]), p.λ_opt, θranges, λranges)
            ll[1] = mle_targetll
        else
            interval[1]=NaN 
        end

        if interval[2] <= bracket_r[2]
            interval_points[θi,2] = interval[2]
            univariate_optimiser(interval[2], p)
            variablemapping1d!(@view(interval_points[:, 2]), p.λ_opt, θranges, λranges)
            ll[2] = mle_targetll
        else 
            interval[2]=NaN
        end

    else
        # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
        if univariate_optimiser(bracket_l[1], p) <= 0.0
            interval[1] = find_zero(univariate_optimiser, bracket_l, atol=atol, Roots.Brent(), p=p) 
            interval_points[θi,1] = interval[1]
            variablemapping1d!(@view(interval_points[:, 1]), p.λ_opt, θranges, λranges)
            ll[1] = mle_targetll
        else
            interval[1] = NaN
        end

        if univariate_optimiser(bracket_r[2], p) <= 0.0
            interval[2] = find_zero(univariate_optimiser, bracket_r, atol=atol, Roots.Brent(), p=p)
            interval_points[θi,2] = interval[2]
            variablemapping1d!(@view(interval_points[:,2]), p.λ_opt, θranges, λranges)
            ll[2] = mle_targetll
        else
            interval[2] = NaN
        end
    end

    if isnan(interval[1])
        interval_points[θi,1] = bracket_l[1] * 1.0
        ll[1] = univariate_optimiser(bracket_l[1], p) + p.consistent.targetll - ll_shift
        variablemapping1d!(@view(interval_points[:, 1]), p.λ_opt, θranges, λranges)
    end
    if isnan(interval[2])         
        interval_points[θi,2] = bracket_r[2] * 1.0
        ll[2] = univariate_optimiser(bracket_r[2], p) + p.consistent.targetll - ll_shift
        variablemapping1d!(@view(interval_points[:, 2]), p.λ_opt, θranges, λranges)
    end
  
    points = PointsAndLogLikelihood(interval_points, ll, [1,2])

    if num_points_in_interval > 0
        points = get_points_in_interval_single_row(univariate_optimiser, model,
                                                    num_points_in_interval, θi,
                                                    profile_type, points, additional_width)
    end

    return UnivariateConfidenceStruct(interval, points)
end

function univariate_confidenceinterval_master(univariate_optimiser::Function,
                                        model::LikelihoodModel,
                                        consistent::NamedTuple, 
                                        θi::Int,
                                        confidence_level::Float64, 
                                        profile_type::AbstractProfileType, 
                                        atol::Real,
                                        mle_targetll::Float64,
                                        ll_shift::Float64,
                                        use_existing_profiles::Bool,
                                        num_points_in_interval::Int,
                                        additional_width::Float64)
    if use_existing_profiles
        bracket_l, bracket_r = get_interval_brackets(model, θi, confidence_level,
                                                        profile_type)                

        interval_struct = univariate_confidenceinterval(univariate_optimiser, model, consistent,
                                                        θi, profile_type, atol, mle_targetll, ll_shift, 
                                                        num_points_in_interval, additional_width,
                                                        bracket_l=bracket_l, bracket_r=bracket_r)
    else
        interval_struct = univariate_confidenceinterval(univariate_optimiser, model, consistent,
                                                        θi, profile_type, atol, mle_targetll, ll_shift, 
                                                        num_points_in_interval, additional_width)
    end
    return interval_struct
end

# profile provided θ indices
"""
atol is the absolute tolerance that decides if f(x) ≈ 0.0. I.e. if the loglikelihood function is approximately at the boundary of interest.
"""
function univariate_confidenceintervals!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Int64}; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol::Float64=1e-8,
                                        use_existing_profiles::Bool=false,
                                        θs_is_unique::Bool=false,
                                        use_distributed::Bool=false, 
                                        num_points_in_interval::Int=0,
                                        additional_width::Float64=0.0)
                                        # existing_profiles::Symbol=:overwrite)

    atol > 0 || throw(DomainError("atol must be a strictly positive integer"))
    num_points_in_interval >= 0 || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
    additional_width >= 0 || throw(DomainError("additional_width must be greater than or equal to zero"))

    additional_width = num_points_in_interval > 0 ? additional_width : 0.0

    if profile_type isa AbstractEllipseProfileType
        check_ellipse_approx_exists!(model)
    end

    # confidenceDict = Dict{Symbol, Union{UnivariateConfidenceStruct, UnivariateConfidenceStructAnalytical}}()

    univariate_optimiser = get_univariate_opt_func(profile_type)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 1)
    mle_targetll = get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), 1)
    ll_shift = ll_correction(model, profile_type, 0.0)

    θs_is_unique || (sort(θs_to_profile); unique!(θs_to_profile))

    init_uni_profile_row_exists!(model, θs_to_profile, profile_type)

    # check if profile has already been evaluated
    θs_to_keep = trues(length(θs_to_profile))
    for (i, θi) in enumerate(θs_to_profile)
        if model.uni_profile_row_exists[(θi, profile_type)][confidence_level] != 0
            θs_to_keep[i] = false
        end
    end
    θs_to_profile = θs_to_profile[θs_to_keep]
    len_θs_to_profile = length(θs_to_profile)
    len_θs_to_profile > 0 || return nothing

    num_rows_required = (len_θs_to_profile + model.num_uni_profiles) - nrow(model.uni_profiles_df)

    if num_rows_required > 0
        add_uni_profiles_rows!(model, num_rows_required)
    end

    not_evaluated_internal_points = num_points_in_interval > 0 ? false : true

    if !use_distributed
        for θi in θs_to_profile
            model.num_uni_profiles += 1
            model.uni_profile_row_exists[(θi, profile_type)][confidence_level] = model.num_uni_profiles * 1

            interval_struct = univariate_confidenceinterval_master(univariate_optimiser, model,
                                                                consistent, θi, 
                                                                confidence_level, profile_type,
                                                                atol, mle_targetll, ll_shift,
                                                                use_existing_profiles,
                                                                num_points_in_interval,
                                                                additional_width)

            model.uni_profiles_dict[model.num_uni_profiles] = interval_struct
            
            set_uni_profiles_row!(model, θi, not_evaluated_internal_points, true, confidence_level, 
                                    profile_type, num_points_in_interval+2, additional_width)
        end

    else
        profiles_to_add = @distributed (vcat) for θi in θs_to_profile
            (θi, univariate_confidenceinterval_master(univariate_optimiser, model,
                                                        consistent, θi, 
                                                        confidence_level, profile_type,
                                                        atol, mle_targetll, ll_shift,
                                                        use_existing_profiles,
                                                        num_points_in_interval,
                                                        additional_width))
        end

        for (θi, interval_struct) in profiles_to_add
            model.num_uni_profiles += 1
            model.uni_profile_row_exists[(θi, profile_type)][confidence_level] = model.num_uni_profiles * 1

            model.uni_profiles_dict[model.num_uni_profiles] = interval_struct

            set_uni_profiles_row!(model, θi, not_evaluated_internal_points, true, confidence_level, 
                                    profile_type, num_points_in_interval+2, additional_width)
        end        
    end
    
    return nothing
end

# profile just provided θnames
function univariate_confidenceintervals!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Symbol}; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol::Real=1e-8,
                                        use_existing_profiles::Bool=false,
                                        θs_is_unique::Bool=false,
                                        use_distributed::Bool=false, 
                                        num_points_in_interval::Int=0)

    indices_to_profile = convertθnames_toindices(model, θs_to_profile)
    univariate_confidenceintervals!(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type, atol=atol,
                                use_existing_profiles=use_existing_profiles,
                                θs_is_unique=θs_is_unique, use_distributed=use_distributed,
                                num_points_in_interval=num_points_in_interval)
    return nothing
end

# profile m random parameters (sampling without replacement), where 0 < m ≤ model.core.num_pars
function univariate_confidenceintervals!(model::LikelihoodModel, 
                                        profile_m_random_parameters::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol::Real=1e-8,
                                        use_existing_profiles::Bool=false,
                                        use_distributed::Bool=false, 
                                        num_points_in_interval::Int=0)

    profile_m_random_parameters = max(0, min(profile_m_random_parameters, model.core.num_pars))
    profile_m_random_parameters > 0 || throw(DomainError("profile_m_random_parameters must be a strictly positive integer"))

    indices_to_profile = sample(1:model.core.num_pars, profile_m_random_parameters, replace=false)

    univariate_confidenceintervals!(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type, atol=atol,
                                use_existing_profiles=use_existing_profiles,
                                θs_is_unique=true, use_distributed=use_distributed,
                                num_points_in_interval=num_points_in_interval)
    return nothing
end

# profile all 
function univariate_confidenceintervals!(model::LikelihoodModel; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol::Real=1e-8,
                                        use_existing_profiles::Bool=false,
                                        use_distributed::Bool=false, 
                                        num_points_in_interval::Int=0)
    univariate_confidenceintervals!(model, collect(1:model.core.num_pars), confidence_level=confidence_level,
                            profile_type=profile_type, atol=atol,
                            use_existing_profiles=use_existing_profiles,
                            θs_is_unique=true, use_distributed=use_distributed,
                            num_points_in_interval=num_points_in_interval)
    return nothing
end