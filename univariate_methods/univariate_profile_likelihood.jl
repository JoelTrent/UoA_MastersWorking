function init_uni_profile_row_exists!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Int},
                                        profile_type::AbstractProfileType)

    for θi in θs_to_profile
        if !haskey(model.uni_profile_row_exists, (θi, profile_type))
            model.uni_profile_row_exists[(θi, profile_type)] = DefaultDict{Float64, Int}(0)
        end
    end

    return nothing
end

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
        end
        if conf_ind<len_keys
            bracket_l[1], bracket_r[2] = get_uni_confidenceinterval(model, model.uni_profile_row_exists[(θi, profile_type)][prof_keys[conf_ind+1]])
        end
    else
        bracket_l, bracket_r = Float64[], Float64[]
    end
    return bracket_l, bracket_r
end

"""
TODO: user control of how many new rows get added if/when we need to add them.
"""
function add_uni_profile_rows!(model::LikelihoodModel)
    new_rows = init_uni_profiles_df(model.core.num_pars, existing_largest_row=nrow(model.uni_profiles_df))

    model.uni_profiles_df = vcat(model.uni_profiles_df, new_rows)
    return nothing
end

function update_uni_profile_row!(model::LikelihoodModel, 
                                    θi::Int,
                                    evaluated_internal_points::Bool,
                                    confidence_level::Float64,
                                    profile_type::AbstractProfileType,
                                    num_points::Int)
    model.uni_profiles_df[model.num_uni_profiles, 2:end] .= θi*1, 
                                                            evaluated_internal_points,
                                                            confidence_level,
                                                            profile_type,
                                                            num_points
    return nothing
end

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
                                        atol::Float64; 
                                        bracket_l::Vector{<:Float64}=Float64[],
                                        bracket_r::Vector{<:Float64}=Float64[])

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

        return UnivariateConfidenceStructAnalytical(interval)
    end

    if isempty(bracket_l)
        bracket_l = [model.core.θlb[θi], model.core.θmle[θi]]
    end
    if isempty(bracket_r)
        bracket_r = [model.core.θmle[θi], model.core.θub[θi]]
    end

    # else
    p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-1))

    # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
    if univariate_optimiser(bracket_l[1], p) <= 0.0
        interval[1] =  find_zero(univariate_optimiser, bracket_l, atol=atol, Roots.Brent(), p=p) 
        boundarySamples[θi,1] = interval[1]
        variablemapping1d!(@view(boundarySamples[:, 1]), p.λ_opt, θranges, λranges)
    else
        interval[1] =  NaN
        boundarySamples[:, 1] .= NaN
    end

    if univariate_optimiser(bracket_r[2], p) <= 0.0
        interval[2] =  find_zero(univariate_optimiser, bracket_r, atol=atol, Roots.Brent(), p=p)
        boundarySamples[θi,2] = interval[2]
        variablemapping1d!(@view(boundarySamples[:,2]), p.λ_opt, θranges, λranges)
    else
        interval[2] =  NaN
        boundarySamples[:, 2] .= NaN
    end

    return UnivariateConfidenceStruct(interval, boundarySamples)
end

function univariate_confidenceinterval_master(univariate_optimiser::Function,
                                        model::LikelihoodModel,
                                        consistent::NamedTuple, 
                                        θi::Int,
                                        confidence_level::Float64, 
                                        profile_type::AbstractProfileType, 
                                        atol::Float64,
                                        use_existing_profiles::Bool)
    if use_existing_profiles
        bracket_l, bracket_r = get_interval_brackets(model, θi, confidence_level,
                                                        profile_type)                

        interval_struct = univariate_confidenceinterval(univariate_optimiser, model, consistent,
                                                        θi, atol, bracket_l=bracket_l,
                                                        bracket_r=bracket_r)
    else
        interval_struct = univariate_confidenceinterval(univariate_optimiser, model, consistent,
                                                        θi, atol)
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
                                        use_distributed::Bool=false)
                                        # existing_profiles::Symbol=:overwrite)

    if profile_type isa AbstractEllipseProfileType
        check_ellipse_approx_exists!(model)
    end

    # confidenceDict = Dict{Symbol, Union{UnivariateConfidenceStruct, UnivariateConfidenceStructAnalytical}}()

    univariate_optimiser = get_univariate_opt_func(profile_type)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 1)

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
    length(θs_to_profile) > 0 || return nothing

    if !use_distributed
        for θi in θs_to_profile
            model.num_uni_profiles += 1
            model.uni_profile_row_exists[(θi, profile_type)][confidence_level] = model.num_uni_profiles * 1

            interval_struct = univariate_confidenceinterval_master(univariate_optimiser, model,
                                                                consistent, θi, 
                                                                confidence_level, profile_type,
                                                                atol, use_existing_profiles)

            model.uni_profiles_dict[model.num_uni_profiles] = interval_struct
            
            # fill out a new row in the dataframe (make sure that we haven't run out of rows yet)
            if nrow(model.uni_profiles_df) < model.num_uni_profiles
                add_uni_profile_rows!(model)
            end
            update_uni_profile_row!(model, θi, false, confidence_level, profile_type, 2)
        end

    else
        profiles_to_add = @distributed (vcat) for θi in θs_to_profile
            (θi, univariate_confidenceinterval_master(univariate_optimiser, model,
                                                        consistent, θi, 
                                                        confidence_level, profile_type,
                                                        atol, use_existing_profiles))
        end

        for (θi, interval_struct) in profiles_to_add
            model.num_uni_profiles += 1
            model.uni_profile_row_exists[(θi, profile_type)][confidence_level] = model.num_uni_profiles * 1

            model.uni_profiles_dict[model.num_uni_profiles] = interval_struct

            # fill out a new row in the dataframe (make sure that we haven't run out of rows yet)
            if nrow(model.uni_profiles_df) < model.num_uni_profiles
                add_uni_profile_rows!(model)
            end
            update_uni_profile_row!(model, θi, false, confidence_level, profile_type, 2)
        end        
    end
    
    return nothing
end

# profile just provided θnames
function univariate_confidenceintervals!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Symbol}; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol=1e-8,
                                        use_existing_profiles::Bool=false,
                                        θs_is_unique::Bool=false)

    indices_to_profile = convertθnames_toindices(model, θs_to_profile)
    return univariate_confidenceintervals!(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type, atol=atol,
                                use_existing_profiles=use_existing_profiles,
                                θs_is_unique=θs_is_unique)
end

# profile m random parameters (sampling without replacement), where 0 < m ≤ model.core.num_pars
function univariate_confidenceintervals!(model::LikelihoodModel, 
                                        profile_m_random_parameters::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol=1e-8,
                                        use_existing_profiles::Bool=false)

    profile_m_random_parameters = max(0, min(profile_m_random_parameters, model.core.num_pars))

    profile_m_random_parameters > 0 || throw(DomainError("profile_m_random_parameters must be a strictly positive integer"))

    indices_to_profile = sample(1:model.core.num_pars, profile_m_random_parameters, replace=false)

    return univariate_confidenceintervals!(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type, atol=atol,
                                use_existing_profiles=use_existing_profiles,
                                θs_is_unique=true)
end

# profile all 
function univariate_confidenceintervals!(model::LikelihoodModel; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        atol=1e-8,
                                        use_existing_profiles::Bool=false)
    return univariate_confidenceintervals!(model, collect(1:model.core.num_pars), confidence_level=confidence_level,
                            profile_type=profile_type, atol=atol,
                            use_existing_profiles=use_existing_profiles,
                            θs_is_unique=true)
end