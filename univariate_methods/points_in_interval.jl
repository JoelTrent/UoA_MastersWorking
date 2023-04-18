function update_uni_dict_internal!(model::LikelihoodModel,
                                    uni_row_number::Int,
                                    internal_points::Array{Float64},
                                    ll::Vector{<:Float64})
    
    points = PointsAndLogLikelihood(internal_points, ll)

    interval_struct = model.uni_profiles_dict[uni_row_number]

    model.uni_profiles_dict[uni_row_number] = @set interval_struct.internal_points = points

    return nothing
end

function update_uni_df_internal_points!(model::LikelihoodModel,
                                        uni_row_number::Int,
                                        num_new_points::Int)

    model.uni_profiles_df[uni_row_number, [:evaluated_internal_points, :num_points]] .= true, num_new_points+2

    return nothing
end

"""
Will get points in the interval and at the interval boundaries. E.g. if a boundary is NaN, that means the true boundary is on the other side of a user provided bound - we want to generate the point on the bound to make plots and forward propogation useful.
"""
function get_points_in_interval_single_row!(model::LikelihoodModel,
                                uni_row_number::Int,
                                num_new_points::Int;
                                update_df::Bool=true)

    num_new_points > 0 || throw(DomainError("num_new_points must be a strictly positive integer"))

    θi = model.uni_profiles_df.θindex[uni_row_number]
    profile_type = model.uni_profiles_df.profile_type[uni_row_number]
    univariate_optimiser = get_univariate_opt_func(profile_type)

    interval = get_uni_confidenceinterval(model, uni_row_number) * 1.0
    interval[1] = isnan(interval[1]) ? model.core.θlb[θi] : interval[1]
    interval[2] = isnan(interval[2]) ? model.core.θub[θi] : interval[2]
    
    point_locations = LinRange(interval[1], interval[2], num_new_points+2)
    ll = zeros(num_new_points+2)

    if univariate_optimiser == univariateΨ_ellipse_analytical

        internal_points = collect(point_locations)
        
        for (i, θ) in enumerate(internal_points)
            ll[i] = analytic_ellipse_loglike([θ], [θi], (θmle=model.core.θmle, Γmle=model.ellipse_MLE_approx.Γmle))
        end
        
    else
        internal_points = zeros(model.core.num_pars, num_new_points+2)

        newLb, newUb, initGuess, θranges, λranges = 
                                init_univariate_parameters(model, θi)

        consistent = get_consistent_tuple(model, 0.0, profile_type, 1)
        p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
            θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-1))

        for (i, θ) in enumerate(point_locations)
            ll[i] = univariate_optimiser(θ, p)
            variablemapping1d!(@view(internal_points[:,i]), p.λ_opt, θranges, λranges)
        end
        internal_points[θi,:] .= point_locations
    end

    update_uni_dict_internal!(model, uni_row_number, internal_points, ll)

    if update_df
        update_uni_df_internal_points!(model, uni_row_number, num_new_points)
    end
    return nothing
end

function get_points_in_interval!(model::LikelihoodModel,
                                    num_new_points::Int;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[]
                                    )
    df = model.uni_profiles_df
    row_subset = trues(nrow(df))
    row_subset .= df.num_points .!= (num_new_points+2)

    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end

    sub_df = @view(df[row_subset, :])

    for i in 1:nrow(sub_df)
        get_points_in_interval_single_row!(model, sub_df[i, :row_ind], 
                                            num_new_points, update_df=false)
    end

    sub_df[:, :evaluated_internal_points] .= true
    sub_df[:, :num_points] .= num_new_points+2

    return nothing
end