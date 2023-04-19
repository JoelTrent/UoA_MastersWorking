function update_uni_dict_internal!(model::LikelihoodModel,
                                    uni_row_number::Int,
                                    points::PointsAndLogLikelihood)

    interval_struct = model.uni_profiles_dict[uni_row_number]
    model.uni_profiles_dict[uni_row_number] = @set interval_struct.interval_points = points

    return nothing
end

# function update_uni_df_internal_points!(model::LikelihoodModel,
#                                         uni_row_number::Int,
#                                         num_new_points::Int)
# 
#     model.uni_profiles_df[uni_row_number, [:not_evaluated_internal_points, :num_points]] .= false, num_new_points+2
# 
#     return nothing
# end

"""
Will get points in the interval and at the interval boundaries - current_interval_points has at least the boundary points. 
"""
function get_points_in_interval_single_row(univariate_optimiser::Function, 
                                model::LikelihoodModel,
                                num_new_points::Int,
                                θi::Int,
                                profile_type::AbstractProfileType,
                                current_interval_points::PointsAndLogLikelihood)

    num_new_points > 0 || throw(DomainError("num_new_points must be a strictly positive integer"))
    
    ll = zeros(num_new_points+2)
    ll[[1,end]] .= current_interval_points.ll[1], current_interval_points.ll[end]

    if univariate_optimiser == univariateΨ_ellipse_analytical

        point_locations = LinRange(current_interval_points.points[1], current_interval_points.points[end], num_new_points+2)
        interval_points = collect(point_locations)
        
        for i in 2:(num_new_points+1)
            ll[i] = analytic_ellipse_loglike([interval_points[i]], [θi], 
                        (θmle=model.core.θmle, Γmle=model.ellipse_MLE_approx.Γmle))
        end
       
    else
        point_locations = LinRange(current_interval_points.points[θi,1], current_interval_points.points[θi,end], num_new_points+2)

        interval_points = zeros(model.core.num_pars, num_new_points+2)

        newLb, newUb, initGuess, θranges, λranges = 
                                init_univariate_parameters(model, θi)

        consistent = get_consistent_tuple(model, 0.0, profile_type, 1)
        p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
            θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-1))

        for i in 2:(num_new_points+1)
            ll[i] = univariate_optimiser(point_locations[i], p)
            variablemapping1d!(@view(interval_points[:,i]), p.λ_opt, θranges, λranges)
        end
        interval_points[θi,2:(end-1)]  .= point_locations[2:(end-1)]
        interval_points[:,1]   .= current_interval_points.points[:,1]
        interval_points[:,end] .= current_interval_points.points[:,end]
    end

    return PointsAndLogLikelihood(interval_points, ll)
end

function get_points_in_interval_single_row(model::LikelihoodModel,
                                uni_row_number::Int,
                                num_new_points::Int)

    θi = model.uni_profiles_df.θindex[uni_row_number]
    profile_type = model.uni_profiles_df.profile_type[uni_row_number]
    univariate_optimiser = get_univariate_opt_func(profile_type)
    current_interval_points = model.uni_profiles_dict[uni_row_number].interval_points

    return get_points_in_interval_single_row(univariate_optimiser, model, num_new_points, 
                                                θi, profile_type, current_interval_points)
end

function get_points_in_interval!(model::LikelihoodModel,
                                    num_new_points::Int;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[]
                                    )

    0 < num_new_points || throw(DomainError("num_new_points must be a strictly positive integer"))
    df = model.uni_profiles_df
    row_subset = trues(nrow(df))

    row_subset .= (df.num_points .!= (num_new_points+2))
    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end

    sub_df = @view(df[row_subset, :])

    for i in 1:nrow(sub_df)
        points = get_points_in_interval_single_row(model, sub_df[i, :row_ind], 
                                            num_new_points)

        update_uni_dict_internal!(model, sub_df[i, :row_ind], points)
    end

    sub_df[:, :not_evaluated_internal_points] .= false
    sub_df[:, :not_evaluated_predictions] .= true
    sub_df[:, :num_points] .= num_new_points+2

    return nothing
end