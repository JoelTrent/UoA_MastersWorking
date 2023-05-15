function update_uni_dict_internal!(model::LikelihoodModel,
                                    uni_row_number::Int,
                                    points::PointsAndLogLikelihood)

    interval_struct = model.uni_profiles_dict[uni_row_number]
    model.uni_profiles_dict[uni_row_number] = @set interval_struct.interval_points = points

    return nothing
end

# function update_uni_df_internal_points!(model::LikelihoodModel,
#                                         uni_row_number::Int,
#                                         num_points_in_interval::Int)
# 
#     model.uni_profiles_df[uni_row_number, [:not_evaluated_internal_points, :num_points]] .= false, num_points_in_interval+2
# 
#     return nothing
# end

"""
Will get points in the interval and at the interval boundaries - current_interval_points has at least the boundary points. 

additional_width is the additional width past the interval boundary to also evaluate points on.
"""
function get_points_in_interval_single_row(univariate_optimiser::Function, 
                                model::LikelihoodModel,
                                num_points_in_interval::Int,
                                θi::Int,
                                profile_type::AbstractProfileType,
                                current_interval_points::PointsAndLogLikelihood,
                                additional_width::Real=0.0)

    num_points_in_interval > 0 || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
    additional_width >= 0 || throw(DomainError("additional_width must be greater than or equal to zero"))
    
    boundary_indices = current_interval_points.boundary_col_indices
    
    newLb, newUb, initGuess, θranges, λranges = init_univariate_parameters(model, θi)
    
    consistent = get_consistent_tuple(model, 0.0, profile_type, 1)
    p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, λranges=λranges, consistent=consistent, 
        λ_opt=zeros(model.core.num_pars-1))
    
    if additional_width > 0
        boundary = current_interval_points.points[θi, boundary_indices]
        boundary_width = diff(boundary)[1]
        half_add_width = boundary_width * (additional_width / 2.0)
        interval_to_eval = [max(boundary[1]-half_add_width, model.core.θlb[θi]), 
                                min(boundary[2]+half_add_width, model.core.θub[θi])]
        
        interval_width = diff(interval_to_eval)[1]

        additional_widths = [boundary[1]-interval_to_eval[1], interval_to_eval[2]-boundary[2]]
        points_in_each_interval = [0, num_points_in_interval+2, 0]

        point_locations = Float64[]

        
        if additional_widths[1] > 0.0
            num_points = convert(Int, max(1.0, round((additional_widths[1]/interval_width)*num_points_in_interval, RoundDown)))
            points_in_each_interval[1] = num_points

            append!(point_locations, LinRange(interval_to_eval[1], 
                                                boundary[1], 
                                                points_in_each_interval[1]+1)[1:(end-1)]
                    )
        end

        append!(point_locations, LinRange(boundary[1], boundary[2], num_points_in_interval+2)
                    )

        if additional_widths[2] > 0.0
            num_points = convert(Int, max(1.0, round((additional_widths[2]/interval_width)*num_points_in_interval, RoundDown)))
            points_in_each_interval[3] = num_points

            append!(point_locations, LinRange(boundary[2],
                                                interval_to_eval[2],
                                                points_in_each_interval[3]+1)[2:end]
                    )
        end
        new_boundary_indices = [points_in_each_interval[1]+1, points_in_each_interval[1]+points_in_each_interval[2]]

    else
        new_boundary_indices = [1, num_points_in_interval+2]
        point_locations = LinRange(current_interval_points.points[θi, boundary_indices[1]], 
                                    current_interval_points.points[θi, boundary_indices[2]], 
                                    num_points_in_interval+2)
    end

    total_points = length(point_locations)

    ll = zeros(total_points)
    interval_points = zeros(model.core.num_pars, total_points)

    ll[new_boundary_indices] .= current_interval_points.ll[boundary_indices]

    interval_points[:,new_boundary_indices[1]] .= current_interval_points.points[:,boundary_indices[1]]
    interval_points[:,new_boundary_indices[2]] .= current_interval_points.points[:,boundary_indices[2]]

    iter_inds = setdiff(1:total_points, new_boundary_indices)

    for i in iter_inds
        ll[i] = univariate_optimiser(point_locations[i], p)
        variablemapping1d!(@view(interval_points[:,i]), p.λ_opt, θranges, λranges)
        # p.initGuess .= p.λ_opt .* 1.0
    end
    interval_points[θi,iter_inds] .= point_locations[iter_inds]

    return PointsAndLogLikelihood(interval_points, ll, new_boundary_indices)
end

function get_points_in_interval_single_row(model::LikelihoodModel,
                                uni_row_number::Int,
                                num_points_in_interval::Int,
                                additional_width::Real)

    θi = model.uni_profiles_df.θindex[uni_row_number]
    profile_type = model.uni_profiles_df.profile_type[uni_row_number]
    univariate_optimiser = get_univariate_opt_func(profile_type)
    current_interval_points = model.uni_profiles_dict[uni_row_number].interval_points

    return get_points_in_interval_single_row(univariate_optimiser, model, num_points_in_interval, 
                                                θi, profile_type, current_interval_points, additional_width)
end

function get_points_in_interval!(model::LikelihoodModel,
                                    num_points_in_interval::Int;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    additional_width::Real=0.0
                                    )

    0 < num_points_in_interval || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
    additional_width >= 0 || throw(DomainError("additional_width must be greater than or equal to zero"))
    df = model.uni_profiles_df
    row_subset = df.num_points .> 0

    row_subset .= row_subset .&& (df.num_points .!= (num_points_in_interval+2) .|| 
                                    (df.additional_width .!= additional_width))
    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end

    sub_df = @view(df[row_subset, :])

    if nrow(sub_df) < 1
        return nothing
    end

    for i in 1:nrow(sub_df)
        points = get_points_in_interval_single_row(model, sub_df[i, :row_ind], 
                                            num_points_in_interval, additional_width)

        update_uni_dict_internal!(model, sub_df[i, :row_ind], points)
    end

    sub_df[:, :not_evaluated_internal_points] .= false
    sub_df[:, :not_evaluated_predictions] .= true
    sub_df[:, :num_points] .= num_points_in_interval+2
    sub_df[:, :additional_width] .= additional_width

    return nothing
end