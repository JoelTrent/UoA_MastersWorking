# plotting functions #################
function plot1Dprofile!(plt, parRange, parProfile, llstar, parMLE; legend=false, kwargs...)

    plot!(plt, parRange, parProfile, lw=3; legend=legend, kwargs...)
    hline!(plt, [llstar], lw=3)
    vline!(plt, [parMLE], lw=3)

    return plt
end

function plot1Dprofile_comparison!(plt, parRange1, parProfile1, parRange2, parProfile2, llstar, parMLE; legend=false, kwargs...)

    plot!(plt, parRange1, parProfile1, lw=3; legend=legend, kwargs...)
    plot!(plt, parRange2, parProfile2, lw=3, linestyle=:dash)
    hline!(plt, [llstar], lw=3)
    vline!(plt, [parMLE], lw=3)

    return plt
end

function plot2Dboundary_comparison(parBoundarySamples1, parBoundarySamples2, parMLEs, N; 
    kwargs...)

    boundaryPlot=scatter([parMLEs[1]], [parMLEs[2]],
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=5; kwargs...)

    for i in 1:2*N
        boundaryPlot=scatter!([parBoundarySamples1[1][i]], [parBoundarySamples1[2][i]], 
                                markersize=3, markershape=:circle, markercolor=:blue,
                                msw=0, ms=5)
    end

    for i in 1:2*N
        boundaryPlot=scatter!([parBoundarySamples2[1][i]], [parBoundarySamples2[2][i]], 
                                markersize=3, markershape=:utriangle, markercolor=:lightblue,
                                msw=0, ms=5)
    end

    return boundaryPlot
end

function plot2Dboundary!(plt, parBoundarySamples, parMLEs; 
                        legend=false, kwargs...)

    scatter!(plt, [parMLEs[1]], [parMLEs[2]],
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=5; legend=legend, kwargs...)

    for i in axes(parBoundarySamples, 2)
        scatter!(plt, [parBoundarySamples[1,i]], [parBoundarySamples[2,i]], 
                                markersize=3, markershape=:circle, markercolor=:blue,
                                msw=0, ms=5)
    end
    return plt
end

function plotprediction(tt, predictions, confEstimate; confColor, kwargs...)

    predictionPlot = plot(tt, predictions[:,:], color=:grey; kwargs...)
    predictionPlot = plot!(tt, confEstimate[1], lw=3, color=confColor)
    predictionPlot = plot!(tt, confEstimate[2], lw=3, color=confColor)
    predictionPlot = plot!(ymle, tt[1], tt[end], lw=3, color=:turquoise1)

    return predictionPlot
end

function plotprediction_noMLE(tt, predictions, confEstimate; confColor, kwargs...)

    predictionPlot = plot(tt, predictions[:,:], color=:grey; kwargs...)
    predictionPlot = plot!(tt, confEstimate[1], lw=3, color=confColor)
    predictionPlot = plot!(tt, confEstimate[2], lw=3, color=confColor)

    return predictionPlot
end

function plotprediction_comparison(tt, predictionsFull, confFull, confEstimate, ymle; kwargs...)
    predictionPlot = plot(tt, predictionsFull[:,:], color=:grey; kwargs...)
    predictionPlot = plot!(tt, confFull[1], lw=3, color=:gold)
    predictionPlot = plot!(tt, confFull[2], lw=3, color=:gold)
    predictionPlot = plot!(tt, confEstimate[1], lw=3, linestyle=:dash, color=:red)
    predictionPlot = plot!(tt, confEstimate[2], lw=3, linestyle=:dash, color=:red)
    predictionPlot = plot!(tt, ymle, lw=3, color=:turquoise1)

    return predictionPlot
end

function plot_univariate_profiles(model::LikelihoodModel, 
                                    xlim_scaler=0.2,
                                    ylim_scaler=0.2;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[], 
                                    num_points_in_interval::Int=0, 
                                    kwargs...)


    if num_points_in_interval > 0
        get_points_in_interval!(model, num_points_in_interval, 
                                confidence_levels=confidence_levels, 
                                profile_types=profile_types)
    end

    df = model.uni_profiles_df
    row_subset = df.num_points .> 0
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

    profile_plots = [plot() for _ in 1:nrow(sub_df)]

    for i in 1:nrow(sub_df)

        row = @view(sub_df[i,:])
        interval = model.uni_profiles_dict[row.row_ind].interval_points
        boundary_col_indices = model.uni_profiles_dict[row.row_ind].interval_points.boundary_col_indices

        llstar = get_target_loglikelihood(model, row.conf_level, EllipseApprox(), 1)
        parMLE = model.core.θmle[row.θindex]
        θname = model.core.θnames[row.θindex]

        x_range = interval.points[row.θindex, boundary_col_indices[2]] - interval.points[row.θindex, boundary_col_indices[1]]

        plot1Dprofile!(profile_plots[i], interval.points[row.θindex, :], interval.ll, 
                        llstar, parMLE; xlabel=string(θname), ylabel="ll",
                        xlims=[interval.points[row.θindex,  boundary_col_indices[1]]-x_range*xlim_scaler, 
                                interval.points[row.θindex,  boundary_col_indices[2]]+x_range*xlim_scaler],
                        ylims=[llstar + llstar*ylim_scaler, 0.1],
                        title=string("Profile type: ", row.profile_type, "\nConfidence level: ", row.conf_level),
                        titlefontsize=10, kwargs...)
    end

    return profile_plots
end

function plot_bivariate_profiles(model::LikelihoodModel,
                                    xlim_scaler=0.2,
                                    ylim_scaler=0.2;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                    kwargs...)

    df = model.biv_profiles_df
    row_subset = df.num_points .> 0
    if !isempty(confidence_levels)
        row_subset .= row_subset .&& (df.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .&& (df.profile_type .∈ Ref(profile_types))
    end
    if !isempty(methods)
        row_subset .= row_subset .&& (df.method .∈ Ref(methods))
    end

    sub_df = @view(df[row_subset, :])

    if nrow(sub_df) < 1
        return nothing
    end

    profile_plots = [plot() for _ in 1:nrow(sub_df)]

    for i in 1:nrow(sub_df)

        row = @view(sub_df[i,:])
        θindices = zeros(Int,2); 
        for j in 1:2; θindices[j] = row.θindices[j] end

        full_boundary = model.biv_profiles_dict[row.row_ind].confidence_boundary
        boundary = @view(full_boundary[θindices, :])

        parMLEs = model.core.θmle[θindices]
        θnames = model.core.θnames[θindices]

        min_vals = minimum(boundary, dims=2)
        max_vals = maximum(boundary, dims=2)
        ranges = max_vals .- min_vals

        plot2Dboundary!(profile_plots[i], boundary, parMLEs; xlabel=string(θnames[1]), ylabel=string(θnames[2]),
                        xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                                max_vals[1]+ranges[1]*xlim_scaler],
                        ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
                        max_vals[2]+ranges[2]*ylim_scaler],
                        title=string("Profile type: ", row.profile_type, "\nConfidence level: ", row.conf_level),
                        titlefontsize=10, kwargs...)
    end

    return profile_plots
end