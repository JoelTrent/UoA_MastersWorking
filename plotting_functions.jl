function profilecolor(profile_type::AbstractProfileType)
    if profile_type isa EllipseApproxAnalytical
        return 1
    elseif profile_type isa EllipseApprox
        return 2
    end
    return 3        
end

function profile1Dlinestyle(profile_type::AbstractProfileType)
    if profile_type isa EllipseApproxAnalytical
        return :dash
    elseif profile_type isa EllipseApprox
        return :dashdot
    end
    return :solid
end

function profile2Dmarkershape(profile_type::AbstractProfileType)
    if profile_type isa EllipseApproxAnalytical
        return :diamond
    elseif profile_type isa EllipseApprox
        return :utriangle
    end
    return :circle
end

function subset_of_interest(df::DataFrame, 
                            confidence_levels::Vector{<:Float64},
                            profile_types::Vector{<:AbstractProfileType},
                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[])

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

    return @view(df[row_subset, :])
end

# plotting functions #################

# 1D
function plot1Dprofile!(plt, parRange, parProfile, label="profile"; kwargs...)
    plot!(plt, parRange, parProfile, lw=3, label=label; kwargs...)
    return plt
end

function addMLEandLLstar!(plt, llstar, parMLE, MLE_color, llstar_color; kwargs...)
    vline!(plt, [parMLE], lw=2, color=MLE_color, label="MLE point", linestyle=:dash)
    hline!(plt, [llstar], lw=2, color=llstar_color, label="ll cutoff", linestyle=:dash; kwargs...)
    return plt
end

# 2D
function plot2Dboundary!(plt, parBoundarySamples, label="boundary"; kwargs...)
    scatter!(plt, parBoundarySamples[1,:], parBoundarySamples[2,:], 
                            markersize=3,
                            msw=0, ms=5,
                            label=label;
                            kwargs...)
    return plt
end

function addMLE!(plt, parMLEs; kwargs...)
    scatter!(plt, [parMLEs[1]], [parMLEs[2]],
            markersize=3, markershape=:circle,
            msw=0, ms=5,
            label="MLE point"; kwargs...)
    return plt
end

# Predictions
function plotprediction(plt, tt, predictions, extrema; extrema_color=:gold, kwargs...)
    plot!(plt, tt, predictions[:,:], color=:grey; kwargs...)
    plot!(plt, tt, extrema, lw=3, color=confColor)
    return plt
end

function add_yMLE!(plt, ymle, tt; kwargs...)
    plot!(plt, ymle, tt[1], tt[end], lw=3, color=:turquoise1; kwargs...)
    return plt
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
                                    palette_to_use::Symbol=:Paired_6, 
                                    kwargs...)

    if num_points_in_interval > 0
        get_points_in_interval!(model, num_points_in_interval, 
                                confidence_levels=confidence_levels, 
                                profile_types=profile_types)
    end

    sub_df = subset_of_interest(model.uni_profiles_df, confidence_levels, profile_types)

    if nrow(sub_df) < 1
        return nothing
    end
    
    color_palette = palette(palette_to_use)
    profile_plots = [plot() for _ in 1:nrow(sub_df)]

    for i in 1:nrow(sub_df)

        row = @view(sub_df[i,:])
        interval = model.uni_profiles_dict[row.row_ind].interval_points
        boundary_col_indices = model.uni_profiles_dict[row.row_ind].interval_points.boundary_col_indices

        llstar = get_target_loglikelihood(model, row.conf_level, EllipseApprox(), 1)
        parMLE = model.core.θmle[row.θindex]
        θname = model.core.θnames[row.θindex]
        
        x_range = interval.points[row.θindex, boundary_col_indices[2]] - interval.points[row.θindex, boundary_col_indices[1]]
        
        plot1Dprofile!(profile_plots[i], interval.points[row.θindex, :], interval.ll; 
        xlims=[interval.points[row.θindex,  boundary_col_indices[1]]-x_range*xlim_scaler, 
        interval.points[row.θindex,  boundary_col_indices[2]]+x_range*xlim_scaler],
        color=color_palette[profilecolor(row.profile_type)], kwargs...)
        
        addMLEandLLstar!(profile_plots[i], llstar, parMLE, color_palette[end-1], color_palette[end]; 
                        xlabel=string(θname), ylabel="ll", 
                        ylims=[llstar + llstar*ylim_scaler, 0.1],
                        title=string("Profile type: ", row.profile_type, 
                                        "\nConfidence level: ", row.conf_level),
                        titlefontsize=10)
        
    end

    return profile_plots
end

function plot_univariate_profiles_comparison(model::LikelihoodModel, 
                                    xlim_scaler=0.2,
                                    ylim_scaler=0.2;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=[EllipseApprox(), LogLikelihood()], 
                                    num_points_in_interval::Int=0,
                                    palette_to_use::Symbol=:Paired_6, 
                                    kwargs...)

    if num_points_in_interval > 0
        get_points_in_interval!(model, num_points_in_interval, 
                                confidence_levels=confidence_levels, 
                                profile_types=profile_types)
    end

    sub_df = subset_of_interest(model.uni_profiles_df, confidence_levels, profile_types)

    if nrow(sub_df) < 1
        return nothing
    end

    if isempty(confidence_levels)
        confidence_levels = unique(sub_df.conf_level)
    end

    color_palette = palette(palette_to_use)
    profile_plots = [plot()]
    plot_i=1

    row_subset = trues(nrow(sub_df))

    for θi in 1:model.core.num_pars
        for confidence_level in confidence_levels

            row_subset .= (sub_df.θindex .== θi) .&& (sub_df.conf_level .== confidence_level)
            conf_df = @view(sub_df[row_subset, :])

            if nrow(conf_df) > 1
                if plot_i > 1
                    append!(profile_plots, [plot()])
                end
                llstar = get_target_loglikelihood(model, confidence_level, EllipseApprox(), 1)
                parMLE = model.core.θmle[θi]
                θname = model.core.θnames[θi]

                xlims = zeros(2)

                for i in 1:nrow(conf_df)

                    row = @view(conf_df[i,:])
                    interval = model.uni_profiles_dict[row.row_ind].interval_points
                    boundary_col_indices = model.uni_profiles_dict[row.row_ind].interval_points.boundary_col_indices
                    
                    x_range = interval.points[row.θindex, boundary_col_indices[2]] - interval.points[row.θindex, boundary_col_indices[1]]

                    if i == 1
                        xlims .= [interval.points[row.θindex, boundary_col_indices[1]] - x_range*xlim_scaler, 
                            interval.points[row.θindex, boundary_col_indices[2]] + x_range*xlim_scaler]
                    else
                        xlims[1] = min(xlims[1], interval.points[row.θindex, boundary_col_indices[1]] - x_range*xlim_scaler) 
                        xlims[2] = max(xlims[2], interval.points[row.θindex, boundary_col_indices[2]] + x_range*xlim_scaler)
                    end
                    
                    plot1Dprofile!(profile_plots[plot_i], interval.points[row.θindex, :], interval.ll; 
                                    label=string(row.profile_type),
                                    linestyle=profile1Dlinestyle(row.profile_type),
                                    color=color_palette[profilecolor(row.profile_type)], kwargs...)
                end

                addMLEandLLstar!(profile_plots[plot_i], llstar, parMLE, color_palette[end-1], color_palette[end], 
                                xlabel=string(θname), ylabel="ll", 
                                xlims=xlims,
                                ylims=[llstar + llstar*ylim_scaler, 0.1],
                                title=string("Confidence level: ", confidence_level),
                                titlefontsize=10)

                plot_i+=1
            end
        end
    end

    return profile_plots
end

function plot_bivariate_profiles(model::LikelihoodModel,
                                    xlim_scaler=0.2,
                                    ylim_scaler=0.2;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                    palette_to_use::Symbol=:Paired_6, 
                                    markeralpha=nothing,
                                    kwargs...)

    sub_df = subset_of_interest(model.biv_profiles_df, confidence_levels, profile_types, methods)
    
    if nrow(sub_df) < 1
        return nothing
    end
    
    color_palette = palette(palette_to_use)
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
        
        plot2Dboundary!(profile_plots[i], boundary, 
                            markershape=:circle, 
                            markercolor=color_palette[profilecolor(row.profile_type)],
                            markeralpha=markeralpha)

        addMLE!(profile_plots[i], parMLEs; 
            markercolor=color_palette[end],
            xlabel=string(θnames[1]), ylabel=string(θnames[2]),
            xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                    max_vals[1]+ranges[1]*xlim_scaler],
            ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
            max_vals[2]+ranges[2]*ylim_scaler],
            title=string("Profile type: ", row.profile_type, 
                        "\nConfidence level: ", row.conf_level),
                        titlefontsize=10, kwargs...)
                    end

    return profile_plots
end

function plot_bivariate_profiles_comparison(model::LikelihoodModel,
                                    xlim_scaler=0.2,
                                    ylim_scaler=0.2;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                    compare_within_methods::Bool=false,
                                    palette_to_use::Symbol=:Paired_6, 
                                    markeralpha=0.7,
                                    kwargs...)


    sub_df = subset_of_interest(model.biv_profiles_df, confidence_levels, profile_types, methods)

    if nrow(sub_df) < 1
        return nothing
    end

    if isempty(confidence_levels)
        confidence_levels = unique(sub_df.conf_level)
    end

    if compare_within_methods
        if isempty(methods); methods = unique(sub_df.method) end
    else
        if isempty(profile_types); profile_types = unique(sub_df.profile_type) end
    end

    color_palette = palette(palette_to_use)
    profile_plots = [plot() for _ in 1:nrow(sub_df)]

    profile_plots = [plot()]
    plot_i=1

    row_subset = trues(nrow(sub_df))

    θcombinations = unique(sub_df.θindices)
    θindices = zeros(Int,2); 

    for θindices_tuple in θcombinations
        for j in 1:2; θindices[j] = θindices_tuple[j] end
        parMLEs = model.core.θmle[θindices]
        θnames = model.core.θnames[θindices]

        for confidence_level in confidence_levels
            if compare_within_methods
                for method in methods
                    row_subset .= (sub_df.θindices .== Ref(θindices_tuple)) .&& 
                                    (sub_df.conf_level .== confidence_level) .&&
                                    (sub_df.method .== Ref(method))

                    conf_df = @view(sub_df[row_subset, :])

                    if nrow(conf_df) < 2; continue end

                    if plot_i > 1
                        append!(profile_plots, [plot()])
                    end
                    
                    min_vals = zeros(2)
                    max_vals = zeros(2)

                    for i in 1:nrow(conf_df)
                        row = @view(conf_df[i,:])

                        full_boundary = model.biv_profiles_dict[row.row_ind].confidence_boundary
                        boundary = @view(full_boundary[θindices, :])

                        if i == 1
                            min_vals .= minimum(boundary, dims=2)
                            max_vals .= maximum(boundary, dims=2)
                        else
                            min_vals .= min.(min_vals, minimum(boundary, dims=2))
                            max_vals .= max.(max_vals, maximum(boundary, dims=2))
                        end

                        plot2Dboundary!(profile_plots[plot_i], boundary, 
                            label=string(row.profile_type),
                            markershape=profile2Dmarkershape(row.profile_type), 
                            markercolor=color_palette[profilecolor(row.profile_type)],
                            markeralpha=markeralpha)
                    end

                    ranges = max_vals .- min_vals

                    addMLE!(profile_plots[plot_i], parMLEs; 
                        markercolor=color_palette[end],
                        xlabel=string(θnames[1]), ylabel=string(θnames[2]),
                        xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                                max_vals[1]+ranges[1]*xlim_scaler],
                        ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
                        max_vals[2]+ranges[2]*ylim_scaler],
                        title=string("Method: ", method, 
                                    "\nConfidence level: ", confidence_level),
                        titlefontsize=10, kwargs...)

                    plot_i+=1
                end
            else
                row_subset .= (sub_df.θindices .== Ref(θindices_tuple)) .&& 
                                    (sub_df.conf_level .== confidence_level)

                conf_df = @view(sub_df[row_subset, :])

                if !(nrow(conf_df) > 1 && length(unique(conf_df.profile_type)) > 1)
                    continue
                end

                if plot_i > 1
                    append!(profile_plots, [plot()])
                end
                
                min_vals = zeros(2)
                max_vals = zeros(2)
                i = 1
                for prof_type in profile_types
                    rows = @view(conf_df[conf_df.profile_type .== Ref(prof_type),:])
                    boundary = zeros(2,0)

                    if nrow(rows) == 0
                        continue
                    elseif nrow(rows) == 1
                        full_boundary = model.biv_profiles_dict[rows.row_ind[1]].confidence_boundary
                        boundary = @view(full_boundary[θindices, :])
                    else
                        for j in 1:nrow(rows)
                            full_boundary = model.biv_profiles_dict[rows.row_ind[j]].confidence_boundary
                            
                            boundary = reduce(hcat, (boundary, full_boundary[θindices, :]))
                        end
                    end


                    if i == 1
                        min_vals .= minimum(boundary, dims=2)
                        max_vals .= maximum(boundary, dims=2)
                    else
                        min_vals .= min.(min_vals, minimum(boundary, dims=2))
                        max_vals .= max.(max_vals, maximum(boundary, dims=2))
                    end

                    plot2Dboundary!(profile_plots[plot_i], boundary, 
                        label=string(prof_type),
                        markershape=profile2Dmarkershape(prof_type), 
                        markercolor=color_palette[profilecolor(prof_type)],
                        markeralpha=markeralpha)
                    
                    i += 1
                end

                ranges = max_vals .- min_vals

                addMLE!(profile_plots[plot_i], parMLEs; 
                    markercolor=color_palette[end],
                    xlabel=string(θnames[1]), ylabel=string(θnames[2]),
                    xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                            max_vals[1]+ranges[1]*xlim_scaler],
                    ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
                    max_vals[2]+ranges[2]*ylim_scaler],
                    title=string("\nConfidence level: ", confidence_level),
                    titlefontsize=10, kwargs...)

                plot_i+=1
            end
        end
    end

    return profile_plots
end

function plotpredictions(model::LikelihoodModel,
                            prediction_type::Symbol=:union,
                            dimensions::Vector{<:Int}=[1,2];
                            include_MLE=type==:union,
                            confidence_levels::Vector{<:Float64}=Float64[],
                            profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[])

    


end