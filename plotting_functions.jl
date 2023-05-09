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

function profile2Dmarkershape(profile_type::AbstractProfileType, on_boundary::Bool)
    if profile_type isa EllipseApproxAnalytical
        return on_boundary ? :diamond : :hexagon
    elseif profile_type isa EllipseApprox
        return on_boundary ? :utriangle : :dtriangle
    end
    return on_boundary ? :circle : :rect
end

function θs_to_plot_typeconversion(model::LikelihoodModel,
                                    θs_to_plot::Union{Vector{<:Symbol}, Vector{<:Int64}})
    if θs_to_plot isa Vector{<:Symbol}
        return convertθnames_toindices(model, θs_to_plot)
    end
    return θs_to_plot
end

function θcombinations_to_plot_typeconversion(model::LikelihoodModel,
                                                θcombinations_to_plot::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}, Vector{Vector{Int}}, Vector{Tuple{Int,Int}}})
    if θcombinations_to_plot isa Vector{Vector{Symbol}} || θcombinations_to_plot isa Vector{Tuple{Symbol, Symbol}}
        return [tuple(sort(arr)...) for arr in convertθnames_toindices(model, θcombinations_to_plot)]
    end

    if θcombinations_to_plot isa Vector{Vector{Int}}
        return [tuple(sort(arr)...) for arr in θcombinations_to_plot] 
    end
    return θcombinations_to_plot
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
function plot2Dboundary!(plt, parBoundarySamples, label="boundary"; use_lines=false, kwargs...)
    
    plot_func! = use_lines ? plot! : scatter!
    plot_func!(plt, parBoundarySamples[1,:], parBoundarySamples[2,:], 
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
function plotprediction!(plt, t, predictions, extrema, linealpha; extremacolor=:red, kwargs...)
    plot!(plt, t, predictions, color=:grey, linealpha=linealpha; label=hcat("Predictions", fill("", 1, size(predictions, 2)-1)), kwargs...)
    plot!(plt, t, extrema, lw=3, color=extremacolor, label=["Simultaneous confidence band (≈)" ""])
    return plt
end

function add_yMLE!(plt, t, yMLE; kwargs...)
    plot!(plt, t, yMLE, lw=3, color=:turquoise1, label="MLE"; kwargs...)
    return plt
end


function add_extrema!(plt, t, extrema; extremacolor=:gold, label=["Sampled band (≈)" ""])
    plot!(plt, t, extrema, lw=3, color=extremacolor, label=label)
    return plt
end

function plotprediction_comparison(tt, predictionsFull, confFull, confEstimate, ymle; kwargs...)
    predictionPlot = plot(tt, predictionsFull, color=:grey; kwargs...)
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
                                    θs_to_plot::Vector=Int[],
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
    
    θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)

    sub_df = desired_df_subset(model.uni_profiles_df, θs_to_plot, confidence_levels, profile_types)

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
                                    θs_to_plot::Vector=Int[],
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

    θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)

    sub_df = desired_df_subset(model.uni_profiles_df, θs_to_plot, confidence_levels, profile_types)

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
                                    xlim_scaler::Real=0.2,
                                    ylim_scaler::Real=0.2;
                                    θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                    palette_to_use::Symbol=:Paired_6, 
                                    include_internal_points::Bool=true,
                                    markeralpha=1.0,
                                    kwargs...)

    θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)

    sub_df = desired_df_subset(model.biv_profiles_df, θcombinations_to_plot, confidence_levels,
                                profile_types, methods)
    
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
                            use_lines=row.method isa ContinuationMethod,
                            markershape=profile2Dmarkershape(row.profile_type, true),
                            markercolor=color_palette[profilecolor(row.profile_type)],
                            linecolor=color_palette[profilecolor(row.profile_type)],
                            markeralpha=markeralpha,
                            linealpha=markeralpha)

        if include_internal_points && !row.not_evaluated_internal_points
            plot2Dboundary!(profile_plots[i], 
                            @view(model.biv_profiles_dict[row.row_ind].internal_points[θindices, :]),
                            "internal points", 
                            use_lines=row.method isa ContinuationMethod,
                            markershape=profile2Dmarkershape(row.profile_type, false), 
                            markercolor=color_palette[profilecolor(row.profile_type)], 
                            linecolor=color_palette[profilecolor(row.profile_type)],
                            markeralpha=markeralpha*0.5,
                            linealpha=markeralpha*0.5)
        end

        addMLE!(profile_plots[i], parMLEs; 
            markercolor=color_palette[end],
            xlabel=string(θnames[1]), ylabel=string(θnames[2]),
            xlims=[min_vals[1]-ranges[1]*xlim_scaler, 
                    max_vals[1]+ranges[1]*xlim_scaler],
            ylims=[min_vals[2]-ranges[2]*ylim_scaler, 
            max_vals[2]+ranges[2]*ylim_scaler],
            title=string("Profile type: ", row.profile_type, 
                        "\nMethod: ", row.method,
                        "\nConfidence level: ", row.conf_level),
                        titlefontsize=10, kwargs...)
                    end

    return profile_plots
end

function plot_bivariate_profiles_comparison(model::LikelihoodModel,
                                    xlim_scaler=0.2,
                                    ylim_scaler=0.2;
                                    θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                    compare_within_methods::Bool=false,
                                    palette_to_use::Symbol=:Paired_6, 
                                    markeralpha=0.7,
                                    kwargs...)

    θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)

    sub_df = desired_df_subset(model.biv_profiles_df, θcombinations_to_plot, confidence_levels,
                                profile_types, methods)

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
                            markershape=profile2Dmarkershape(row.profile_type, true), 
                            markercolor=color_palette[profilecolor(row.profile_type)],
                            linecolor=color_palette[profilecolor(row.profile_type)],
                            markeralpha=markeralpha,
                            linealpha=markeralpha)
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
                        markershape=profile2Dmarkershape(prof_type, true), 
                        markercolor=color_palette[profilecolor(prof_type)],
                        linecolor=color_palette[profilecolor(prof_type)],
                        markeralpha=markeralpha,
                        linealpha=markeralpha)
                    
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
                    title=string("Confidence level: ", confidence_level),
                    titlefontsize=10, kwargs...)

                plot_i+=1
            end
        end
    end

    return profile_plots
end

function plot_predictions_individual(model::LikelihoodModel,
                            # prediction_type::Symbol=:union,
                            t::Vector,
                            profile_dimension::Int=1,
                            xlabel::String="t",
                            ylabel::String="f(t)";
                            for_dim_samples::Bool=false,
                            # ylim_scaler::Real=0.2;
                            include_MLE::Bool=true,
                            θs_to_plot::Vector=Int[],
                            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                            θindices_to_plot::Vector=Vector{Int}[],
                            confidence_levels::Vector{<:Float64}=Float64[],
                            profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                            # palette_to_use::Symbol=:Paired_6, 
                            linealpha=0.4, 
                            kwargs...)
                            
    if for_dim_samples
        if !(1 ≤ profile_dimension && profile_dimension ≤ model.core.num_pars)
            throw(DomainError(string("profile_dimension must be between 1 and ", 
                    model.core.num_pars, " (the number of model parameters)")))
        end

        θindices_to_plot = θindices_to_plot isa Vector{Vector{Symbol}} ? 
                            convertθnames_toindices(model, θindices_to_plot) :
                            θindices_to_plot
        sub_df = desired_df_subset(model.dim_samples_df, confidence_levels, sample_types;
                                    sample_dimension=profile_dimension, for_prediction_plots=true)
        predictions_dict = model.dim_predictions_dict
    else
        profile_dimension in [1,2] || throw(DomainError("profile_dimension must be 1 or 2"))

        if profile_dimension == 1
            θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)
            sub_df = desired_df_subset(model.uni_profiles_df, θs_to_plot, confidence_levels,
                                        profile_types; for_prediction_plots=true)
            predictions_dict = model.uni_predictions_dict
        elseif profile_dimension == 2
            θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)
            sub_df = desired_df_subset(model.biv_profiles_df, θcombinations_to_plot, confidence_levels,
                                        profile_types, methods; for_prediction_plots=true)
            predictions_dict = model.biv_predictions_dict
        end
    end

    if nrow(sub_df) < 1
        return nothing
    end
    
    # color_palette = palette(palette_to_use)
    prediction_plots = [plot() for _ in 1:nrow(sub_df)]

    for i in 1:nrow(sub_df)

        row = @view(sub_df[i,:])
        predictions = predictions_dict[row.row_ind].predictions
        extrema = predictions_dict[row.row_ind].extrema

        if for_dim_samples
            title=string("Sample type: ", row.sample_type, 
                        "\nConfidence level: ", row.conf_level,
                        "\nTarget parameter(s): ", model.core.θnames[row.θindices])
        else
            if profile_dimension == 1
                title=string("Profile type: ", row.profile_type, 
                            "\nConfidence level: ", row.conf_level,
                            "\nTarget parameter: ", model.core.θnames[row.θindex])
            else
                θindices = zeros(Int,2); 
                for j in 1:2; θindices[j] = row.θindices[j] end

                title=string("Profile type: ", row.profile_type, 
                            "\nMethod: ", row.method, 
                            "\nConfidence level: ", row.conf_level,
                            "\nTarget parameters: ", model.core.θnames[θindices])
            end
        end
        
        # y_extrema = [minimum(extrema[:,1]), maximum(extrema[:,2])]
        # range = diff(y_extrema)[1]

        # ylims=[y_extrema[1]-range*ylim_scaler, y_extrema[2]+range*ylim_scaler]

        plotprediction!(prediction_plots[i], t, predictions, extrema, linealpha; 
                        xlabel=xlabel, ylabel=ylabel,
                        # ylims=ylims,
                        title=title,
                        titlefontsize=10, 
                        kwargs...)
    end

    if include_MLE 
        ymle = model.core.predictfunction(model.core.θmle, model.core.data, t)
        for plt in prediction_plots
            add_yMLE!(plt, t, ymle)
        end
    end

    return prediction_plots
end

"""
include_lower_confidence_levels is only used for 2d bivariate boundaries
"""
function plot_predictions_union(model::LikelihoodModel,
                            t::Vector,
                            profile_dimension::Int=1,
                            xlabel::String="t",
                            ylabel::String="f(t)",
                            confidence_level::Float64=0.95;
                            for_dim_samples::Bool=false,
                            include_MLE::Bool=true,
                            θs_to_plot::Vector = Int[],
                            θcombinations_to_plot::Vector=Tuple{Int,Int}[],
                            θindices_to_plot::Vector=Vector{Int}[],
                            profile_types::Vector{<:AbstractProfileType}=[LogLikelihood()],
                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                            # palette_to_use::Symbol=:Paired_6, 
                            # union_within_methods::Bool=false,
                            compare_to_full_sample_type::Union{Missing, AbstractSampleType}=missing,
                            include_lower_confidence_levels::Bool=false,
                            linealpha=0.4,
                            kwargs...)

    if for_dim_samples
        if !(1 ≤ profile_dimension && profile_dimension ≤ model.core.num_pars)
                throw(DomainError(string("profile_dimension must be between 1 and ", 
                      model.core.num_pars, " (the number of model parameters)")))
        end

        θindices_to_plot = θindices_to_plot isa Vector{Vector{Symbol}} ? 
                            convertθnames_toindices(model, θindices_to_plot) :
                            θindices_to_plot
        sub_df = desired_df_subset(model.dim_samples_df, confidence_level, sample_types; 
                                    sample_dimension=profile_dimension, for_prediction_plots=true)
        predictions_dict = model.dim_predictions_dict
        title = string("Parameter dimension: " , profile_dimension,
                        "\nMethod: sampled",
                        "\nConfidence level: ", confidence_level)
    else
        profile_dimension in [1,2] || throw(DomainError("profile_dimension must be 1 or 2"))
        if profile_dimension == 1
            θs_to_plot = θs_to_plot_typeconversion(model, θs_to_plot)
            sub_df = desired_df_subset(model.uni_profiles_df, θs_to_plot, confidence_level, profile_types; for_prediction_plots=true)
            predictions_dict = model.uni_predictions_dict
        elseif profile_dimension == 2
            θcombinations_to_plot = θcombinations_to_plot_typeconversion(model, θcombinations_to_plot)
            sub_df = desired_df_subset(model.biv_profiles_df, θcombinations_to_plot, confidence_level, profile_types, methods; for_prediction_plots=true, include_lower_confidence_levels)
            predictions_dict = model.biv_predictions_dict
        end

        title = string("Parameter dimension: " , profile_dimension,
                        "\nMethod: boundary",
                        "\nConfidence level: ", confidence_level)
    end

    if nrow(sub_df) < 1
        return nothing
    end

    prediction_plot = plot()

    predictions_union = []
    extrema_union = []

    for i in 1:nrow(sub_df)
        row = @view(sub_df[i,:])
        predictions = predictions_dict[row.row_ind].predictions
        extrema = predictions_dict[row.row_ind].extrema
        
        if i == 1
            predictions_union = predictions .* 1.0
            extrema_union = extrema .* 1.0
        else
            predictions_union = reduce(hcat, (predictions_union, predictions))
            extrema_union[:,1] .= min.(extrema_union[:,1], extrema[:,1])
            extrema_union[:,2] .= max.(extrema_union[:,2], extrema[:,2])
        end
    end

    plotprediction!(prediction_plot, t, predictions_union, extrema_union, linealpha; 
                    xlabel=xlabel, ylabel=ylabel,
                    # ylims=ylims,
                    title=title,
                    titlefontsize=10, 
                    kwargs...)

    if !ismissing(compare_to_full_sample_type)
        row_subset = desired_df_subset(model.dim_samples_df, confidence_level,
                                        [compare_to_full_sample_type], for_prediction_plots=true,
                                        include_higher_confidence_levels=true)
        
        if nrow(row_subset) > 0
            if nrow(row_subset) > 1
                subset_to_use = @view(row_subset[argmin(row_subset.conf_level), :])
            elseif nrow(row_subset) == 1
                subset_to_use = @view(row_subset[1,:])
            end

            if subset_to_use.conf_level == confidence_level
                sampled_extrema = extrema = model.dim_predictions_dict[subset_to_use.row_ind].extrema
            else
                ll_level = get_target_loglikelihood(model, confidence_level, 
                                                    EllipseApproxAnalytical(), model.core.num_pars)

                valid_point = model.dim_samples_dict[subset_to_use.row_ind].ll .> ll_level 
                sampled_extrema = model.dim_predictions_dict[subset_to_use.row_ind].extrema[valid_point]
            end

            add_extrema!(prediction_plot, t, sampled_extrema)
        end
    end

    if include_MLE 
        ymle = model.core.predictfunction(model.core.θmle, model.core.data, t)
        add_yMLE!(prediction_plot, t, ymle)
    end

    return prediction_plot
end

function plot_predictions_sampled(model::LikelihoodModel,
                                    t::Vector,
                                    xlabel::String="t",
                                    ylabel::String="f(t)";
                                    include_MLE::Bool=true,
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                    linealpha=0.4,
                                    kwargs...)

    sub_df = desired_df_subset(model.dim_samples_df, confidence_levels, sample_types,
                                for_prediction_plots=true)

    if nrow(sub_df) < 1
        return nothing
    end

    prediction_plots = [plot() for _ in 1:nrow(sub_df)]

    for i in 1:nrow(sub_df)
        row = @view(sub_df[i,:])
        predictions = model.dim_predictions_dict[row.row_ind].predictions
        extrema = model.dim_predictions_dict[row.row_ind].extrema
        title = string("Sample type: ", row.sample_type,
                        "\nConfidence level: ", row.conf_level)

        plotprediction!(prediction_plots[i], t, predictions, extrema, linealpha; 
                        extremacolor=:gold,
                        xlabel=xlabel, ylabel=ylabel,
                        # ylims=ylims,
                        title=title,
                        titlefontsize=10, 
                        kwargs...)
    end

    if include_MLE 
        ymle = model.core.predictfunction(model.core.θmle, model.core.data, t)
        for plt in prediction_plots
            add_yMLE!(plt, t, ymle)
        end
    end

    return prediction_plots
end