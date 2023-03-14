# plotting functions #################
function plot1DProfile(parRange, parProfile, llstar, parMLE; 
    xlims, ylims, xlabel, ylabel, legend=false)

    profilePlot=plot(parRange, parProfile, ylims=ylims, xlims=xlims, legend=legend, lw=3)
    profilePlot=hline!([llstar], lw=3)
    profilePlot=vline!([parMLE], lw=3, xlabel=xlabel, ylabel=ylabel)

    return profilePlot
end

function plot1DProfileComparison(parRange1, parProfile1, parRange2, parProfile2, llstar, parMLE; 
    xlims, ylims, xlabel, ylabel, legend=false)

    profilePlot=plot(parRange1, parProfile1, ylims=ylims, xlims=xlims, legend=legend, lw=3)
    profilePlot=plot!(parRange2, parProfile2, ylims=ylims, xlims=xlims, legend=legend, lw=3,
                        linestyle=:dash)
    profilePlot=hline!([llstar], lw=3)
    profilePlot=vline!([parMLE], lw=3, xlabel=xlabel, ylabel=ylabel)

    return profilePlot
end

function plot2DBoundaryComparison(parBoundarySamples1, parBoundarySamples2, parMLEs, N; 
    xlims, ylims, xticks, yticks, xlabel, ylabel, legend=false)

    boundaryPlot=scatter([parMLEs[1]], [parMLEs[2]], xlims=xlims, ylims=ylims, 
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=5, 
            xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks, legend=legend)

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

function plot2DBoundary(parBoundarySamples, parMLEs, N; 
    xlims, ylims, xticks, yticks, xlabel, ylabel, legend=false)

    boundaryPlot=scatter([parMLEs[1]], [parMLEs[2]], xlims=xlims, ylims=ylims, 
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=5, 
            xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks, legend=legend)

    for i in 1:2*N
        boundaryPlot=scatter!([parBoundarySamples[1][i]], [parBoundarySamples[2][i]], 
                                markersize=3, markershape=:circle, markercolor=:blue,
                                msw=0, ms=5)
    end
    return boundaryPlot
end

function plot2DBoundaryNoTicks(parBoundarySamples, parMLEs, N; 
    xlims, ylims, xlabel, ylabel, legend=false)

    boundaryPlot=scatter([parMLEs[1]], [parMLEs[2]], xlims=xlims, ylims=ylims, 
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=5, 
            xlabel=xlabel, ylabel=ylabel, legend=legend)

    for i in 1:2*N
        boundaryPlot=scatter!([parBoundarySamples[1][i]], [parBoundarySamples[2][i]], 
                                markersize=3, markershape=:circle, markercolor=:blue,
                                msw=0, ms=5)
    end
    return boundaryPlot
end

function plotPrediction(tt, predictions, confEstimate; confColor, xlabel,
    ylabel, ylims, xticks, yticks, legend=false)

    predictionPlot = plot(tt, predictions[:,:], color=:grey, xlabel=xlabel, ylabel=ylabel, 
                            ylims=ylims, xticks=xticks, yticks=yticks, legend=legend)
    predictionPlot = plot!(tt, confEstimate[1], lw=3, color=confColor)
    predictionPlot = plot!(tt, confEstimate[2], lw=3, color=confColor)
    predictionPlot = plot!(ymle, tt[1], tt[end], lw=3, color=:turquoise1)

    return predictionPlot
end

function plotPredictionNoMLE(tt, predictions, confEstimate; confColor, xlabel,
    ylabel, xlims, ylims, legend=false)

    predictionPlot = plot(tt, predictions[:,:], color=:grey, xlabel=xlabel, ylabel=ylabel, 
                            xlims=xlims, ylims=ylims,
                            legend=legend)
    predictionPlot = plot!(tt, confEstimate[1], lw=3, color=confColor)
    predictionPlot = plot!(tt, confEstimate[2], lw=3, color=confColor)

    return predictionPlot
end

function plotPredictionComparison(tt, predictionsFull, confFull, confEstimate; xlabel,
    ylabel, ylims, xticks, yticks, legend=false)

    predictionPlot = plot(tt, predictionsFull[:,:], color=:grey, xlabel=xlabel, ylabel=ylabel, 
                            ylims=ylims, xticks=xticks, yticks=yticks, legend=legend)
    predictionPlot = plot!(tt, confFull[1], lw=3, color=:gold)
    predictionPlot = plot!(tt, confFull[2], lw=3, color=:gold)
    predictionPlot = plot!(tt, confEstimate[1], lw=3, linestyle=:dash, color=:red)
    predictionPlot = plot!(tt, confEstimate[2], lw=3, linestyle=:dash, color=:red)
    predictionPlot = plot!(ymle, tt[1], tt[end], lw=3, color=:turquoise1)

    return predictionPlot
end

function plotPredictionComparisonNoTicks(tt, predictionsFull, confFull, confEstimate, mle; xlabel,
    ylabel, xlims, ylims, legend=false)

    predictionPlot = plot(tt, predictionsFull[:,:], color=:grey, xlabel=xlabel, ylabel=ylabel, 
                            xlims=xlims, ylims=ylims, legend=legend)
    predictionPlot = plot!(tt, confFull[1], lw=3, color=:gold)
    predictionPlot = plot!(tt, confFull[2], lw=3, color=:gold)
    predictionPlot = plot!(tt, confEstimate[1], lw=3, linestyle=:dash, color=:red)
    predictionPlot = plot!(tt, confEstimate[2], lw=3, linestyle=:dash, color=:red)
    predictionPlot = plot!(tt, mle, lw=3, color=:turquoise1)

    return predictionPlot
end