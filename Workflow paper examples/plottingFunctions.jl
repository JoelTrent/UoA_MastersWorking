# plotting functions #################
function plot1Dprofile(parRange, parProfile, llstar, parMLE; legend=false, kwargs...)

    profilePlot=plot(parRange, parProfile, lw=3; legend=legend, kwargs...)
    profilePlot=hline!([llstar], lw=3)
    profilePlot=vline!([parMLE], lw=3)

    return profilePlot
end

function plot1Dprofile_comparison(parRange1, parProfile1, parRange2, parProfile2, llstar, parMLE; legend=false, kwargs...)

    profilePlot=plot(parRange1, parProfile1, lw=3; legend=legend, kwargs...)
    profilePlot=plot!(parRange2, parProfile2, lw=3, linestyle=:dash)
    profilePlot=hline!([llstar], lw=3)
    profilePlot=vline!([parMLE], lw=3)

    return profilePlot
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

function plot2Dboundary(parBoundarySamples, parMLEs, N; kwargs...)

    boundaryPlot=scatter([parMLEs[1]], [parMLEs[2]],
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=5; kwargs...)

    for i in 1:2*N
        boundaryPlot=scatter!([parBoundarySamples[1][i]], [parBoundarySamples[2][i]], 
                                markersize=3, markershape=:circle, markercolor=:blue,
                                msw=0, ms=5)
    end
    return boundaryPlot
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