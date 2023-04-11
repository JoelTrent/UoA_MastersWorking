function variablemapping1dranges(num_pars::T, index::T) where T <: Int
    θranges = (1:(index-1), (index+1):num_pars)
    λranges = (1:(index-1), index:(num_pars-1))
    return θranges, λranges
end

function variablemapping1d!(θ::Union{Vector, SubArray},
                            λ::Union{Vector, SubArray},
                            θranges::Tuple{T, T}, 
                            λranges::Tuple{T, T}) where T <: UnitRange
    θ[θranges[1]] .= @view(λ[λranges[1]])
    θ[θranges[2]] .= @view(λ[λranges[2]])
    return θ
end

function boundsmapping1d!(newbounds::Vector{<:Float64}, bounds::Vector{<:Float64}, index::Int)
    newbounds[1:(index-1)] .= @view(bounds[1:(index-1)])
    newbounds[index:end]   .= @view(bounds[(index+1):end])
    return nothing
end