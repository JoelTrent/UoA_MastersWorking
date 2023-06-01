function findpointonbounds(model::LikelihoodModel, 
                            internalpoint::Vector{<:Float64}, 
                            direction_πradians::Float64, 
                            cosDir::Float64, 
                            sinDir::Float64, 
                            ind1::Int, 
                            ind2::Int,
                            returnboundindex::Bool=false)

    # by construction 0 < direction_πradians < 2, i.e. direction_radians ∈ [1e-10, 2 - 1e-10]
    quadrant = convert(Int, div(direction_πradians, 0.5, RoundUp))

    if quadrant == 1
        xlim, ylim = model.core.θub[ind1], model.core.θub[ind2]
    elseif quadrant == 2
        xlim, ylim = model.core.θlb[ind1], model.core.θub[ind2]
    elseif quadrant == 3
        xlim, ylim = model.core.θlb[ind1], model.core.θlb[ind2]
    else
        xlim, ylim = model.core.θub[ind1], model.core.θlb[ind2]
    end
    
    r_vals = abs.([(xlim-internalpoint[1]) / cosDir , (ylim-internalpoint[2]) / sinDir])

    r = minimum(r_vals)
    r_pos = argmin(r_vals)

    boundpoint = [0.0, 0.0]

    if r_pos == 1
        boundpoint[1] = xlim
        boundpoint[2] = internalpoint[2] + r * sinDir
        bound_ind = ind1 * 1
    else
        boundpoint[1] = internalpoint[1] + r * cosDir
        boundpoint[2] = ylim
        bound_ind = ind2 * 1
    end

    if returnboundindex
        upper_or_lower = (r_pos==1 && quadrant ∈ [1,4]) || quadrant ∈ [1,2]  ? "upper" : "lower"
        return boundpoint, bound_ind, upper_or_lower
    end

    return boundpoint
end

function findpointonbounds(model::LikelihoodModel, 
                            internalpoint::Vector{<:Float64}, 
                            direction2D::AbstractVector{Float64}, 
                            ind1::Int, 
                            ind2::Int,
                            returnboundindex::Bool=false)

    direction_πradians = atan(direction2D[2], direction2D[1]) / pi
    
    # define direction_πradians on [0,2] rather than [-1,1]
    if direction_πradians < 0
        direction_πradians = 2 + direction_πradians
    end
    
    return findpointonbounds(model, internalpoint, direction_πradians, direction2D[1], direction2D[2], ind1, ind2, returnboundindex)
end

function findpointonbounds(model::LikelihoodModel, 
                            internalpoint::Vector{<:Float64}, 
                            direction_πradians::Float64, 
                            ind1::Int, 
                            ind2::Int,
                            returnboundindex::Bool=false)

    cosDir = cospi(direction_πradians)
    sinDir = sinpi(direction_πradians)

    return findpointonbounds(model, internalpoint, direction_πradians, cosDir, sinDir, ind1, ind2, returnboundindex)
end