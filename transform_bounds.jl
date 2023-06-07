"""
    checkforInf(x::AbstractVector{<:Real})

Warns via a message if any of the bounds returned given the provided forward transformation are +/-Inf.

# Arguments
- `x`: vector of transformed bounds. 
"""
function checkforInf(x::AbstractVector{<:Real})
    if any(isinf.(x))
        @warn "the specified transformation causes some of the returned bounds to be +/-Inf. It is recommended to prevent this from occurring by altering the initial bounds."
    end
    return nothing
end

"""
parameters are vectors of ints - i.e. call using vectors of ints directly or look up position of parameter
from a symbol vector using a lookup table.
Note. we assume that ordering remains the same.
A 'independentParameter' is one where the new parameter Θ[i] depends only on f(θ[i]).
A 'dependentParameter' is one where the new parameter Θ[i] depends on f(θ[i], θ[j], j!=i)

I suspect that the dependentParameter heuristic may fail if there are multiple local minima - a binary integer 
programme may be required instead (however, integer requirement on variables can be relaxed)

ONLY VALID FOR MONOTONIC (increasing or decreasing) TRANSFORMATIONS OF VARIABLES
"""
function transformbounds(transformfun::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real},
    independentParameterIndexes::Vector{<:Int}=Int[], dependentParameterIndexes::Vector{<:Int}=Int[])

    newlb, newub = zeros(length(lb)), zeros(length(lb))

    if isempty(dependentParameterIndexes)
        potentialbounds = zeros(2, length(lb))
        potentialbounds[1,:] .= transformfun(lb)
        potentialbounds[2,:] .= transformfun(ub)

        newlb[:] .= minimum(potentialbounds, dims=1)[:]
        newub[:] .= maximum(potentialbounds, dims=1)[:]

        checkforInf.((newlb, newub))
        return newlb, newub
    end

    if !isempty(independentParameterIndexes)
        potentialbounds = zeros(2, length(lb))
        potentialbounds[1,:] .= transformfun(lb)
        potentialbounds[2,:] .= transformfun(ub)
        
        newlb[independentParameterIndexes] .= minimum(@view(potentialbounds[:, independentParameterIndexes]), dims=1)[:]
        newub[independentParameterIndexes] .= maximum(@view(potentialbounds[:, independentParameterIndexes]), dims=1)[:]
    end

    for i in dependentParameterIndexes # MAKE THIS PART VECTORISED? E.g by making it recursive?
        maximisingbounds = copy(ub)
        currentMax = transformfun(maximisingbounds)[i]
        for j in eachindex(ub)
            maximisingbounds[j] = lb[j]
            candidate = transformfun(maximisingbounds)[i]

            if candidate > currentMax
                currentMax = candidate
            else
                maximisingbounds[j] = ub[j]
            end
        end

        newub[i] = currentMax * 1.0
    end

    for i in dependentParameterIndexes
        minimisingbounds = copy(lb)
        currentMin = transformfun(minimisingbounds)[i]
        for j in eachindex(ub)
            minimisingbounds[j] = ub[j]
            candidate = transformfun(minimisingbounds)[i]

            if candidate < currentMin
                currentMin = candidate
            else
                minimisingbounds[j] = lb[j]
            end
        end

        newlb[i] = currentMin * 1.0
    end

    checkforInf.((newlb, newub))
    return newlb, newub
end

# IS VALID FOR MONOTONIC (increasing or decreasing) TRANSFORMATIONS OF VARIABLES SO LONG
# AS START POSITION OF x VARIABLES PUSHES IT TOWARDS THE GLOBAL MINIMA, RATHER THAN A LOCAL 
# MINIMA
# using JuMP
# import Ipopt
# function transformbounds_NLP_JuMP(transformfun::Function, lb::Vector{<:Float64}, ub::Vector{<:Float64})

#     function bounds_transform(x...)
#         bins = collect(x)
#         bounds = (((1 .- bins) .* lb) .+ (bins .* ub)) 
#         return transformfun(bounds)[NLP_transformIndex]
#     end
    
#     num_vars = length(ub)
    
#     m = Model(Ipopt.Optimizer)
#     set_silent(m)
    
#     register(m, :my_obj, num_vars, bounds_transform; autodiff = true)
    
#     # variables will be binary integer automatically due to how the obj function is setup
#     # IF the transformation function applied to each θ[i] is monotonic between lb[i] and ub[i]
#     @variable(m, x[1:num_vars], lower_bound=0.0, upper_bound=1.0, start=0.5)
    
#     newlb = zeros(num_vars)
#     newub = zeros(num_vars)
#     NLP_transformIndex=0
#     for i in 1:num_vars
#         NLP_transformIndex += 1
        
#         @NLobjective(m, Min, my_obj(x...) )
#         JuMP.optimize!(m)
#         newlb[i] = objective_value(m)
    
#         @NLobjective(m, Max, my_obj(x...) )
#         JuMP.optimize!(m)
#         newub[i] = objective_value(m)
#     end
    
#     return newlb, newub
# end

"""
    transformbounds_NLopt(transformfun::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real})

"""
function transformbounds_NLopt(transformfun::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real})

    function bounds_transform(x)
        return minOrMax * transformfun( (((1.0 .- x) .* lb) .+ (x .* ub)) )[NLP_transformIndex]
    end
    
    # variables will be binary integer automatically due to how the obj function is setup
    # IF the transformation function applied to each θ[i] is monotonic between lb[i] and ub[i]
    num_vars = length(ub)
    
    newlb = zeros(num_vars)
    newub = zeros(num_vars)
    initialGuess = fill(0.5, num_vars)
    NLPlb = zeros(num_vars)
    NLPub = ones(num_vars)

    minOrMax = -1.0
    NLP_transformIndex=0
    for i in 1:num_vars
        NLP_transformIndex += 1
        
        minOrMax = -1.0
        newlb[i] = optimise(bounds_transform, initialGuess, NLPlb, NLPub)[2] * -1.0
        
        minOrMax = 1.0
        newub[i] = optimise(bounds_transform, initialGuess, NLPlb, NLPub)[2]
    end

    checkforInf.((newlb, newub))
    return newlb, newub
end