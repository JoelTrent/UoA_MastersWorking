using JuMP
import Ipopt

# parameters are vectors of ints - i.e. call using vectors of ints directly or look up position of parameter
# from a symbol vector using a lookup table.
# Note. we assume that ordering remains the same.
# A 'independentParameter' is one where the new parameter Θ[i] depends only on f(θ[i]).
# A 'dependentParameter' is one where the new parameter Θ[i] depends on f(θ[i], θ[j], j!=i)
# 
# I suspect that the dependentParameter heuristic may fail if there are multiple local minima - a binary integer 
# programme may be required instead (however, integer requirement on variables can be relaxed)
# 
# ONLY VALID FOR MONOTONIC (increasing or decreasing) TRANSFORMATIONS OF VARIABLES
function transformbounds(transformfun::Function, lb, ub,
    independentParameterIndexes::Vector{<:Int}=Int[], dependentParameterIndexes::Vector{<:Int}=Int[])

    if isempty(dependentParameterIndexes)
        newlb = transformfun(lb)
        newub = transformfun(ub)
        return newlb, newub
    end

    newlb = lb .* 1.0
    newub = ub .* 1.0

    if !isempty(independentParameterIndexes)
        potentiallb = transformfun(lb)
        potentialub = transformfun(ub)
        
        newlb[independentParameterIndexes] .= @view(potentiallb[independentParameterIndexes])
        newub[independentParameterIndexes] .= @view(potentialub[independentParameterIndexes])
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

    return newlb, newub
end


# IS VALID FOR MONOTONIC (increasing or decreasing) TRANSFORMATIONS OF VARIABLES SO LONG
# AS START POSITION OF x VARIABLES PUSHES IT TOWARDS THE GLOBAL MINIMA, RATHER THAN A LOCAL 
# MINIMA
function transformbounds_NLP(transformfun::Function, lb, ub)

    function bounds_transform(x...)
        bins = collect(x)
        bounds = ((bins .* lb) .+ ((1 .- bins) .* ub)) 
        return transformfun(bounds)[NLP_transformIndex]
    end
    
    num_vars = length(ub)
    
    m = Model(Ipopt.Optimizer)
    set_silent(m)
    
    register(m, :my_obj, num_vars, bounds_transform; autodiff = true)
    
    # variables will be binary integer automatically due to how the obj function is setup
    # IF the transformation function applied to each θ[i] is monotonic between lb[i] and ub[i]
    @variable(m, x[1:num_vars], lower_bound=0.0, upper_bound=1.0, start=0.5)
    
    newlb = zeros(num_vars)
    newub = zeros(num_vars)
    NLP_transformIndex=0
    for i in 1:num_vars
        NLP_transformIndex += 1
        
        @NLobjective(m, Min, my_obj(x...) )
        JuMP.optimize!(m)
        newlb[i] = objective_value(m)
    
        @NLobjective(m, Max, my_obj(x...) )
        JuMP.optimize!(m)
        newub[i] = objective_value(m)
    end
    
    return newlb, newub
end