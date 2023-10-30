using Distributions, StaticArrays

"""
    univariateUnimodalHDR(d::UnivariateDistribution, region::Real)
"""
function univariateUnimodalHDR(d::UnivariateDistribution, region::Real, test_arguments::Bool=true)
    
    # if test_arguments
    # end
    if region < 0.0 || region > 1.0
        throw(ArgumentError("region must be ∈ [0.0, 1.0]"))
        return nothing
    end
    if region == 0.0
        mode_d = mode(d)
        return SA[mode_d, mode_d]
    end
    if region == 1.0
        support_d = support(d)
        return SA[support_d.lb, support_d.ub]
    end
    
    # bisection esque approach
    l=0.0
    r=1.0-region
    c=(r-l)/2.
   
    l_interval = SA[quantile(d, l), quantile(d, l+region)]
    r_interval = SA[quantile(d, r), quantile(d, 0.999999999999)] 
    l_width = l_interval[2] - l_interval[1]
    r_width = r_interval[2] - r_interval[1]

    if l_width < r_width
        current_best = l * 1.0
        current_width = l_width * 1.
        current_interval = l_interval .* 1
        explore_left=true
    else
        current_best = r * 1.0
        current_width = r_width * 1.
        current_interval = r_interval .* 1
        explore_left=false
    end

    while (r-l) > 0.0001
        c_interval = SA[quantile(d, c), quantile(d, c+region)]
        c_width = c_interval[2] - c_interval[1]

        if c_width < current_width 
            current_interval=c_interval
            current_width = c_width
            current_best=c*1.
        end

        if explore_left
            r=c*1.
            r_width=c_width
            if l_width < r_width
                explore_left=true
            else
                explore_left=false
            end
        else
            l = c * 1.0
            l_width = c_width
            if l_width < r_width
                explore_left = true
            else
                explore_left = false
            end
        end
        c=((r-l)/2.) + l
    end

    return current_best, current_width, current_interval
end

using Optimization, OptimizationNLopt

function univariateUnimodalHDR_optimization(d::UnivariateDistribution, region::Real, test_arguments::Bool=true)
    
    # if test_arguments
    # end
    if region < 0.0 || region > 1.0
        throw(ArgumentError("region must be ∈ [0.0, 1.0]"))
        return nothing
    end
    if region == 0.0
        mode_d = mode(d)
        return SA[mode_d, mode_d]
    end
    if region == 1.0
        support_d = support(d)
        return SA[support_d.lb, support_d.ub]
    end

    p=(d=d, region=region, interval=zeros(d isa ContinuousUnivariateDistribution ? Float64 : Int, 2))

    function fun(l, p)
        p.interval .= quantile(p.d, l[1]), quantile(p.d, min(1.0, l[1]+p.region))
        if any(isinf.(p.interval))
            return 1e100
        end
        return p.interval[2] - p.interval[1]
    end

    fopt = OptimizationFunction(fun)
    prob = OptimizationProblem(fopt, [(1.0-region)/2.], p; lb=[0.0], ub=[1.0-region], xtol_abs=0.00001)
    sol = solve(prob, NLopt.LN_NELDERMEAD())
    return sol.u, sol.objective, p.interval
end

univariateUnimodalHDR(Beta(1.1,2),0.95)
univariateUnimodalHDR_optimization(Beta(1.1,2), 0.95)