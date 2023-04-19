module SloppihoodTools
using Distributions
using LinearAlgebra
using ForwardDiff
using NLopt

# - Original param likelihood methods
# - Note: everything is pretty dimension agnostic...(except 2D box helper)
# - ...xy doesn't need to be 2D!

# -- log likelihood as function of parameter given (iid) data
function construct_lnlike_xy(distrib_xy,data;dist_type=:uni)
    if dist_type === :uni
        return xy -> sum(logpdf.(distrib_xy(xy),data))
    else
        return xy -> sum(logpdf(distrib_xy(xy),data))
    end
end

# - Coordinate change likelihood methods
function construct_lnlike_XY(lnlike_xy,XYtoxy)
    return XY -> lnlike_xy(XYtoxy(XY))
end

# - Generic (coord free) likelihood methods 
# -- if finite differencing needed (coord free)
function finite_diff_gradient(f, θ; h=1e-8)
    n = length(θ)
    numerical_gradient = [(f(θ + h*ei) - f(θ - h*ei)) / (2*h) for ei in eachcol(1I(n))]
    return numerical_gradient
end

# -- function maximise with NLOpt
function construct_lnlike_to_max(lnlike)
    # - use as optimisation target
    function lnlike_to_max(θ, grad; grad_type=:auto)
        if grad_type === :auto
            grad = ForwardDiff.gradient(lnlike,θ)
        else
            grad = finite_diff_gradient(lnlike,θ)
        end
            return lnlike(θ)
    end
    return lnlike_to_max
end

# -- profile a target component (combine with coordinate change for general target param)
function profile_target(lnlike,target_indices,lbs_all,ubs_all,nuisance_guess;grid_steps=[1000],use_last_as_guess=true,method=:LN_BOBYQA,xtol_rel=1e-9,ftol_rel=1e-9)
    # get all the relevant extra dimensions and indices
    dim_all = length(lbs_all)
    dim_target = length(target_indices)
    nuisance_indices = setdiff(1:dim_all,target_indices) #assume all have lower bounds -- need to set -inf etc by default.
    dim_nuisance = length(nuisance_indices)
    # check for point estimation problem (no interest parameter)
    if dim_target == 0
        println("solving MLE problem")
        opt = Opt(method,dim_all)
        opt.lower_bounds = lbs_all 
        opt.upper_bounds = ubs_all 
        opt.xtol_rel = xtol_rel
        opt.ftol_rel = ftol_rel
        opt.max_objective = construct_lnlike_to_max(lnlike)
        (lnlike_opt,ωi_opt,ret) = optimize(opt,nuisance_guess)
        return ωi_opt, lnlike_opt .- maximum(lnlike_opt)
    end
    # set up grids based on input info
    lbs_target = lbs_all[target_indices]
    ubs_target = ubs_all[target_indices]
    target_grids = Vector{Vector{Float64}}(undef,dim_target)
    for i in eachindex(lbs_target)
        if length(grid_steps) == 1
            target_grids[i] = LinRange(lbs_target[i], ubs_target[i], grid_steps[1])
        else
            target_grids[i] = LinRange(lbs_target[i], ubs_target[i], grid_steps[i])
        end
    end
    # check for non-trivial (non-pure gridding problem) and set approp. options.
    if length(nuisance_indices) > 0
        opt = Opt(method,dim_nuisance)
        opt.lower_bounds = lbs_all[nuisance_indices] 
        opt.upper_bounds = ubs_all[nuisance_indices] 
        opt.xtol_rel = xtol_rel
        opt.ftol_rel = ftol_rel
    end 
    # get Cartesian product of interest parameter grid then iterate over
    interest_combinations = Base.product(target_grids...)
    ω₀ = nuisance_guess
    ψω_indices_in_θ = [target_indices...,setdiff(1:dim_all,target_indices)...]
    θ_values = Vector{Vector{Float64}}(undef,length(interest_combinations))
    lnlike_values = Vector{Float64}(undef,length(interest_combinations))
    for (i,ψi) in enumerate(interest_combinations)
        ψω_to_θ = ψω -> ψω[ψω_indices_in_θ]
        # update optimisation target if non-trivial nuisance
        if length(nuisance_indices) > 0
            # construct likelihood in nuisance parameter for given psi_i then put in to_max form
            opt.max_objective = construct_lnlike_to_max(ω -> lnlike(ψω_to_θ([ψi...,ω...])))
            (lnlike_opt,ωi_opt,ret) = optimize(opt,ω₀)
        else #pure gridding/trivial nuisance
            lnlike_opt = lnlike(ψω_to_θ([ψi...])) # need to ensure vector not tuple
            ωi_opt = []
        end
        θ_values[i] = ψω_to_θ([ψi...,ωi_opt...])
        lnlike_values[i] = lnlike_opt
        if use_last_as_guess
            ω₀ = ωi_opt
        end
    end
    return θ_values, lnlike_values .- maximum(lnlike_values)
end

# -- ellipse approx
function construct_ellipse_lnlike_approx(lnlike, θ_est; h_type=:auto,return_h=true)
    if h_type === :auto
        H = -ForwardDiff.hessian(lnlike, θ_est)
    else
        println("warning no hessian")
        # todo finite diff 
    end
    if return_h
        return θ -> -0.5*(θ-θ_est)'*H*(θ-θ_est), H
    else
        return θ -> -0.5*(θ-θ_est)'*H*(θ-θ_est)
    end
end

# -- 2D constraint box inside cartesian product
function construct_2D_internal_constraint_box(lbs,ubs,lb_funcs,ub_funcs;grid_steps=[100],safety_factors=[0.0,0.0])
    # set up grids based on input info
    grids = Vector{Vector{Float64}}(undef,2) 
    for i in 1:2
        if length(grid_steps) == 1
            grids[i] = LinRange(lbs[i], ubs[i], grid_steps[1])
        else
            grids[i] = LinRange(lbs[i], ubs[i], grid_steps[i])
        end
    end
    # lower and upper of first across second
    lb1 = max(maximum(lb_funcs[1],grids[2]),lbs[1]) + safety_factors[1]
    ub1 = min(minimum(ub_funcs[1],grids[2]),ubs[1]) - safety_factors[1]
    # lower and upper of second across first
    lb2 = max(maximum(lb_funcs[2],grids[1]),lbs[2]) + safety_factors[2]
    ub2 = min(minimum(ub_funcs[2],grids[1]),ubs[2]) - safety_factors[2]
    new_lbs = [lb1,lb2]
    new_ubs = [ub1,ub2]
    return new_lbs, new_ubs
end

end;