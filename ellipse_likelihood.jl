# using Memoize

# memoizing doesn't appear to have a performance advantage in this situation for small number of parameters (is a performance hit in fact) - may be advantageous for a larger number of parameters
# @memoize function invCovariance(Γ::Matrix{T}, θIndexes::Vector{Int}) where T<:Float64
#     return inv(Γ[θIndexes, θIndexes]) 
# end
# function analytic_ellipse_loglike(θ::Vector{T}, θIndexes::Vector{Int}, 
#                                     θmle::Vector{T}, Γ::Matrix{T}) where T<:Float64
#     return -0.5 * (θ-θmle[θIndexes])' * invCovariance(Γ, θIndexes) * (θ-θmle[θIndexes])
# end

# Analytic ellipse log-likelihood has no knowledge of lower and upper bounds on parameters. 
# Hence profiles generated by optimising out the nuisance parameters for a given
# interest parameter may look different if the analytical profile enters space
# where a bound would be active.
# Pushing forward from these confidence bounds may be infeasible - if analytical profile has entered a space
# where a parameter bound is active.
function analytic_ellipse_loglike(θ::Vector{T}, θIndexes::Vector{Int}, 
    mleTuple::@NamedTuple{θmle::Vector{T}, Γmle::Matrix{T}}) where T<:Float64
    return -0.5 * (θ-mleTuple.θmle[θIndexes])' * inv(mleTuple.Γmle[θIndexes, θIndexes]) * (θ-mleTuple.θmle[θIndexes])
end

# function ellipse_loglike(θ::Vector{T}, θmle::Vector{T}, H::Matrix{T})::Float64 where T<:Float64
function ellipse_loglike(θ::Vector{T}, mleTuple::@NamedTuple{θmle::Vector{T}, Hmle::Matrix{T}})::Float64 where T<:Float64
    return -0.5 * ((θ - mleTuple.θmle)' * mleTuple.Hmle * (θ - mleTuple.θmle))
end

# function ellipse_like(θ::Vector{T}, θmle::Vector{T}, H::Matrix{T}) where T<:Real
function ellipse_like(θ::Vector{T}, mleTuple::@NamedTuple{θmle::Vector{T}, Hmle::Matrix{T}}) where T<:Real
    return exp(ellipse_loglike(θ, mleTuple))
end

function getMLE_hessian_and_covariance(f::Function, θmle::Vector{<:Float64})

    Hmle = -ForwardDiff.hessian(f, θmle)

    # if inverse fails then may have locally non-identifiable parameter OR parameter is
    # a delta distribution given data.
    Γmle = inv(Hmle)

    return Hmle, Γmle
end


function getMLE_ellipse_approximation!(model::LikelihoodModel)

    function funmle(θ); return model.core.loglikefunction(θ, model.core.data) end

    Hmle, Γmle = getMLE_hessian_and_covariance(funmle, model.core.θmle)

    model.ellipse_MLE_approx = EllipseMLEApprox(Hmle, Γmle)

    return model.ellipse_MLE_approx.Hmle, model.ellipse_MLE_approx.Γmle
end

function check_ellipse_approx_exists!(model::LikelihoodModel)
    if ismissing(model.ellipse_MLE_approx)
        getMLE_ellipse_approximation!(model)
    end
    return nothing
end