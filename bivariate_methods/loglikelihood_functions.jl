"""
If the `profile_type` is LogLikelihood, then corrects a log-likelihood such that an input log-likelihood (which has value of zero at the mle) will now have a value of model.core.maximisedmle at the mle. Otherwise, a copy of ll is returned.
"""
function ll_correction(model::LikelihoodModel, profile_type::AbstractProfileType, ll::Float64)
    if profile_type isa LogLikelihood
        return ll + model.core.maximisedmle
    end
    return ll * 1.0
end

function bivariateΨ!(Ψ::Real, p)
    θs=zeros(p.consistent.num_pars)
    
    function fun(λ)
        θs[p.ind1] = p.Ψ_x[1]
        θs[p.ind2] = Ψ
        return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.λ_opt .= xopt
    return llb
end

function bivariateΨ_vectorsearch!(Ψ, p)
    θs=zeros(p.consistent.num_pars)
    Ψxy = p.pointa + Ψ*p.uhat
    
    function fun(λ)
        θs[p.ind1], θs[p.ind2] = Ψxy
        return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.λ_opt .= xopt
    return llb
end

function bivariateΨ_continuation!(Ψ, p)
    θs=zeros(p.consistent.num_pars)
    Ψxy = p.pointa + Ψ*p.uhat
    
    function fun(λ)
        θs[p.ind1], θs[p.ind2] = Ψxy
        return p.consistent.loglikefunction(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.targetll
    p.λ_opt .= xopt
    return llb
end

"""
Requires optimal values of nuisance parameters at point Ψ to be contained in p.λ_opt
"""
function bivariateΨ_gradient!(Ψ::Vector, p)
    θs=zeros(eltype(Ψ), p.consistent.num_pars)

    θs[p.ind1], θs[p.ind2] = Ψ
    variablemapping2d!(θs, p.λ_opt, p.θranges, p.λranges)
    return p.consistent.loglikefunction(θs, p.consistent.data)
end

function bivariateΨ_ellipse_analytical(Ψ, p)
    return analytic_ellipse_loglike([p.Ψ_x[1], Ψ], [p.ind1, p.ind2], p.consistent.data) - p.consistent.targetll
end

function bivariateΨ_ellipse_analytical_vectorsearch(Ψ, p)
    return analytic_ellipse_loglike(p.pointa + Ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data) - p.consistent.targetll
end

function bivariateΨ_ellipse_analytical_continuation(Ψ, p)
    return analytic_ellipse_loglike(p.pointa + Ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data) - p.targetll
end

function bivariateΨ_ellipse_analytical_gradient(Ψ::Vector, p)
    return analytic_ellipse_loglike(Ψ, [p.ind1, p.ind2], p.consistent.data) - p.consistent.targetll
end