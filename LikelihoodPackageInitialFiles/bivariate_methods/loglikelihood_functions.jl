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
    return analytic_ellipse_loglike([p.Ψ_x[1], Ψ], [p.ind1, p.ind2], p.consistent.data_analytic) - p.consistent.targetll
end

function bivariateΨ_ellipse_analytical_vectorsearch(Ψ, p)
    return analytic_ellipse_loglike(p.pointa + Ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data_analytic) - p.consistent.targetll
end

function bivariateΨ_ellipse_analytical_continuation(Ψ, p)
    return analytic_ellipse_loglike(p.pointa + Ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data_analytic) - p.targetll
end

function bivariateΨ_ellipse_analytical_gradient(Ψ::Vector, p)
    return analytic_ellipse_loglike(Ψ, [p.ind1, p.ind2], p.consistent.data_analytic) - p.consistent.targetll
end

function bivariateΨ_ellipse_unbounded(Ψ::Vector, p)
    θs=zeros(p.consistent.num_pars)
    θs[p.ind1], θs[p.ind2] = Ψ

    function fun(λ)
        return ellipse_loglike(variablemapping2d!(θs, λ, p.θranges, p.λranges), p.consistent.data) 
    end

    (xopt,_)=optimise_unbounded(fun, p.initGuess)
    # (xopt,fopt)=optimise_unbounded(fun, p.initGuess)
    # llb=fopt-p.consistent.targetll
    # p.λ_opt .= xopt
    # return llb
    return xopt
end