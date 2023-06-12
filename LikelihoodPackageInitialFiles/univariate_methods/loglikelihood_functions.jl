# function univariateΨ_ellipse_analytical(Ψ, p)
#     return analytic_ellipse_loglike([Ψ], [p.ind], p.consistent.data) - p.consistent.targetll
# end

function univariateΨ_ellipse_unbounded(Ψ, p)
    θs=zeros(p.consistent.num_pars)

    function fun(λ)
        θs[p.ind] = Ψ
        return ellipse_loglike(variablemapping1d!(θs, λ, p.θranges, p.λranges), p.consistent.data) 
    end

    (xopt,fopt)=optimise_unbounded(fun, p.initGuess)
    llb=fopt-p.consistent.targetll
    p.λ_opt .= xopt
    return llb
end

function univariateΨ(Ψ, p)
    θs=zeros(p.consistent.num_pars)

    function fun(λ)
        θs[p.ind] = Ψ
        return p.consistent.loglikefunction(variablemapping1d!(θs, λ, p.θranges, p.λranges), p.consistent.data) 
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.λ_opt .= xopt
    return llb
end