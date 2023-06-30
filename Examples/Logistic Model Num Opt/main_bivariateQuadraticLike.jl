# Section 1: set up packages and parameter values
using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using Roots, NLopt
using ForwardDiff
gr()

Random.seed!(12348)
fileDirectory = joinpath("Workflow paper examples", "Logistic Model Num Opt", "Plots")
include(joinpath("..", "plottingFunctions.jl"))
include(joinpath("..", "..", "JuLikelihood.jl"))

# Workflow functions ##########################################################################

# Section 2: Define ODE model
function DE!(dC, C, p, t)
    λ,K=p
    dC[1]= λ * C[1] * (1.0 - C[1]/K)
end

# Section 3: Solve ODE model
function odesolver(t, λ, K, C0)
    p=(λ,K)
    tspan=(0.0, maximum(t))
    prob=ODEProblem(DE!, [C0], tspan, p)
    sol=solve(prob, saveat=t)
    return sol[1,:]
end

# Section 4: Define function to solve ODE model 
function model(t, a, σ)
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

# Section 6: Define loglikelihood function
function loglhood(data, a, σ)
    y=model(t, a, σ)
    e=0
    dist=Normal(0, σ);
    e=loglikelihood(dist, data-y) 
    return sum(e)
end

# Section 7: Numerical optimisation 
function optimise(fun, θ₀, lb, ub;
    dv = false,
    method = dv ? :LD_LBFGS : :LN_BOBYQA,
    )

    if dv || String(method)[2] == 'D'
        tomax = fun
    else
        tomax = (θ,∂θ) -> fun(θ)
    end

    opt = Opt(method,length(θ₀))
    opt.max_objective = tomax
    opt.lower_bounds = lb       # Lower bound
    opt.upper_bounds = ub       # Upper bound
    opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
    res = optimize(opt, θ₀)
    return res[[2,1]]
end

# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
function funmle(a); return loglhood(data, a, σ) end 

# Data setup #################################################################################
# true parameters
λ=0.01; K=100.0; C0=10.0; t=0:100:1000; σ=10.0;
tt=0:5:1000
a=[λ, K, C0]

# true data
data0 = model(t, a, σ)

# noisy data
data = data0 + σ*randn(length(t))

# Bounds on model parameters #################################################################
λmin, λmax = (0.00, 0.05)
Kmin, Kmax = (50, 150)
C0min, C0max = (0, 50)

θG = [λ, K, C0]
lb = [λmin, Kmin, C0min]
ub = [λmax, Kmax, C0max]

##############################################################################################
# Section 9: Find MLE by numerical optimisation, visually compare data and MLE solution
# Use Nelder-Mead algorithm to estimate maximum likelihood solution for parameters given 
# noisy data
(xopt, fopt) = optimise(funmle, θG, lb, ub)
fmle=fopt
λmle, Kmle, C0mle = xopt .* 1.0
θmle = [λmle, Kmle, C0mle]
ymle(t) = Kmle*C0mle/((Kmle-C0mle)*exp(-λmle*t)+C0mle) # full solution

p1 = plot(ymle, 0, 1000, color=:turquoise1, xlabel="t", ylabel="C(t)",
            legend=false, lw=4, xlims=(0,1100), ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100])

p1 = scatter!(t, data, legend=false, msw=0, ms=7,
            color=:darkorange, msa=:darkorange)
display(p1)
savefig(p1, joinpath(fileDirectory,"mle.pdf"))

# 3D approximation of the likelihood around the MLE solution
H, Γ = getMLE_hessian_and_covariance(funmle, θmle)

function g2(θmle); ForwardDiff.jacobian(funmle,θmle) end
# g2(θmle) = ForwardDiff.jacobian(g1,θmle)
# g3(θmle) = ForwardDiff.jacobian(g2,θmle)
# g4(θmle) = ForwardDiff.jacobian(g3,θmle)

# g2(θmle)
# g3(θmle)
# g4(θmle)
# g2(θmle)
# ForwardDiff.hessian(funmle, θmle)
# ForwardDiff.hessian(g2, θmle)

# Section 10: Depending on MLE we can refine our bounds if required
# λmin=0.0
# λmax=0.05
# Kmin=70
# Kmax=130
# C0min=0
# C0max=40

##############################################################################################
# Section 11: Prediction interval from the full likelihood
# Compute and propogate uncertainty forward from the full 3D likelihood function ######
df = 3 # degrees of freedom
llstar = -quantile(Chisq(df), 0.95)/2 # 95% confidence interval threshold for log likelihood

# Monte Carlo sampling to obtain the 3D likelihood function
N = 100000
λs = rand(Uniform(λmin,λmax),N);
Ks = rand(Uniform(Kmin,Kmax),N);
C0s = rand(Uniform(C0min,C0max),N);

CtraceQuad = reduce(hcat, [model(tt,[λs[i],Ks[i],C0s[i]],σ) for i in 1:N 
                            if (loglhood(data,[λs[i],Ks[i],C0s[i]],σ) - fmle) > llstar])

# evaluate the lower and upper bounds of the confidence intervals
CUF = maximum(CtraceF, dims=2)
CLF = minimum(CtraceF, dims=2)

# plot the family of curves, lower and upper confidence bounds and maximum likelihood solution given data
qq1 = plotPrediction(tt, CtraceF, (CUF, CLF), confColor=:gold, xlabel="t", 
            ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], yticks=[0,50,100], legend=false)


# Quadratic likelihood approximation
CtraceQuad = reduce(hcat, [model(tt,[λs[i],Ks[i],C0s[i]],σ) for i in 1:N if ellipse_loglike([λs[i],Ks[i],C0s[i]], θmle, H) > llstar])

CUQuad = maximum(CtraceQuad, dims=2)
CLQuad = minimum(CtraceQuad, dims=2)

qq1 = plotPredictionComparison(tt, CtraceF, (CUF, CLF), (CUQuad, CLQuad), ymle.(tt),
                                xlabel="t", ylabel="C(t)", ylims=(0,120),
                                xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(qq1)

savefig(qq1, joinpath(fileDirectory, "Comparison_boundary_fullVsQuadraticLike.pdf"))

##############################################################################################
# Section 16: Construct bivariate profiles and associated pair-wise predictions starting with the bivariate quadratic approximation of log-likelihood profile likelihood for (λ,K)  
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter λ and K
df=2
llstar=-quantile(Chisq(df),0.95)/2

# Define function to compute the bivariate profile
function bivariateλK(λ,K)
    function funλK(a::Vector{<:Float64}); return ellipse_loglike([λ,K,a[1]], θmle, H) end
    
    θG = [C0]
    lb=[C0min]
    ub=[C0max]
    (xopt,fopt)=optimise(funλK,θG,lb,ub)
    return fopt, xopt[1]
end 

#############################################################################################
# bivariate profile likelihood for λ and K ##################################################
# Bisection method to locate points in lambda, K space that are on the 95% confidence interval threshold for log likelihood
f(x,y) = bivariateλK(x,y)
g(x,y)=f(x,y)[1]-llstar

N=100
λsamples_boundary=zeros(2*N)
Ksamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)

# Fix λ and estimate K that puts it on the log likelihood boundary, optimising out C0 using the MLE estimate for C0
# Define small parameter on the scale of parameter K
ϵ=(Kmax-Kmin)/10^8
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(λmin,λmax))
    y0=rand(Uniform(Kmin,Kmax))
    y1=rand(Uniform(Kmin,Kmax))

    # If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the 
    # bisection algorithm to find the location of the threshold on the vertical line separating 
    # the two points
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        # println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)

        λsamples_boundary[count]=x;
        Ksamples_boundary[count]=y1;
        C0samples_boundary[count]=f(x,y1)[2]
    end
end 

# Fix K and estimate λ that puts it on the log likelihood boundary, optimising out C0 using the MLE estimate for C0
ϵ=(λmax-λmin)/10^6
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(Kmin,Kmax))
    x0=rand(Uniform(λmin,λmax))
    x1=rand(Uniform(λmin,λmax))
    
    #If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
    #horizontal line separating the two points   
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        # println(count)

        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        λsamples_boundary[N+count]=x1;
        Ksamples_boundary[N+count]=y;
        C0samples_boundary[N+count]=f(x1,y)[2]
    end
end 

λsamples_boundary1 = λsamples_boundary .* 1.0
Ksamples_boundary1 = Ksamples_boundary .* 1.0

# Plot the MLE and the 2N points identified on the boundary
a1 = plot2DBoundary((λsamples_boundary, Ksamples_boundary), (λmle, Kmle), N, 
                    xticks=[0,0.015,0.03], yticks=[80, 100, 120],
                    xlims=(0,0.03), ylims=(80,120), xlabel="λ", ylabel="K", legend=false)

display(a1)

# Analytical version of ellipse #############################################################
# bivariate profile likelihood for λ and K ##################################################
# Bisection method to locate points in lambda, K space that are on the 95% confidence interval threshold for log likelihood

g(x,y) = analytic_ellipse_loglike([x, y], [1,2], θmle, Γ)-llstar

N=100
λsamples_boundary=zeros(2*N)
Ksamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)

# Fix λ and estimate K that puts it on the log likelihood boundary, optimising out C0 using the MLE estimate for C0
# Define small parameter on the scale of parameter K
ϵ=(Kmax-Kmin)/10^8
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(λmin,λmax))
    y0=rand(Uniform(Kmin,Kmax))
    y1=rand(Uniform(Kmin,Kmax))

    # If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the 
    # bisection algorithm to find the location of the threshold on the vertical line separating 
    # the two points
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        # println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)

        λsamples_boundary[count]=x;
        Ksamples_boundary[count]=y1;
        C0samples_boundary[count]=f(x,y1)[2]
    end
end 

# Fix K and estimate λ that puts it on the log likelihood boundary, optimising out C0 using the MLE estimate for C0
ϵ=(λmax-λmin)/10^6
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(Kmin,Kmax))
    x0=rand(Uniform(λmin,λmax))
    x1=rand(Uniform(λmin,λmax))
    
    #If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
    #horizontal line separating the two points   
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        # println(count)

        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        λsamples_boundary[N+count]=x1;
        Ksamples_boundary[N+count]=y;
        C0samples_boundary[N+count]=f(x1,y)[2]
    end
end 

λsamples_boundary11 = λsamples_boundary .* 1.0
Ksamples_boundary11 = Ksamples_boundary .* 1.0

# Plot the MLE and the 2N points identified on the boundary
a1 = plot2DBoundary((λsamples_boundary, Ksamples_boundary), (λmle, Kmle), N, 
                    xticks=[0,0.015,0.03], yticks=[80, 100, 120],
                    xlims=(0,0.03), ylims=(80,120), xlabel="λ", ylabel="K", legend=false)

a1 = plot2DBoundaryComparison((λsamples_boundary1, Ksamples_boundary1), (λsamples_boundary11, Ksamples_boundary11), (λmle, Kmle), N, 
                        xticks=[0,0.015,0.03], yticks=[80, 100, 120],
                        xlims=(0,0.03), ylims=(80,120), xlabel="λ", ylabel="K", legend=false)

display(a1)

##############################################################################################
# Section 16: Construct bivariate profiles and associated pair-wise predictions starting with the bivariate profile likelihood for (λ,K)  
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter λ and K
df=2
llstar=-quantile(Chisq(df),0.95)/2

# Define function to compute the bivariate profile
function bivariateλK(λ,K)
    function funλK(a); return loglhood(data,[λ,K,a[1]],σ) end
    
    θG = [C0]
    lb=[C0min]
    ub=[C0max]
    (xopt,fopt)=optimise(funλK,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt[1]
end 

#############################################################################################
# bivariate profile likelihood for λ and K ##################################################
# Bisection method to locate points in lambda, K space that are on the 95% confidence interval threshold for log likelihood
f(x,y) = bivariateλK(x,y)
g(x,y)=f(x,y)[1]-llstar

N=100
λsamples_boundary=zeros(2*N)
Ksamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)

# Fix λ and estimate K that puts it on the log likelihood boundary, optimising out C0 using the MLE estimate for C0
# Define small parameter on the scale of parameter K
ϵ=(Kmax-Kmin)/10^8
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(λmin,λmax))
    y0=rand(Uniform(Kmin,Kmax))
    y1=rand(Uniform(Kmin,Kmax))

    # If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the 
    # bisection algorithm to find the location of the threshold on the vertical line separating 
    # the two points
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)

        λsamples_boundary[count]=x;
        Ksamples_boundary[count]=y1;
        C0samples_boundary[count]=f(x,y1)[2]
    end
end 

# Fix K and estimate λ that puts it on the log likelihood boundary, optimising out C0 using the MLE estimate for C0
ϵ=(λmax-λmin)/10^6
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(Kmin,Kmax))
    x0=rand(Uniform(λmin,λmax))
    x1=rand(Uniform(λmin,λmax))
    
    #If the points (x0,y) and (x1,y) are either side of the appropriate threshold, use the bisection algorithm to find the location of the threshold on the 
    #horizontal line separating the two points   
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)

        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        λsamples_boundary[N+count]=x1;
        Ksamples_boundary[N+count]=y;
        C0samples_boundary[N+count]=f(x1,y)[2]
    end
end 

# Plot the MLE and the 2N points identified on the boundary
a1 = plot2DBoundary((λsamples_boundary, Ksamples_boundary), (λmle, Kmle), N, 
                    xticks=[0,0.015,0.03], yticks=[80, 100, 120],
                    xlims=(0,0.03), ylims=(80,120), xlabel="λ", ylabel="K", legend=false)

a1 = plot2DBoundaryComparison((λsamples_boundary, Ksamples_boundary), (λsamples_boundary1, Ksamples_boundary1), (λmle, Kmle), N, 
                    xticks=[0,0.015,0.03], yticks=[80, 100, 120],
                    xlims=(0,0.03), ylims=(80,120), xlabel="λ", ylabel="K", legend=false)

display(a1)


#############################################################################################
# Compute model for parameter values on the boundary
Ctrace1_boundary = zeros(length(tt), 2*N)
for i in 1:2*N
    Ctrace1_boundary[:,i]=model(tt, [λsamples_boundary[i], Ksamples_boundary[i],
                                C0samples_boundary[i]], σ)
end
    
CU1_boundary = maximum(Ctrace1_boundary, dims=2)
CL1_boundary = minimum(Ctrace1_boundary, dims=2)

pp1 = plotPrediction(tt, Ctrace1_boundary, (CU1_boundary, CL1_boundary), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp3 = plot(a1, pp1, layout=(1,2))

display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateLK_boundaryComparison.pdf"))

#############################################################################################
# Section 17: Repeat Section 16 for the (λ,C(0)) bivariate   
# Compute and propogate uncertainty forward from the bivariate quadratic approximation of log-likelihood profile for parameter λ and C0
function bivariateλC0(λ,C0)
    function funλC0(a::Vector{<:Float64}); return ellipse_loglike([λ,a[1],C0], θmle, H) end
    
    θG = [K]
    lb=[Kmin]
    ub=[Kmax]
    (xopt,fopt)=optimise(funλC0,θG,lb,ub)
    return fopt, xopt[1]
end 

# bivariate profile likelihood for λ and C0 #################################################
# Bisection method to locate points in lambda, C0 space that are on the 95% confidence interval threshold for log likelihood
f(x,y) = bivariateλC0(x,y)
g(x,y) = f(x,y)[1]-llstar

λsamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
Ksamples_boundary=zeros(2*N)

# Fix λ and estimate C0 that puts it on the log likelihood boundary, optimising out K using the MLE estimate for K
ϵ=(C0max-C0min)/10^8
h(y,p)=g(p,y)
count=0
while count < N 
    x=rand(Uniform(λmin,λmax))
    y0=rand(Uniform(C0min,C0max))
    y1=rand(Uniform(C0min,C0max))
            
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        # println(count)
        
        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
        
        λsamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        Ksamples_boundary[count]=f(x,y1)[2]
    end
end 
        
# Fix C0 and estimate λ that puts it on the log likelihood boundary, optimising out K using the MLE estimate for K
ϵ=(λmax-λmin)/10^6
h(x,p)=g(x,p)
count=0
while count < N 
    y=rand(Uniform(C0min,C0max))
    x0=rand(Uniform(λmin,λmax))
    x1=rand(Uniform(λmin,λmax))
                
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        # println(count)
                
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)             
                
        λsamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        Ksamples_boundary[N+count]=f(x1,y)[2]
    end
end 

λsamples_boundary1 = λsamples_boundary .* 1.0
C0samples_boundary1 = C0samples_boundary .* 1.0

a2 = plot2DBoundary((λsamples_boundary, C0samples_boundary), (λmle, C0mle), N, 
                    xlims=(0.0,0.03), ylims=(0,35), xticks=[0,0.015,0.03], yticks=[0,15,30], xlabel="λ", ylabel="C(0)", legend=false)

# Analytical version ########################################################################
# bivariate profile likelihood for λ and C0 #################################################
# Bisection method to locate points in lambda, C0 space that are on the 95% confidence interval threshold for log likelihood
g(x,y) = analytic_ellipse_loglike([x, y], [1,3], θmle, Γ)-llstar

λsamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
Ksamples_boundary=zeros(2*N)

# Fix λ and estimate C0 that puts it on the log likelihood boundary, optimising out K using the MLE estimate for K
ϵ=(C0max-C0min)/10^8
h(y,p)=g(p,y)
count=0
while count < N 
    x=rand(Uniform(λmin,λmax))
    y0=rand(Uniform(C0min,C0max))
    y1=rand(Uniform(C0min,C0max))
            
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        # println(count)
        
        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
        
        λsamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        Ksamples_boundary[count]=f(x,y1)[2]
    end
end 
        
# Fix C0 and estimate λ that puts it on the log likelihood boundary, optimising out K using the MLE estimate for K
ϵ=(λmax-λmin)/10^6
h(x,p)=g(x,p)
count=0
while count < N 
    y=rand(Uniform(C0min,C0max))
    x0=rand(Uniform(λmin,λmax))
    x1=rand(Uniform(λmin,λmax))
                
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        # println(count)
                
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)             
                
        λsamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        Ksamples_boundary[N+count]=f(x1,y)[2]
    end
end 

λsamples_boundary11 = λsamples_boundary .* 1.0
C0samples_boundary11 = C0samples_boundary .* 1.0

a2 = plot2DBoundaryComparison((λsamples_boundary1, C0samples_boundary1), 
                            (λsamples_boundary11, C0samples_boundary11), (λmle, C0mle), N, 
                            xlims=(0.0,0.03), ylims=(0,35), xticks=[0,0.015,0.03], yticks=[0,15,30],
                            xlabel="λ", ylabel="C(0)", legend=false)
display(a2)

#############################################################################################
# Section 17: Repeat Section 16 for the (λ,C(0)) bivariate   
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter λ and C0
function bivariateλC0(λ,C0)
    function funλC0(a); return loglhood(data,[λ,a[1],C0],σ) end
    
    θG = [K]
    lb=[Kmin]
    ub=[Kmax]
    (xopt,fopt)=optimise(funλC0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt[1]
end 

# bivariate profile likelihood for λ and C0 #################################################
# Bisection method to locate points in lambda, C0 space that are on the 95% confidence interval threshold for log likelihood
f(x,y) = bivariateλC0(x,y)
g(x,y) = f(x,y)[1]-llstar

λsamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
Ksamples_boundary=zeros(2*N)

# Fix λ and estimate C0 that puts it on the log likelihood boundary, optimising out K using the MLE estimate for K
ϵ=(C0max-C0min)/10^8
h(y,p)=g(p,y)
count=0
while count < N 
    x=rand(Uniform(λmin,λmax))
    y0=rand(Uniform(C0min,C0max))
    y1=rand(Uniform(C0min,C0max))
            
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)
        
        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
        
        λsamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        Ksamples_boundary[count]=f(x,y1)[2]
    end
end 
        
# Fix C0 and estimate λ that puts it on the log likelihood boundary, optimising out K using the MLE estimate for K
ϵ=(λmax-λmin)/10^6
h(x,p)=g(x,p)
count=0
while count < N 
    y=rand(Uniform(C0min,C0max))
    x0=rand(Uniform(λmin,λmax))
    x1=rand(Uniform(λmin,λmax))
                
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)
                
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)             
                
        λsamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        Ksamples_boundary[N+count]=f(x1,y)[2]
    end
end 

a2 = plot2DBoundary((λsamples_boundary, C0samples_boundary), (λmle, C0mle), N, 
                    xlims=(0.0,0.03), ylims=(0,35), xticks=[0,0.015,0.03], yticks=[0,15,30], xlabel="λ", ylabel="C(0)", legend=false)

a2 = plot2DBoundaryComparison((λsamples_boundary, C0samples_boundary), 
                            (λsamples_boundary1, C0samples_boundary1), (λmle, C0mle), N, 
                            xlims=(0.0,0.03), ylims=(0,35), xticks=[0,0.015,0.03], yticks=[0,15,30],
                            xlabel="λ", ylabel="C(0)", legend=false)
display(a2)

#############################################################################################
# Compute model for parameter values on the boundary
Ctrace2_boundary = zeros(length(tt),2*N)
for i in 1:2*N
    Ctrace2_boundary[:,i]=model(tt, [λsamples_boundary[i], Ksamples_boundary[i],
                                 C0samples_boundary[i]], σ);
end
            
CU2_boundary = maximum(Ctrace2_boundary, dims=2)
CL2_boundary = minimum(Ctrace2_boundary, dims=2)

pp1 = plotPrediction(tt, Ctrace2_boundary, (CU2_boundary, CL2_boundary), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp3=plot(a2, pp1, layout=(1,2))
display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateLC0_boundaryComparison.pdf")) 

#############################################################################################
# Compute and propogate uncertainty forward from the bivariate quadratic approximation of log-likelihood profile likelihood for parameter K and C0
function bivariateKC0(K,C0)
    function funKC0(a::Vector{<:Float64}); return ellipse_loglike([a[1],K,C0], θmle, H) end

    θG = [λ]
    lb=[λmin]
    ub=[λmax]
    (xopt,fopt)  = optimise(funKC0,θG,lb,ub)
    return fopt, xopt[1]
end 

# bivariate profile likelihood for K and C0 #################################################
# Bisection method to locate points in K, C0 space that are on the 95% confidence interval threshold for log likelihood
f(x,y) = bivariateKC0(x,y)
g(x,y)=f(x,y)[1]-llstar

Ksamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
λsamples_boundary=zeros(2*N)

# Fix K and estimate C0 that puts it on the log likelihood boundary, optimising out λ using the MLE estimate for λ
ϵ=(C0max-C0min)/10^8
h(y,p)=g(p,y)   
count=0
while count < N
    x=rand(Uniform(Kmin,Kmax))
    y0=rand(Uniform(C0min,C0max))
    y1=rand(Uniform(C0min,C0max))
            
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        # println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
            
        Ksamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        λsamples_boundary[count]=f(x,y1)[2]
    end
end 

# Fix C0 and estimate K that puts it on the log likelihood boundary, optimising out λ using the MLE estimate for λ
ϵ=1(Kmax-Kmin)/10^8
h(x,p)=g(x,p)
count=0
while count < N 
    y=rand(Uniform(C0min,C0max))
    x0=rand(Uniform(Kmin,Kmax))
    x1=rand(Uniform(Kmin,Kmax))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        # println(count)

        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        Ksamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        λsamples_boundary[N+count]=f(x1,y)[2]
    end
end 

Ksamples_boundary1 = Ksamples_boundary .* 1.0
C0samples_boundary1 = C0samples_boundary .* 1.0
    
a3 = plot2DBoundary((Ksamples_boundary, C0samples_boundary), (Kmle, C0mle), N, 
                    xlims=(80, 120), ylims=(C0min,35), xticks=[80,100,120], yticks=[0,15,30], xlabel="K", ylabel="C(0)", legend=false)
      
display(a3)

#############################################################################################
# Compute and propogate uncertainty forward from the bivariate quadratic approximation of log-likelihood profile likelihood for parameter K and C0
function bivariateKC0(K,C0)
    function funKC0(a::Vector{<:Float64}); return ellipse_loglike([a[1],K,C0], θmle, H) end

    θG = [λ]
    lb=[λmin]
    ub=[λmax]
    (xopt,fopt)  = optimise(funKC0,θG,lb,ub)
    return fopt, xopt[1]
end 

# bivariate profile likelihood for K and C0 #################################################
# Bisection method to locate points in K, C0 space that are on the 95% confidence interval threshold for log likelihood
f(x,y) = bivariateKC0(x,y)
g(x,y)=f(x,y)[1]-llstar

Ksamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
λsamples_boundary=zeros(2*N)

# Fix K and estimate C0 that puts it on the log likelihood boundary, optimising out λ using the MLE estimate for λ
ϵ=(C0max-C0min)/10^8
h(y,p)=g(p,y)   
count=0
while count < N
    x=rand(Uniform(Kmin,Kmax))
    y0=rand(Uniform(C0min,C0max))
    y1=rand(Uniform(C0min,C0max))
            
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        # println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
            
        Ksamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        λsamples_boundary[count]=f(x,y1)[2]
    end
end 

# Fix C0 and estimate K that puts it on the log likelihood boundary, optimising out λ using the MLE estimate for λ
ϵ=1(Kmax-Kmin)/10^8
h(x,p)=g(x,p)
count=0
while count < N 
    y=rand(Uniform(C0min,C0max))
    x0=rand(Uniform(Kmin,Kmax))
    x1=rand(Uniform(Kmin,Kmax))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        # println(count)

        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        Ksamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        λsamples_boundary[N+count]=f(x1,y)[2]
    end
end 

Ksamples_boundary1 = Ksamples_boundary .* 1.0
C0samples_boundary1 = C0samples_boundary .* 1.0
    
a3 = plot2DBoundary((Ksamples_boundary, C0samples_boundary), (Kmle, C0mle), N, 
                    xlims=(80, 120), ylims=(C0min,35), xticks=[80,100,120], yticks=[0,15,30], xlabel="K", ylabel="C(0)", legend=false)
      
display(a3)

# Analytical version ########################################################################
# bivariate profile likelihood for K and C0 #################################################
# Bisection method to locate points in K, C0 space that are on the 95% confidence interval threshold for log likelihood
g(x,y) = analytic_ellipse_loglike([x, y], [2,3], θmle, Γ)-llstar

Ksamples_boundary=zeros(2*N)
C0samples_boundary=zeros(2*N)
λsamples_boundary=zeros(2*N)

# Fix K and estimate C0 that puts it on the log likelihood boundary, optimising out λ using the MLE estimate for λ
ϵ=(C0max-C0min)/10^8
h(y,p)=g(p,y)   
count=0
while count < N
    x=rand(Uniform(Kmin,Kmax))
    y0=rand(Uniform(C0min,C0max))
    y1=rand(Uniform(C0min,C0max))
            
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        # println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
            
        Ksamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        λsamples_boundary[count]=f(x,y1)[2]
    end
end 

# Fix C0 and estimate K that puts it on the log likelihood boundary, optimising out λ using the MLE estimate for λ
ϵ=1(Kmax-Kmin)/10^8
h(x,p)=g(x,p)
count=0
while count < N 
    y=rand(Uniform(C0min,C0max))
    x0=rand(Uniform(Kmin,Kmax))
    x1=rand(Uniform(Kmin,Kmax))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        # println(count)

        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        Ksamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        λsamples_boundary[N+count]=f(x1,y)[2]
    end
end 

Ksamples_boundary11 = Ksamples_boundary .* 1.0
C0samples_boundary11 = C0samples_boundary .* 1.0
    
a3 = plot2DBoundary((Ksamples_boundary, C0samples_boundary), (Kmle, C0mle), N, 
                    xlims=(80, 120), ylims=(C0min,35), xticks=[80,100,120], yticks=[0,15,30], xlabel="K", ylabel="C(0)", legend=false)


a3 = plot2DBoundaryComparison((Ksamples_boundary1, C0samples_boundary1), 
                            (Ksamples_boundary11, C0samples_boundary11), (Kmle, C0mle), N, 
                            xlims=(80, 120), ylims=(C0min,35), xticks=[80,100,120], 
                            yticks=[0,15,30], xlabel="K", ylabel="C(0)", legend=false)
      
display(a3)
    
#############################################################################################
# Compute model for parameter values on the boundary
Ctrace3_boundary = zeros(length(tt),2*N)

for i in 1:2*N
    Ctrace3_boundary[:,i]=model(tt, [λsamples_boundary[i], Ksamples_boundary[i],
                                C0samples_boundary[i]], σ);
end

CU3_boundary = maximum(Ctrace3_boundary, dims=2)
CL3_boundary = minimum(Ctrace3_boundary, dims=2)

pp1 = plotPrediction(tt, Ctrace3_boundary, (CU3_boundary, CL3_boundary), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp3=plot(a3, pp1, layout=(1,2))
display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateKC0_boundaryComparison.pdf")) 


###########################################################################################
# Construct approximate prediction intervals from union of bivariate profile intervals 
# Compare to intervals obtained from the full likelihood

# Compute the union of the three pair-wise profile predictions using the identified boundary
CU_boundary = max.(CU1_boundary, CU2_boundary, CU3_boundary)
CL_boundary = min.(CL1_boundary, CL2_boundary, CL3_boundary)

# Plot the family of predictions made using the boundary tracing method, the MLE and the prediction intervals defined by the full log-liklihood and the union of the three bivariate profile likelihoods 
qq1 = plotPredictionComparison(tt, CtraceF, (CUF, CLF), (CU_boundary, CL_boundary), ymle.(tt),
                                xlabel="t", ylabel="C(t)", ylims=(0,120),
                                xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(qq1)
savefig(qq1, joinpath(fileDirectory, "Bivariatecomparison_boundary.pdf"))
