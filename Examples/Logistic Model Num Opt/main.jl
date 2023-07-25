# Section 1: set up packages and parameter values
using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using Roots, NLopt
using ForwardDiff
gr()

Random.seed!(12348)
fileDirectory = joinpath("Examples", "Logistic Model Num Opt", "Plots")
include(joinpath("..", "plottingFunctions.jl"))

# Workflow functions ##########################################################################

# Section 2: Define ODE model
# function DE!(dC, C, p, t)
#     λ,K=p
#     dC[1]= λ * C[1] * (1.0 - C[1]/K)
# end

# Section 3: Solve ODE model
# function odesolver(t, λ, K, C0)
#     p=(λ,K)
#     tspan=(0.0, maximum(t))
#     prob=ODEProblem(DE!, [C0], tspan, p)
#     sol=solve(prob, saveat=t)
#     return sol[1,:]
# end

# Section 4: Define function to solve ODE model 
# function model(t, a, σ)
#     y=odesolver(t, a[1],a[2],a[3])
#     return y
# end

function model(t, a, σ)
    return (a[2]*a[3]) ./ ((a[2]-a[3]) .* (exp.(-a[1] .* t)) .+ a[3])
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
    opt.maxeval = 6000
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
λmle, Kmle, C0mle = xopt
ymle(t) = Kmle*C0mle/((Kmle-C0mle)*exp(-λmle*t)+C0mle) # full solution

p1 = plot(ymle, 0, 1000, color=:turquoise1, xlabel="t", ylabel="C(t)",
            legend=false, lw=4, xlims=(0,1100), ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100])

p1 = scatter!(t, data, legend=false, msw=0, ms=7,
            color=:darkorange, msa=:darkorange)
display(p1)
savefig(p1, joinpath(fileDirectory,"mle.pdf"))

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
lls = zeros(N)

for i in 1:N
    lls[i] = loglhood(data,[λs[i],Ks[i],C0s[i]],σ) - fmle # equation 12 in paper
end

# graph of log-likelihoods obtained from full likelihood
# Line gives cutoff for 95% confidence interval
q1=scatter(lls, xlabel="i", ylabel="log-likelihood[i]", legend=false)
q1=hline!([llstar], lw=2)
display(q1)

# determine the number of log-likelihoods greater than the 95% confidence interval threshold
M=0
for i in 1:N
    if lls[i] >= llstar
        global M+=1
    end
end

# evaluate model for sets of parameters that give the required log-likelihood
λsampled=zeros(M)
Ksampled=zeros(M)
C0sampled=zeros(M)
CtraceF = zeros(length(tt),M)
j=0
for i in 1:N
    if lls[i] > llstar
        global j = j + 1
        λsampled[j]=λs[i]
        Ksampled[j]=Ks[i]
        C0sampled[j]=C0s[i]
        CtraceF[:,j]=model(tt,[λs[i],Ks[i],C0s[i]],σ);
    end
end

# evaluate the lower and upper bounds of the confidence intervals
CUF = maximum(CtraceF, dims=2)
CLF = minimum(CtraceF, dims=2)

# plot the family of curves, lower and upper confidence bounds and maximum likelihood solution given data
qq1 = plotprediction(tt, CtraceF, (CUF, CLF), confColor=:gold, xlabel="t", 
            ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], yticks=[0,50,100], legend=false)

##############################################################################################
# Section 12: Prediction interval from the univariate profile likelihoods
# Compute and propogate uncertainty forward from the univariate likelihood for parameter λ
df = 1
llstar = -quantile(Chisq(df), 0.95)/2

# Function to define univariate profile for λ    
function univariateλ(λ)
    a=zeros(2)    
    function funλ(a); return loglhood(data,[λ,a[1],a[2]],σ) end

    θG=[K,C0]
    lb=[Kmin,C0min]
    ub=[Kmax,C0max]
    (xopt,fopt)=optimise(funλ,θG,lb,ub)
    llb=fopt-fmle
    return llb, xopt
end

# profile log-likelihood for lambda ##########################################################
f(x) = univariateλ(x)[1]
M=100;
λrange=LinRange(λmin,λmax,M)
ff=zeros(M)
for i in 1:M
    global ff[i] = univariateλ(λrange[i])[1]
end

q1 = plot1Dprofile(λrange, ff, llstar, λmle, xlims=(λmin,0.04), ylims=(-3,0.),
                    xlabel="λ", ylabel="ll")
display(q1)
 
#############################################################################################
# Bisection method to find values of λ in the profile that intersect the 95% confidence interval threshold for log likelihood
g(x)=f(x)[1]-llstar
ϵ=(λmax-λmin)/10^6

λλmin = find_zero(g, (λmin, λmle), atol=ϵ, Roots.Brent())
λλmax = find_zero(g, (λmle, λmax), atol=ϵ, Roots.Brent())

#############################################################################################
# Create a uniform grid of values between λ values that intersect the the 95% confidence interval threshold for log likelihood
N=100
λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

# determine MLE values for K and C0 at each value of λsampled
λsampled=LinRange(λλmin,λλmax,N)
for i in 1:N
    Ksampled[i], C0sampled[i] = univariateλ(λsampled[i])[2]
end

# compute model for given parameter values
CUnivariatetrace1 = zeros(length(tt),N)
for i in 1:N
    CUnivariatetrace1[:,i]=model(tt,[λsampled[i],Ksampled[i],C0sampled[i]],σ);
end

CU1 = maximum(CUnivariatetrace1, dims=2)
CL1 = minimum(CUnivariatetrace1, dims=2)

pp1 = plotprediction(tt, CUnivariatetrace1, (CU1, CL1), confColor=:red, xlabel="t", 
            ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(pp1)
savefig(pp1, joinpath(fileDirectory, "UnivariatePredictionLambda.pdf") )


##############################################################################################
# Compute and propogate uncertainty forward from the univariate likelihood for parameter K
function univariateK(K)
    a=zeros(2)    
    function funK(a); return loglhood(data,[a[1],K,a[2]],σ) end

    θG=[λ,C0]
    lb=[λmin,C0min]
    ub=[λmax,C0max]
    (xopt,fopt)=optimise(funK,θG,lb,ub)
    llb=fopt-fmle
    N=xopt
    return llb,N
end 

# profile log-likelihood for K ###############################################################
f(x) = univariateK(x)[1]
M=100;
Krange=LinRange(Kmin,Kmax,M)
ff=zeros(M)
for i in 1:M
    global ff[i]=univariateK(Krange[i])[1]
end

q2 = plot1Dprofile(Krange, ff, llstar, Kmle, xlims=(80,120), ylims=(-3,0.),
                    xlabel="K", ylabel="ll")
display(q2)

#############################################################################################
# Bisection method to find values of K that intersect the 95% confidence interval threshold for log likelihood
     
g(x)=f(x)[1]-llstar
ϵ=(Kmax-Kmin)/10^6

KKmin = find_zero(g, (Kmin, Kmle), atol=ϵ, Roots.Brent())
KKmax = find_zero(g, (Kmle, Kmax), atol=ϵ, Roots.Brent())

#############################################################################################
# Create a uniform grid of values between K values that intersect the the 95% confidence interval threshold for log likelihood
λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

# determine MLE values for λ and C0 at each value of Ksampled
Ksampled=LinRange(KKmin,KKmax,N)
for i in 1:N
    λsampled[i], C0sampled[i] = univariateK(Ksampled[i])[2]
end

# compute model for given parameter values
CUnivariatetrace2 = zeros(length(tt),N)
CU2=zeros(length(tt))
CL2=zeros(length(tt))
for i in 1:N
    CUnivariatetrace2[:,i]=model(tt,[λsampled[i],Ksampled[i],C0sampled[i]],σ);
end
   
for i in 1:length(tt)
    CU2[i] = maximum(CUnivariatetrace2[i,:])
    CL2[i] = minimum(CUnivariatetrace2[i,:])
end

pp1 = plotprediction(tt, CUnivariatetrace2, (CU2, CL2), confColor=:red, xlabel="t", 
            ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(pp1)
savefig(pp1, joinpath(fileDirectory, "UnivariatePredictionK.pdf"))

##############################################################################################
# Compute and propogate uncertainty forward from the univariate likelihood for parameter C0
function univariateC0(C0)
    a=zeros(2)    
    function funC0(a); return loglhood(data,[a[1],a[2],C0],σ) end
    
    θG=[λ,K]
    lb=[λmin,Kmin]
    ub=[λmax,Kmax]
    (xopt,fopt)=optimise(funC0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
end 

# profile log-likelihood for C0 ##########################################################
f(x) = univariateC0(x)[1]
M=100;
C0range=LinRange(0,C0max,M)
ff=zeros(M)
for i in 1:M
    global ff[i]=univariateC0(C0range[i])[1]
end

q3= plot1Dprofile(C0range, ff, llstar, C0mle, xlims=(C0min,30), ylims=(-3,0.),
                    xlabel="C(0)", ylabel="ll")
display(q3)
   
q4=plot(q1,q2,q3,layout=(1,3),legend=false)
display(q4)

savefig(q4, joinpath(fileDirectory, "univariate.pdf"))

#############################################################################################
# Bisection method to find values of C0 that intersect the 95% confidence interval threshold for log likelihood     
g(x)=f(x)[1]-llstar
ϵ=(C0max-C0min)/10^6

CC0min = find_zero(g, (C0min, C0mle), atol=ϵ, Roots.Brent())
CC0max = find_zero(g, (C0mle, C0max), atol=ϵ, Roots.Brent())
   
#############################################################################################
# Create a uniform grid of values between C0 values that intersect the the 95% confidence interval threshold for log likelihood
λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

# determine MLE values for λ and K at each value of C0sampled
C0sampled=LinRange(CC0min,CC0max,N)
for i in 1:N
    λsampled[i], Ksampled[i] = univariateC0(C0sampled[i])[2]
end

# compute model for given parameter values
CUnivariatetrace3 = zeros(length(tt),N)
CU3=zeros(length(tt))
CL3=zeros(length(tt))
for i in 1:N
    CUnivariatetrace3[:,i]=model(tt,[λsampled[i],Ksampled[i],C0sampled[i]],σ);
end

for i in 1:length(tt)
    CU3[i] = maximum(CUnivariatetrace3[i,:])
    CL3[i] = minimum(CUnivariatetrace3[i,:])
end
    
pp1 = plotprediction(tt, CUnivariatetrace3, (CU3, CL3), confColor=:red, xlabel="t", 
            ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(pp1)
savefig(pp1, joinpath(fileDirectory, "UnivariatePredictionC0.pdf"))

###########################################################################################
# Construct approximate prediction intervals from union of univariate profile intervals
# Compare to intervals obtained from the full likelihood
CU = max.(CU1, CU2, CU3)
CL = max.(CL1, CL2, CL3)

qq1 = plotprediction_comparison(tt, CtraceF, (CUF, CLF), (CU, CL), ymle.(tt),
                                xlabel="t", ylabel="C(t)", ylims=(0,120),
                                xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(qq1)
savefig(qq1, joinpath(fileDirectory, "PredictionComparisonUni.pdf"))


##############################################################################################
# Section 16: Construct bivariate profiles and associated pair-wise predictions starting with the bivariate profile likelihood for (λ,K )  
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
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        λsamples_boundary[N+count]=x1;
        Ksamples_boundary[N+count]=y;
        C0samples_boundary[N+count]=f(x1,y)[2]
    end
end 

# Plot the MLE and the 2N points identified on the boundary
a1 = plot2Dboundary((λsamples_boundary, Ksamples_boundary), (λmle, Kmle), N, 
                    xticks=[0,0.015,0.03], yticks=[80, 100, 120],
                    xlims=(0,0.03), ylims=(80,120), xlabel="λ", ylabel="K", legend=false)

display(a1)

# ls = ellipse_loglike.(λrange, fill(Kmle, M), fill(C0mle,M))

# plot1Dprofile(λrange, ls, llstar, λmle, xlims=(λmin,0.04), ylims=(-3,0.),
#                     xlabel="λ", ylabel="ll")


#############################################################################################
# Compute model for parameter values on the boundary
Ctrace1_boundary = zeros(length(tt), 2*N)
for i in 1:2*N
    Ctrace1_boundary[:,i]=model(tt, [λsamples_boundary[i], Ksamples_boundary[i],
                                C0samples_boundary[i]], σ)
end
    
CU1_boundary = maximum(Ctrace1_boundary, dims=2)
CL1_boundary = minimum(Ctrace1_boundary, dims=2)

pp1 = plotprediction(tt, Ctrace1_boundary, (CU1_boundary, CL1_boundary), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp3 = plot(a1, pp1, layout=(1,2))

display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateLK_boundary.pdf"))

#############################################################################################
# Section 16.  Instead of identifying the boundary, evaluate the log-likelihood on a Q × Q uniform grid of the bivariate 
# bivariate profile likelihood for λ and K ##################################################
# MESH method to locate points in lambda, K space that are on the 95% confidence interval threshold for log likelihood
Q=20; 
λλ=LinRange(0,0.03,Q);
KK=LinRange(80,120,Q);

# Filled contour plot of the bivariate profile likelihood function, superimpose the MLE and the curve at the threshold value
aa1=contourf(λλ, KK, (λλ,KK)->f(λλ,KK)[1], lw=0, xlabel="λ", ylabel="K", c=:greens,
            colorbar=false)
aa1=contour!(λλ, KK, (λλ,KK)->f(λλ,KK)[1], levels=[llstar], lw=4, xlabel="λ", ylabel="K",
            c=:red,legend=false)
aa1=scatter!([λmle], [Kmle], markersize=3, markershape=:circle, markercolor=:fuchsia,
            msw=0, ms=5, label=false)
for ii in 1:length(λλ)
    for jj in 1:length(KK)
        global aa1=scatter!([λλ[ii]], [KK[jj]], markersize=2, markershape=:x,
                            markercolor=:gold, msw=0, label=false)
    end
end
display(aa1)

# find the parameter combinations of λ and K that are inside the 95% confidence interval threshold for log likelihood
λsamples_grid=zeros(Q^2)
Ksamples_grid=zeros(Q^2)
C0samples_grid=zeros(Q^2)
llsamples_grid=zeros(Q^2)

count=0
for i in 1:Q
    for j in 1:Q
        global count+=1
        λsamples_grid[count]=λλ[i]
        Ksamples_grid[count]=KK[j]
        C0samples_grid[count]=f(λλ[i],KK[j])[2]
        llsamples_grid[count]=loglhood(data,
                                [λsamples_grid[count], Ksamples_grid[count], 
                                 C0samples_grid[count]],
                                σ)-fmle   
    end
end

# compute model for given parameter values inside the log likelihood threshold boundary
Ctrace1_grid = zeros(length(tt),Q^2)
count=0
for i in 1:Q^2
    if llsamples_grid[i] > llstar  
        global count+=1  
        Ctrace1_grid[:,count] = model(tt, 
                                    [λsamples_grid[i], Ksamples_grid[i], C0samples_grid[i]],
                                    σ)
    end
end

# plot model values
Ctrace_withingrid1=zeros(length(tt), count)

for i in 1:count
    Ctrace_withingrid1[:,i] = Ctrace1_grid[:,i]
end 
        
CU1_grid = maximum(Ctrace_withingrid1, dims=2)
CL1_grid = minimum(Ctrace_withingrid1, dims=2)

pp1 = plotprediction(tt, Ctrace_withingrid1, (CU1_grid, CL1_grid), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp3 = plot(aa1, pp1, layout=(1,2))
display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateLK_grid.pdf"))

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
                
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)             
                
        λsamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        Ksamples_boundary[N+count]=f(x1,y)[2]
    end
end 

a2 = plot2Dboundary((λsamples_boundary, C0samples_boundary), (λmle, C0mle), N, 
                    xlims=(0.0,0.03), ylims=(0,35), xticks=[0,0.015,0.03], yticks=[0,15,30], xlabel="λ", ylabel="C(0)", legend=false)
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

pp1 = plotprediction(tt, Ctrace2_boundary, (CU2_boundary, CL2_boundary), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp3=plot(a2, pp1, layout=(1,2))
display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateLC0_boundary.pdf")) 
            
# bivariate profile likelihood for λ and C0 #################################################
# MESH method to locate points in lambda, C0 space that are on the 95% confidence interval threshold for log likelihood
Q=20;
λλ=LinRange(0.0,0.03,Q);
CC0=LinRange(0,35,Q);

# contours of log likelihood across the 20*20 mesh
aa2=contourf(λλ, CC0, (λλ,CC0)->f(λλ,CC0)[1], lw=0, xlabel="λ", ylabel="C(0)", c=:greens,
            colorbar=false)
aa2=contour!(λλ, CC0, (λλ,CC0)->f(λλ,CC0)[1], levels=[llstar], lw=4, xlabel="λ", ylabel="C(0)",
            c=:red)
aa2=scatter!([λmle], [C0mle], markersize=3, markershape=:circle, markercolor=:fuchsia, 
            msw=0, ms=5, label=false)
for ii in 1:length(λλ)
    for jj in 1:length(CC0)
        global aa1=scatter!([λλ[ii]], [CC0[jj]], markersize=2, markershape=:x,
                            markercolor=:gold, msw=0, label=false)
    end
end
display(aa2)

# find the parameter combinations of λ and C0 that are inside the 95% confidence interval threshold for log likelihood
λsamples_grid=zeros(Q^2)
Ksamples_grid=zeros(Q^2)
C0samples_grid=zeros(Q^2)
llsamples_grid=zeros(Q^2)

count=0
for i in 1:Q
    for j in 1:Q
        global count+=1
        λsamples_grid[count]=λλ[i]
        C0samples_grid[count]=CC0[j]
        Ksamples_grid[count]=f(λλ[i],CC0[j])[2]
        llsamples_grid[count]=loglhood(data,
                                        [λsamples_grid[count], Ksamples_grid[count],
                                         C0samples_grid[count]],
                                        σ)-fmle  
    end
end

# compute model for given parameter values inside the log likelihood threshold boundary
Ctrace2_grid = zeros(length(tt),Q^2)
count=0
for i in 1:Q^2
    if llsamples_grid[i] > llstar  
        global count+=1  
        Ctrace2_grid[:,count]=model(tt, 
                                    [λsamples_grid[i], Ksamples_grid[i], C0samples_grid[i]],
                                    σ)
    end
end

# plot model values
Ctrace_withingrid2=zeros(length(tt),count)

for i in 1:count
    Ctrace_withingrid2[:,i]=Ctrace2_grid[:,i]
end 

CU2_grid = maximum(Ctrace_withingrid2, dims=2)
CL2_grid = minimum(Ctrace_withingrid2, dims=2)
    
pp1 = plotprediction(tt, Ctrace_withingrid2, (CU2_grid, CL2_grid), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp3=plot(aa2, pp1, layout=(1,2))
display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateLC0_grid.pdf"))

#############################################################################################
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter K and C0
function bivariateKC0(K,C0)
    function funKC0(a); return loglhood(data,[a[1],K,C0],σ) end

    θG = [λ]
    lb=[λmin]
    ub=[λmax]
    (xopt,fopt)  = optimise(funKC0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt[1]
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

        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        Ksamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        λsamples_boundary[N+count]=f(x1,y)[2]
    end
end 
    
a3 = plot2Dboundary((Ksamples_boundary, C0samples_boundary), (Kmle, C0mle), N, 
                    xlims=(80, 120), ylims=(C0min,35), xticks=[80,100,120], yticks=[0,15,30], xlabel="K", ylabel="C(0)", legend=false)
      
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

pp1 = plotprediction(tt, Ctrace3_boundary, (CU3_boundary, CL3_boundary), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp3=plot(a3, pp1, layout=(1,2))
display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateKC0_boundary.pdf")) 
         
# bivariate profile likelihood for K and C0 #################################################
# MESH method to locate points in lambda, C0 space that are on the 95% confidence interval threshold for log likelihood
Q=20;
KK=LinRange(80,120,Q);
CC0=LinRange(0,35,Q);

# contours of log likelihood across the 20*20 mesh
aa3=contourf(KK, CC0, (KK,CC0)->f(KK,CC0)[1], lw=0, xlabel="K", ylabel="C(0)", c=:greens,
            colorbar=false)
aa3=contour!(KK, CC0, (KK,CC0)->f(KK,CC0)[1], levels=[llstar], lw=4, xlabel="K", ylabel="C(0)",
            c=:red)
aa3=scatter!([Kmle], [C0mle], markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0,
             ms=5,label=false)
for ii in 1:length(KK)
    for jj in 1:length(CC0)
        global aa3=scatter!([KK[ii]], [CC0[jj]], markersize=2, markershape=:x, 
                            markercolor=:gold, msw=0, label=false)
    end
end
display(aa3)

# find the parameter combinations of K and C0 that are inside the 95% confidence interval threshold for log likelihood
λsamples_grid=zeros(Q^2)
Ksamples_grid=zeros(Q^2)
C0samples_grid=zeros(Q^2)
llsamples_grid=zeros(Q^2)

count=0
for i in 1:Q
    for j in 1:Q
        global count+=1
        Ksamples_grid[count]=KK[i]
        C0samples_grid[count]=CC0[j]
        λsamples_grid[count]=f(KK[i],CC0[j])[2]
        llsamples_grid[count]=loglhood(data,
                                        [λsamples_grid[count], Ksamples_grid[count],
                                        C0samples_grid[count]],
                                        σ)-fmle   
    end
end

# compute model for given parameter values inside the log likelihood threshold boundary
Ctrace3_grid = zeros(length(tt),Q^2)
count=0
for i in 1:Q^2
    if llsamples_grid[i] > llstar  
        global count+=1  
        Ctrace3_grid[:,count]=model(tt, 
                                    [λsamples_grid[i], Ksamples_grid[i], C0samples_grid[i]],
                                    σ)
    end
end

# plot model values
Ctrace_withingrid3=zeros(length(tt),count)

for i in 1:count
    Ctrace_withingrid3[:,i]=Ctrace3_grid[:,i]
end 

CU3_grid = maximum(Ctrace_withingrid3, dims=2)
CL3_grid = minimum(Ctrace_withingrid3, dims=2)

pp1 = plotprediction(tt, Ctrace_withingrid3, (CU3_grid, CL3_grid), confColor=:red,
                    xlabel="t", ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], 
                    yticks=[0,50,100], legend=false)
pp1=plot(aa3, pp1, layout=(1,2))
display(pp1)

savefig(pp1, joinpath(fileDirectory, "bivariateKC0_grid.pdf"))

###########################################################################################
# Construct approximate prediction intervals from union of bivariate profile intervals (MESH and boundary methods)
# Compare to intervals obtained from the full likelihood

# Compute the union of the three pair-wise profile predictions using the grid
CU_grid = max.(CU1_grid, CU2_grid, CU3_grid)
CL_grid = max.(CL1_grid, CL2_grid, CL3_grid)

# Compute the union of the three pair-wise profile predictions using the identified boundary
CU_boundary = max.(CU1_boundary, CU2_boundary, CU3_boundary)
CL_boundary = min.(CL1_boundary, CL2_boundary, CL3_boundary)

# Plot the family of predictions made using the boundary tracing method, the MLE and the prediction intervals defined by the full log-liklihood and the union of the three bivariate profile likelihoods 
qq1 = plotprediction_comparison(tt, CtraceF, (CUF, CLF), (CU_boundary, CL_boundary), ymle.(tt),
                                xlabel="t", ylabel="C(t)", ylims=(0,120),
                                xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(qq1)
savefig(qq1, joinpath(fileDirectory, "Bivariatecomparison_boundary.pdf"))

# Plot the family of predictions made using the grid, the MLE and the prediction intervals defined by the full log-liklihood and the union of the three bivariate profile likelihood s
qq1 = plotprediction_comparison(tt, CtraceF, (CUF, CLF), (CU_grid, CL_grid), ymle.(tt),
                                xlabel="t", ylabel="C(t)", ylims=(0,120),
                                xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(qq1)
savefig(qq1, joinpath(fileDirectory, "Bivariatecomparison_grid.pdf"))

############################################################################################
# Construct profile prediction intervals (for the mean)
icdf = -quantile(Chisq(1),0.95)/2
# find values of f(θ,t=200) that make PPL approximately == model.core.maximisedmle + icdf
function VPL_to_max(a, z::Real, t_interest::Real)
    return loglhood(data, a, σ) - ((z-model(t_interest, a, σ))/σ)^2
end
function VPL(z, t_interest)
    function f(a) return VPL_to_max(a, zi, t_interest) end

    zi=0.0
    vpls = zeros(length(z))

    for i in eachindex(z)
        zi = z[i]
        (_, vpls[i]) = optimise(f, θG, lb, ub)
    end
    return vpls
end
function VPL_to_PPL(z, t_interest)
    function f(a) return VPL_to_max(a, zi, t_interest) end

    zi=0.0
    ppls = zeros(length(z))

    for i in eachindex(z)
        zi = z[i]
        (xopt, fopt) = optimise(f, θG, lb, ub)
        ppls[i] = fopt - 0.5 * (model(t_interest, xopt, σ) - zi)^2 / σ^2
    end
    return ppls
end
z = collect(0.0:0.1:120) 

PPI_l = zeros(length(t))
PPI_u = zeros(length(t))
for (i, t_interest) in enumerate(t)
    PPLS = VPL_to_PPL(z, t_interest) .- (fmle + icdf)
    above_threshold = PPLS .> 0.0
    zi1, zi2 = findfirst(above_threshold), findlast(above_threshold)
    PPI_l[i], PPI_u[i] = z[zi1], z[zi2]
end

ymle_t = ymle.(t)
ppi = plotprediction_intervals(tt, ymle.(tt), t, ymle_t, ymle_t .- PPI_l, PPI_u .- ymle_t, 
                                xlabel="t", ylabel="C(t)", ylims=(0, 120),
                                xticks=[0, 500, 1000], yticks=[0, 50, 100], legend=false)
display(ppi)
savefig(ppi, joinpath(fileDirectory, "profileprediction_intervals.pdf"))

ppi = plotprediction_intervals_comparison(tt, CtraceF, (CUF, CLF), ymle.(tt), 
                                t, ymle_t, ymle_t .- PPI_l, PPI_u .- ymle_t,
                                xlabel="t", ylabel="C(t)", ylims=(0, 120),
                                xticks=[0, 500, 1000], yticks=[0, 50, 100], legend=false)
display(ppi)
savefig(ppi, joinpath(fileDirectory, "profileprediction_intervals_comparison.pdf"))