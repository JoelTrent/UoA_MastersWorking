# Section 1: set up packages and parameter values
using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using Roots, NLopt
gr()

Random.seed!(12348)
fileDirectory = joinpath("Workflow paper examples", "Logistic Model Num Opt", "Plots")
include(joinpath("..", "plottingFunctions.jl"))

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

lambda_full_int = extrema(λsampled)
K_full_int = extrema(Ksampled)
C0_full_int = extrema(C0sampled)



# plot the family of curves, lower and upper confidence bounds and maximum likelihood solution given data
qq1 = plotPrediction(tt, CtraceF, (CUF, CLF), confColor=:gold, xlabel="t", 
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

q1 = plot1DProfile(λrange, ff, llstar, λmle, xlims=(λmin,0.04), ylims=(-3,0.),
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

pp1 = plotPrediction(tt, CUnivariatetrace1, (CU1, CL1), confColor=:red, xlabel="t", 
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

q2 = plot1DProfile(Krange, ff, llstar, Kmle, xlims=(80,120), ylims=(-3,0.),
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

pp1 = plotPrediction(tt, CUnivariatetrace2, (CU2, CL2), confColor=:red, xlabel="t", 
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

q3= plot1DProfile(C0range, ff, llstar, C0mle, xlims=(C0min,30), ylims=(-3,0.),
                    xlabel="C(0)", ylabel="ll")
display(q3)
   
q4=plot(q1,q2,q3,layout=(1,3),legend=false)
display(q4)

savefig(q4, joinpath(fileDirectory, "univariateComparison.pdf"))

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
    
pp1 = plotPrediction(tt, CUnivariatetrace3, (CU3, CL3), confColor=:red, xlabel="t", 
            ylabel="C(t)", ylims=(0,120), xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(pp1)
savefig(pp1, joinpath(fileDirectory, "UnivariatePredictionC0.pdf"))

###########################################################################################
# Construct approximate prediction intervals from union of univariate profile intervals
# Compare to intervals obtained from the full likelihood
CU = max.(CU1, CU2, CU3)
CL = max.(CL1, CL2, CL3)

qq1 = plotPredictionComparison(tt, CtraceF, (CUF, CLF), (CU, CL), 
                                xlabel="t", ylabel="C(t)", ylims=(0,120),
                                xticks=[0,500,1000], yticks=[0,50,100], legend=false)

display(qq1)
savefig(qq1, joinpath(fileDirectory, "PredictionComparisonUni.pdf"))



# 1D profile confidence intervals for all parameters
f(x) = -funmle(x)

α = f(xopt) + quantile(Chisq(1), 0.95)/2

res = [get_interval(xopt, i, f, :CICO_ONE_PASS, loss_crit=α, scale = fill(:log,length(xopt))) for i in 1:3];

for i in 1:3; update_profile_points!(res[i]) end

r1 = plot(res[1])
r2 = plot(res[2])
r3 = plot(res[3])

r4=plot(r1,r2,r3,layout=(1,3),legend=false)
display(r4)
intervals = [res[i].result[j].value for i in 1:3 for j in 1:2]

# Compare - very similar - although I'm not 100% sure if it is the same interval in other examples, or if it just works in this one.
λλmin, λλmax, KKmin, KKmax, CC0min, CC0max

lambda_full_int, K_full_int, C0_full_int

intervals

# comparing the loglikelihood's when using OUR implementation which reoptimises for nuisance parameters, we can see that sometimes it is very similar, and other times it is relatively different
univariateλ.(intervals[1:2])
univariateλ.([λλmin, λλmax])

univariateK.(intervals[3:4])
univariateK.([KKmin, KKmax])

univariateC0.(intervals[5:6])
univariateC0.([CC0min, CC0max])
