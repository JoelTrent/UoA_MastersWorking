using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using Roots, NLopt
gr()

Random.seed!(12348)
fileDirectory = joinpath("Workflow paper examples", "Logistic Model Num Opt")

# Workflow functions ##########################################################################

# Logistic growth ODE
function DE!(dC, C, p, t)
    λ,K=p
    dC[1]= λ * C[1] * (1.0 - C[1]/K)
end

# model solver
function odesolver(t, λ, K, C0)
    p=(λ,K)
    tspan=(0.0, maximum(t))
    prob=ODEProblem(DE!, [C0], tspan, p)
    sol=solve(prob, saveat=t)
    return sol[1,:]
end

# model solved at each time point
function model(t, a, σ)
    y=odesolver(t, a[1],a[2],a[3])
    return y
end

# log-loglikelihood of model given data
function loglhood(data, a, σ)
    y=model(t, a, σ)
    e=0
    dist=Normal(0, σ);
    e=loglikelihood(dist, data-y) 
    return sum(e)
end

# Optimisation 
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


##############################################################################################
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
CUF=zeros(length(tt))
CLF=zeros(length(tt))
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
for i in 1:length(tt)
    CUF[i] = maximum(CtraceF[i,:])
    CLF[i] = minimum(CtraceF[i,:])
end

# plot the traces, lower and upper confidence bounds and maximum likelihood solution given data
qq1 = plot(tt, CtraceF[:,:], color=:grey, xlabel="t", ylabel="C(t)", ylims=(0,120), 
            xticks=[0,500,1000], yticks=[0,50,100], legend=false)

qq1 = plot!(ymle, 0, 1000, lw=3, color=:turquoise1)
qq1 = plot!(tt, CUF, lw=3, color=:gold)
qq1 = plot!(tt, CLF, lw=3, color=:gold)

##############################################################################################
# Compute and propogate uncertainty forward from the univariate likelihood for parameter λ
df = 1
llstar = -quantile(Chisq(df), 0.95)/2
    
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

q1=plot(λrange, ff, ylims=(-3,0.), xlims=(λmin,0.04), legend=false, lw=3)
q1=hline!([llstar], lw=3)
q1=vline!([λmle], lw=3, xlabel="λ", ylabel="ll")
display(q1)
 
#############################################################################################
# Bisection method to find values of λ that intersect the 95% confidence interval threshold for log likelihood
g(x)=f(x)[1]-llstar
ϵ=(λmax-λmin)/10^6

# find λ min
x0=λmle
x1=λmin
x2=(x1+x0)/2;

while abs(x1-x0) > ϵ
    global x2=(x1+x0)/2;
    if g(x0)*g(x2) < 0 
        global x1=x2
    else
        global x0=x2
    end
end
λλmin = x2

# find λ max
x0=λmle
x1=λmax
x2=(x1+x0)/2

while abs(x1-x0) > ϵ
    global x2=(x1+x0)/2;
    if g(x0)*g(x2) < 0 
        global x1=x2
    else
        global x0=x2
    end
end
λλmax = x2

#############################################################################################
# Create a uniform grid of values between λ values that intersect the the 95% confidence interval threshold for log likelihood
N=100
λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

# determine MLE values for K and C0 at each value of λsampled
λsampled=LinRange(λλmin,λλmax,N)
for i in 1:N
    Ksampled[i]=univariateλ(λsampled[i])[2][1]
    C0sampled[i]=univariateλ(λsampled[i])[2][2]
end

# compute model for given parameter values
CUnivariatetrace1 = zeros(length(tt),N)
CU1=zeros(length(tt))
CL1=zeros(length(tt))
for i in 1:N
    CUnivariatetrace1[:,i]=model(tt,[λsampled[i],Ksampled[i],C0sampled[i]],σ);
end

for i in 1:length(tt)
    CU1[i] = maximum(CUnivariatetrace1[i,:])
    CL1[i] = minimum(CUnivariatetrace1[i,:])
end

pp1 = plot(tt, CUnivariatetrace1[:,:], color=:grey, xlims=(0,1000), ylims=(0,120), 
            xticks=[0,500,1000], yticks=[0,50,100], legend=false, lw=3)
pp1 = plot!(tt, CU1, color=:red, lw=3)
pp1 = plot!(tt, CL1, color=:red, lw=3)
pp1 = plot!(ymle, 0, 1000, color=:turquoise1, lw=3, xlabel="t", ylabel="C(t)")

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

q2 = plot(Krange, ff, ylims=(-3,0.), xlims=(80,120), legend=false, lw=3)
q2 = hline!([llstar], lw=3)
q2 = vline!([Kmle], lw=3, xlabel="K", ylabel="ll")
display(q2)

#############################################################################################
# Bisection method to find values of K that intersect the 95% confidence interval threshold for log likelihood
     
g(x)=f(x)[1]-llstar
ϵ=(Kmax-Kmin)/10^6

# find K min
x0=Kmle
x1=Kmin
x2=(x1+x0)/2

while abs(x1-x0) > ϵ
    global x2=(x1+x0)/2;
    if g(x0)*g(x2) < 0 
        global x1=x2
    else
        global x0=x2
    end
end
KKmin = x2

# find K max
x0=Kmle
x1=Kmax
x2=(x1+x0)/2;

while abs(x1-x0) > ϵ
    global x2=(x1+x0)/2;
    if g(x0)*g(x2) < 0 
        global x1=x2
    else
        global x0=x2
    end
end
KKmax = x2

#############################################################################################
# Create a uniform grid of values between K values that intersect the the 95% confidence interval threshold for log likelihood
λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

# determine MLE values for λ and C0 at each value of Ksampled
Ksampled=LinRange(KKmin,KKmax,N)
for i in 1:N
    λsampled[i]=univariateK(Ksampled[i])[2][1]
    C0sampled[i]=univariateK(Ksampled[i])[2][2]
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
       
pp1=plot(tt ,CUnivariatetrace2[:,:], color=:grey, xlims=(0,1000), ylims=(0,120), 
            xticks=[0,500,1000], yticks=[0,50,100], legend=false, lw=3)
pp1=plot!(tt, CU2, color=:red, lw=3)
pp1=plot!(tt, CL2, color=:red, lw=3)
pp1=plot!(ymle, 0, 1000, color=:turquoise1, lw=3, xlabel="t", ylabel="C(t)")

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

q3=plot(C0range,ff,ylims=(-3,0.),xlims=(C0min,30),legend=false,lw=3)
q3=hline!([llstar], lw=3)
q3=vline!([C0mle], xlabel="C(0)",ylabel="ll", lw=3)
display(q3)
   
q4=plot(q1,q2,q3,layout=(1,3),legend=false)
display(q4)
savefig(q4, joinpath(fileDirectory, "univariate.pdf")) 
   
   
#############################################################################################
# Bisection method to find values of C0 that intersect the 95% confidence interval threshold for log likelihood     
g(x)=f(x)[1]-llstar
ϵ=(C0max-C0min)/10^6

x0=C0mle
x1=C0min
x2=(x1+x0)/2

# find C0 min
while abs(x1-x0) > ϵ
    global x2=(x1+x0)/2;
    if g(x0)*g(x2) < 0 
        global x1=x2
    else
        global x0=x2
    end
end
CC0min = x2

# find C0max
x0=C0mle
x1=C0max
x2=(x1+x0)/2;

while abs(x1-x0) > ϵ
    global x2=(x1+x0)/2;
    if g(x0)*g(x2) < 0 
        global x1=x2
    else
        global x0=x2
    end
end
CC0max = x2
   
#############################################################################################
# Create a uniform grid of values between C0 values that intersect the the 95% confidence interval threshold for log likelihood
λsampled=zeros(N)
Ksampled=zeros(N)
C0sampled=zeros(N)

# determine MLE values for λ and K at each value of C0sampled
C0sampled=LinRange(CC0min,CC0max,N)
for i in 1:N
    λsampled[i]=univariateC0(C0sampled[i])[2][1]
    Ksampled[i]=univariateC0(C0sampled[i])[2][2]
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
    
pp1 = plot(tt, CUnivariatetrace3[:,:], color=:grey, xlims=(0,1000), ylims=(0,120), 
            xticks=[0,500,1000], yticks=[0,50,100], legend=false, lw=3)
pp1 = plot!(tt, CU3, color=:red, lw=3)
pp1 = plot!(tt, CL3, color=:red, lw=3)
pp1 = plot!(ymle, 0, 1000, color=:turquoise1, lw=3, xlabel="t", ylabel="C(t)")

display(pp1)
savefig(pp1, joinpath(fileDirectory, "UnivariatePredictionC0.pdf"))

###########################################################################################
# Construct approximate prediction intervals from union of univariate profile intervals
# Compare to intervals obtained from the full likelihood
CU = max.(CU1, CU2, CU3)
CL = max.(CL1, CL2, CL3)
   
qq1 = plot(tt, CtraceF[:,:], color=:grey, xlabel="t", ylabel="C(t)", ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100], legend=false)
qq1 = plot!(tt, CUF, lw=3, color=:gold)
qq1 = plot!(tt, CLF, lw=3, color=:gold)
qq1 = plot!(tt, CU, lw=3, linestyle=:dash, color=:red)
qq1 = plot!(tt, CL, lw=3, linestyle=:dash, color=:red)
qq1 = plot!(ymle, 0, 1000, lw=3, color=:turquoise1)

display(qq1)
savefig(qq1, joinpath(fileDirectory, "PredictionComparisonUni.pdf"))


##############################################################################################
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter λ and K
df=2
llstar=-quantile(Chisq(df),0.95)/2
#Construct bivariates
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
count=0
ϵ=(Kmax-Kmin)/10^8
while count < N
    x=rand(Uniform(λmin,λmax))
    y0=rand(Uniform(Kmin,Kmax))
    y1=rand(Uniform(Kmin,Kmax))

    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)
        while abs(y1-y0) > ϵ && y1 < Kmax && y1 > Kmin
            y2=(y1+y0)/2;
            if g(x,y0)*g(x,y2) < 0 
                y1=y2
            else
                y0=y2
            end
        end

        λsamples_boundary[count]=x;
        Ksamples_boundary[count]=y1;
        C0samples_boundary[count]=f(x,y1)[2]
    end
end 

# Fix K and estimate λ that puts it on the log likelihood boundary, optimising out C0 using the MLE estimate for C0
ϵ=(λmax-λmin)/10^6
count=0
while count < N
    y=rand(Uniform(Kmin,Kmax))
    x0=rand(Uniform(λmin,λmax))
    x1=rand(Uniform(λmin,λmax))
    
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)

        while abs(x1-x0) > ϵ && x1 < λmax && x1 > λmin
            x2=(x1+x0)/2;
            if g(x0,y)*g(x2,y) < 0 
                x1=x2
            else
                x0=x2
            end
        end

        λsamples_boundary[N+count]=x1;
        Ksamples_boundary[N+count]=y;
        C0samples_boundary[N+count]=f(x1,y)[2]
    end
end 
    
a1=scatter([λmle], [Kmle], xlims=(λmin,λmax), ylims=(Kmin,Kmax), 
            markersize=3, markershape=:circle, markercolor=:fuchsia, msw=0, ms=5, 
            xlabel="λ", ylabel="K", label=false)
display(a1)

for i in 1:2*N
    global a1=scatter!([λsamples_boundary[i]], [Ksamples_boundary[i]], xlims=(0,0.03), 
                    ylims=(80,120), markersize=3, markershape=:circle, markercolor=:blue,
                    msw=0, ms=5, label=false)
end
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

pp1 = plot(tt, Ctrace1_boundary[:,:], color=:grey, xlabel="t", ylabel="C(t)",
                ylims=(0,120), xticks=[0,500,1000], yticks=[0,50,100], legend=false)
pp1 = plot!(ymle, 0, 1000, lw=3, color=:turquoise1)
pp1 = plot!(tt, CU1_boundary, lw=3, color=:red)
pp1 = plot!(tt, CL1_boundary, lw=3, color=:red)
pp3 = plot(a1, pp1, layout=(1,2))

display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateLK_boundary.pdf"))

#############################################################################################
# bivariate profile likelihood for λ and K ##################################################
# MESH method to locate points in lambda, K space that are on the 95% confidence interval threshold for log likelihood
Q=20; 
λλ=LinRange(0,0.03,Q);
KK=LinRange(80,120,Q);

# contours of log likelihood across the 20*20 mesh
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
    
pp1 = plot(tt, Ctrace_withingrid1[:,:], color=:grey, xlabel="t", ylabel="C(t)", ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100], legend=false)
pp1 = plot!(ymle, 0, 1000, lw=3, color=:turquoise1)
pp1 = plot!(tt, CU1_grid, lw=3, color=:red)
pp1 = plot!(tt, CL1_grid, lw=3, color=:red)
pp3 = plot(aa1, pp1, layout=(1,2))
display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateLK_grid.pdf"))

#############################################################################################
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter λ and K
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
count=0
while count < N 
    x=rand(Uniform(λmin,λmax))
    y0=rand(Uniform(C0min,C0max))
    y1=rand(Uniform(C0min,C0max))
            
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)
        while abs(y1-y0) > ϵ && y1 < C0max && y1 > C0min
            y2=(y1+y0)/2;
            if g(x,y0)*g(x,y2) < 0 
                y1=y2
            else
                y0=y2
            end
        end
        
        λsamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        Ksamples_boundary[count]=f(x,y1)[2]
    end
end 
        
# Fix C0 and estimate λ that puts it on the log likelihood boundary, optimising out K using the MLE estimate for K
ϵ=(λmax-λmin)/10^6
count=0
while count < N 
    y=rand(Uniform(C0min,C0max))
    x0=rand(Uniform(λmin,λmax))
    x1=rand(Uniform(λmin,λmax))
                
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)
                
        while abs(x1-x0) > ϵ && x1 < λmax && x1 > λmin
            x2=(x1+x0)/2;
            if g(x0,y)*g(x2,y) < 0 
                x1=x2
            else
                x0=x2
            end           
        end                
                
        λsamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        Ksamples_boundary[N+count]=f(x1,y)[2]
    end
end 

a2=scatter([λmle], [C0mle], xlims=(λmin,λmax), ylims=(C0min,C0max), markersize=3,
            markershape=:circle, markercolor=:fuchsia, msw=0, ms=5,
            xlabel="λ", ylabel="C(0)", label=false)
display(a2)

for i in 1:2*N
    global a2=scatter!([λsamples_boundary[i]], [C0samples_boundary[i]], xlims=(0,0.03),
                        ylims=(0,35), xticks=[0,0.015,0.03], yticks=[0,15,30], markersize=3,
                        markershape=:circle, markercolor=:blue, msw=0, ms=5, label=false)
end
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
              
pp1=plot(tt, Ctrace2_boundary[:,:], color=:grey, xlabel="t", ylabel="C(t)", 
            ylims=(0,120),xticks=[0,500,1000], yticks=[0,50,100], legend=false)
pp1=plot!(ymle, 0, 1000, lw=3, color=:turquoise1)
pp1=plot!(tt, CU2_boundary, lw=3, color=:red)
pp1=plot!(tt, CL2_boundary, lw=3,color=:red)
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
    
pp1=plot(tt, Ctrace_withingrid2[:,:], color=:grey, xlabel="t", ylabel="C(t)", ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100], legend=false)
pp1=plot!(ymle, 0, 1000, lw=3, color=:turquoise1)
pp1=plot!(tt, CU2_grid, lw=3, color=:red)
pp1=plot!(tt, CL2_grid, lw=3, color=:red)
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
count=0
while count < N
    x=rand(Uniform(Kmin,Kmax))
    y0=rand(Uniform(C0min,C0max))
    y1=rand(Uniform(C0min,C0max))
            
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)
        while abs(y1-y0) > ϵ && y1 < C0max && y1 > C0min
            y2=(y1+y0)/2;
            if g(x,y0)*g(x,y2) < 0 
                y1=y2
            else
                y0=y2
            end
        end
            
        Ksamples_boundary[count]=x;
        C0samples_boundary[count]=y1;
        λsamples_boundary[count]=f(x,y1)[2]
    end
end 

# Fix C0 and estimate K that puts it on the log likelihood boundary, optimising out λ using the MLE estimate for λ
ϵ=1(Kmax-Kmin)/10^8
count=0
while count < N 
    y=rand(Uniform(C0min,C0max))
    x0=rand(Uniform(Kmin,Kmax))
    x1=rand(Uniform(Kmin,Kmax))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)

        while abs(x1-x0) > ϵ && x1 < Kmax && x1 > Kmin
            x2=(x1+x0)/2;
            if g(x0,y)*g(x2,y) < 0 
                x1=x2
            else
                x0=x2
            end
        end

        Ksamples_boundary[N+count]=x1;
        C0samples_boundary[N+count]=y;
        λsamples_boundary[N+count]=f(x1,y)[2]
    end
end 
    
    
    
      
a3=scatter([Kmle], [C0mle], xlims=(Kmin,Kmax), ylims=(C0min,C0max), markersize=3,
            markershape=:circle, markercolor=:fuchsia, msw=0, ms=5, xlabel="K", 
            ylabel="C(0)", label=false)
display(a3)

for i in 1:2*N
    global a3=scatter!([Ksamples_boundary[i]], [C0samples_boundary[i]], xlims=(80,120),
                         ylims=(C0min,35), xticks=[80,100,120], yticks=[0,15,30],
                         markersize=3, markershape=:circle,markercolor=:blue, msw=0, 
                         ms=5, label=false)
end
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

pp1=plot(tt, Ctrace3_boundary[:,:], color=:grey, xlabel="t", ylabel="C(t)", 
            ylims=(0,120),xticks=[0,500,1000], yticks=[0,50,100], legend=false)
pp1=plot!(ymle, 0, 1000, lw=3, color=:turquoise1)
pp1=plot!(tt, CU3_boundary, lw=3, color=:red)
pp1=plot!(tt, CL3_boundary, lw=3,color=:red)
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

pp1=plot(tt, Ctrace_withingrid3[:,:], color=:grey, xlabel="t", ylabel="C(t)", ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100], legend=false)
pp1=plot!(ymle, 0, 1000, lw=3, color=:turquoise1)
pp1=plot!(tt, CU3_grid, lw=3, color=:red)
pp1=plot!(tt, CL3_grid, lw=3, color=:red)
pp1=plot(aa3, pp1, layout=(1,2))
display(pp1)

savefig(pp1, joinpath(fileDirectory, "bivariateKC0_grid.pdf"))

###########################################################################################
# Construct approximate prediction intervals from union of bivariate profile intervals (MESH and boundary methods)
# Compare to intervals obtained from the full likelihood


CU_grid = max.(CU1_grid, CU2_grid, CU3_grid)
CL_grid = max.(CL1_grid, CL2_grid, CL3_grid)

CU_boundary = max.(CU1_boundary, CU2_boundary, CU3_boundary)
CL_boundary = min.(CL1_boundary, CL2_boundary, CL3_boundary)


qq1 = plot(tt, CtraceF[:,:], color=:grey, xlabel="t", ylabel="C(t)", ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100], legend=false)
qq1 = plot!(tt, CUF, lw=3, color=:gold)
qq1 = plot!(tt, CLF, lw=3, color=:gold)
qq1 = plot!(tt, CU_boundary, lw=3, linestyle=:dash, color=:red)
qq1 = plot!(tt, CL_boundary, lw=3, linestyle=:dash, color=:red)
qq1 = plot!(ymle, 0, 1000, lw=3, color=:turquoise1)

display(qq1)
savefig(qq1, joinpath(fileDirectory, "Bivariatecomparison_boundary.pdf"))


qq1 = plot(tt, CtraceF[:,:], color=:grey, xlabel="t", ylabel="C(t)", ylims=(0,120),
            xticks=[0,500,1000], yticks=[0,50,100], legend=false)
qq1 = plot!(tt, CUF, lw=3, color=:gold)
qq1 = plot!(tt, CLF, lw=3, color=:gold)
qq1 = plot!(tt, CU_grid, lw=3, linestyle=:dash, color=:red)
qq1 = plot!(tt, CL_grid, lw=3, linestyle=:dash, color=:red)
qq1 = plot!(ymle, 0, 1000, lw=3, color=:turquoise1)

display(qq1)
savefig(qq1, joinpath(fileDirectory, "Bivariatecomparison_grid.pdf"))