using Plots, DifferentialEquations
using .Threads 
using Interpolations, Random, Distributions
using Roots, NLopt
gr()

Random.seed!(12348)
fileDirectory = joinpath("Workflow paper examples", "Lotka Volterra Model Gaussian Error", "Plots")
include(joinpath("..", "plottingFunctions.jl"))

# Workflow functions ##########################################################################
# Section 2: Define ODE model
function DE!(dC,C,p,t)
    α,β=p
    dC[1]=α*C[1]-C[1]*C[2];
    dC[2]=β*C[1]*C[2]-C[2];
end

# Section 3: Solve ODE model
function odesolver(t,α,β,C01,C02)
    p=(α,β)
    C0=[C01,C02]
    tspan=(0.0,maximum(t))
    prob=ODEProblem(DE!,C0,tspan,p)
    sol=solve(prob,saveat=t);
    cc1=sol[1,:]
    cc2=sol[2,:]
    tt=sol.t[:]
    return cc1,cc2
end

# Section 4: Define function to solve ODE model 
function model(t,a,σ)
    x=zeros(length(t))
    y=zeros(length(t))
    (x,y)=odesolver(t,a[1],a[2],a[3],a[4])
    return x,y
end

# Section 6: Define loglikelihood function
function loglhood(datax,datay,a,σ)
    x=zeros(length(t))
    y=zeros(length(t))
    (x,y)=model(t,a,σ)
    e=0.0
    f=0.0
    dist=Normal(0,σ)

    for i in 1:15   
        e+=loglikelihood(dist,datax[i]-x[i])  
        f+=loglikelihood(dist,datay[i]-y[i])
    end
    return e+f
end

# Section 7: Numerical optimisation 
function optimise(fun,θ₀,lb,ub;
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
    res = optimize(opt,θ₀)
    return res[[2,1]]
end

# Section 8: Function to be optimised for MLE
# note this function pulls in the globals, data and σ and would break if used outside of 
# this file's scope
function funmle(a); return loglhood(datax, datay, a, σ) end

# Data setup #################################################################################
# true parameters
α = 0.9; β=1.1; x0=0.8; y0=0.3; 
t=LinRange(0,10,21);
tt=LinRange(0,10,2001)
σ=0.2

# true data
(datax0, datay0) = model(t,[α,β,x0,y0],σ);

# noisy data
datax = datax0+σ*randn(length(t));
datay = datay0+σ*randn(length(t));

# Bounds on model parameters #################################################################
αmin, αmax   = (0.7, 1.2)
βmin, βmax   = (0.7, 1.4)
x0min, x0max = (0.5, 1.2)
y0min, y0max = (0.1, 0.5)

θG = [α,β,x0,y0]
lb = [αmin,βmin,x0min,y0min]
ub = [αmax,βmax,x0max,y0max]

##############################################################################################
# Section 9: Find MLE by numerical optimisation, visually compare data and MLE solution
# Use Nelder-Mead algorithm to estimate maximum likelihood solution for parameters given 
# noisy data
(xopt,fopt) = optimise(funmle,θG,lb,ub)
fmle=fopt
αmle, βmle, x0mle, y0mle = xopt # unpack mle values


(xxmle,yymle) = odesolver(tt, xopt...)
w1 = scatter(t, datax, msw=0, ms=7, color=:coral, msa=:coral)
w1 = plot!(tt, xxmle, lw=2, xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5))

w2 = scatter(t, datay, msw=0, ms=7, color=:lime, msa=:lime)
w2 = plot!(tt, yymle, lw=2, xlabel="t", ylabel="y(t)", xlims=(0,10), ylims=(0,2.5))

w3 = plot(w1, w2, layout=(1,2), legend=false)
display(w3)
savefig(w3, joinpath(fileDirectory, "mle.pdf"))

##############################################################################################
# Construct bivariate profiles and associated pair-wise predictions starting with the bivariate profile likelihood for (α,β)
df=2
llstar=-quantile(Chisq(df),0.95)/2

# Define function to compute the bivariate profile
function bivariateαβ(α,β)
    function funαβ(a); return loglhood(datax,datay,[α,β,a[1],a[2]],σ) end
    
    θG = [x0mle,y0mle]
    lb=[x0min,y0min]
    ub=[x0max,y0max]
    (xopt,fopt) = optimise(funαβ,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
end 

#############################################################################################
# bivariate profile likelihood for α and β ##################################################
# Bisection method to locate points in α, β space that are on the 95% confidence interval threshold for log likelihood
f(x,y) = bivariateαβ(x,y)
g(x,y)=f(x,y)[1]-llstar

N=100
αsamples=zeros(2*N)
βsamples=zeros(2*N)
x0samples=zeros(2*N)
y0samples=zeros(2*N)

# Fix alpha and estimate beta that puts it on the log likelihood boundary, optimising out other parameters using their MLE estimates
ϵ=(βmax-βmin)/10^3
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(αmin,αmax))
    y0=rand(Uniform(βmin,βmax))
    y1=rand(Uniform(βmin,βmax))

    # If the points (x,y0) and (x,y1) are either side of the appropriate threshold, use the 
    # bisection algorithm to find the location of the threshold on the vertical line separating 
    # the two points
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)

        αsamples[count]=x
        βsamples[count]=y1
        x0samples[count], y0samples[count] = f(x,y1)[2]
    end
end 

# Fix beta and estimate alpha that puts it on the log likelihood boundary, optimising out other parameters using their MLE estimates
ϵ=(αmax-αmin)/10^3
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(βmin,βmax))
    x0=rand(Uniform(αmin,αmax))
    x1=rand(Uniform(αmin,αmax))
    
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)

        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        αsamples[N+count]=x1;
        βsamples[N+count]=y;
        x0samples[N+count], y0samples[N+count]=f(x1,y)[2]    
    end
end 

# Plot the MLE and the 2N points identified on the boundary
a1 = plot2DBoundaryNoTicks((αsamples, βsamples), (αmle, βmle), N,
                    xlims=(0.5,1.5), ylims=(0.5,1.5), xlabel="α", ylabel="β", legend=false)

display(a1)
        
#############################################################################################
# Compute model for parameter values on the boundary
xtrace1 = zeros(length(tt),2*N);
ytrace1 = zeros(length(tt),2*N);

for i in 1:2*N
    (xtrace1[:,i],ytrace1[:,i]) = odesolver(tt, αsamples[i], βsamples[i], 
                                            x0samples[i], y0samples[i]);
end

xU1 = maximum(xtrace1, dims=2)
xL1 = minimum(xtrace1, dims=2)
yU1 = maximum(ytrace1, dims=2)
yL1 = minimum(ytrace1, dims=2)

pp1 = plotPredictionNoMLE(tt, xtrace1, (xU1, xL1), confColor=:red,
                            xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp2 = plotPredictionNoMLE(tt, ytrace1, (yU1, yL1), confColor=:red,
                            xlabel="t", ylabel="y(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp3 = plot(a1,pp1,pp2,layout=(1,3))
display(pp3)
savefig(pp3, joinpath(fileDirectory, "bivariateAB.pdf"))


#############################################################################################
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter alpha and x0
function bivariateαx0(α,x0)
    function funαx0(a); return loglhood(datax,datay,[α,a[1],x0,a[2]],σ) end
    
    θG = [βmle,y0mle]
    lb=[βmin,y0min]
    ub=[βmax,y0max]
    (xopt,fopt)  = optimise(funαx0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
end 
    
f(x,y) = bivariateαx0(x,y)
g(x,y)=f(x,y)[1]-llstar

αsamples=zeros(2*N)
βsamples=zeros(2*N)
x0samples=zeros(2*N)
y0samples=zeros(2*N)

ϵ=(x0max-x0min)/10^3
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(αmin,αmax))
    y0=rand(Uniform(x0min,x0max))
    y1=rand(Uniform(x0min,x0max))
    
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
    
        αsamples[count]=x;
        x0samples[count]=y1;
        βsamples[count], y0samples[count]=f(x,y1)[2]
    end
end 
    
ϵ=(x0max-x0min)/10^3
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(x0min,x0max))
    x0=rand(Uniform(αmin,αmax))
    x1=rand(Uniform(αmin,αmax))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)
    
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)
    
        αsamples[N+count]=x1;
        x0samples[N+count]=y;
        βsamples[N+count], y0samples[N+count]=f(x1,y)[2]
    end
end 
        
# Plot the MLE and the 2N points identified on the boundary
a2 = plot2DBoundaryNoTicks((αsamples, x0samples), (αmle, x0mle), N,
                    xlims=(αmin,αmax), ylims=(x0min,x0max), xlabel="α", ylabel="x(0)", legend=false)

display(a2)

#############################################################################################
# Compute model for parameter values on the boundary
xtrace2 = zeros(length(tt),2*N);
ytrace2 = zeros(length(tt),2*N);
    
for i in 1:2*N
    (xtrace2[:,i],ytrace2[:,i]) = odesolver(tt, αsamples[i], βsamples[i], 
                                            x0samples[i], y0samples[i])
end
    
xU2 = maximum(xtrace2, dims=2)
xL2 = minimum(xtrace2, dims=2)
yU2 = maximum(ytrace2, dims=2)
yL2 = minimum(ytrace2, dims=2)
        
pp4 = plotPredictionNoMLE(tt, xtrace2, (xU2, xL2), confColor=:red,
                            xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp5 = plotPredictionNoMLE(tt, ytrace2, (yU2, yL2), confColor=:red,
                            xlabel="t", ylabel="y(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)  

pp6 = plot(a2,pp4,pp5,layout=(1,3))
display(pp6)
savefig(pp6, joinpath(fileDirectory, "bivariateAx0.pdf"))
    
#############################################################################################
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter alpha and y0
function bivariateαy0(α,y0)
    function funαy0(a); return loglhood(datax,datay,[α,a[1],a[2],y0],σ) end
    
    θG = [βmle,x0mle]
    lb=[βmin,x0min]
    ub=[βmax,x0max]
    (xopt,fopt) = optimise(funαy0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
end 
    
f(x,y) = bivariateαy0(x,y)
g(x,y)=f(x,y)[1]-llstar
    
αsamples=zeros(2*N)
βsamples=zeros(2*N)
x0samples=zeros(2*N)
y0samples=zeros(2*N)

ϵ=(y0max-y0min)/10^3
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(αmin,αmax))
    y0=rand(Uniform(y0min,y0max))
    y1=rand(Uniform(y0min,y0max))
    
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)
        
        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
    
        αsamples[count]=x;
        y0samples[count]=y1;
        βsamples[count], x0samples[count]=f(x,y1)[2]
    end
end 
    
ϵ=(y0max-y0min)/10^3
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(y0min,y0max))
    x0=rand(Uniform(αmin,αmax))
    x1=rand(Uniform(αmin,αmax))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)
    
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)

        αsamples[N+count]=x1;
        y0samples[N+count]=y;
        βsamples[N+count]=f(x1,y)[2][1]
        x0samples[N+count]=f(x1,y)[2][2]
    end
end 
        

a3 = plot2DBoundaryNoTicks((αsamples, y0samples), (αmle, y0mle), N, 
                    xlims=(αmin,αmax), ylims=(y0min,y0max), xlabel="α", ylabel="y(0)", legend=false)

display(a3)

#############################################################################################
# Compute model for parameter values on the boundary
xtrace3 = zeros(length(tt),2*N);
ytrace3 = zeros(length(tt),2*N);

for i in 1:2*N
    (xtrace3[:,i],ytrace3[:,i]) = odesolver(tt, αsamples[i], βsamples[i],
                                            x0samples[i], y0samples[i]);
end
        
xU3 = maximum(xtrace3, dims=2)
xL3 = minimum(xtrace3, dims=2)
yU3 = maximum(ytrace3, dims=2)
yL3 = minimum(ytrace3, dims=2)
      
    
pp7 = plotPredictionNoMLE(tt, xtrace3, (xU3, xL3), confColor=:red,
                            xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp8 = plotPredictionNoMLE(tt, ytrace3, (yU3, yL3), confColor=:red,
                            xlabel="t", ylabel="y(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp9=plot(a3,pp7,pp8,layout=(1,3))
display(pp9)
savefig(pp9, "bivariateAy0.pdf")

#############################################################################################
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter beta and x0
function bivariateβx0(β,x0)
    function funβx0(a); return loglhood(datax,datay,[a[1],β,x0,a[2]],σ) end
    
    θG = [αmle,y0mle]
    lb=[αmin,y0min]
    ub=[αmax,y0max]
    (xopt,fopt)  = optimise(funβx0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
end 
    
f(x,y) = bivariateβx0(x,y)
g(x,y)=f(x,y)[1]-llstar

αsamples=zeros(2*N)
βsamples=zeros(2*N)
x0samples=zeros(2*N)
y0samples=zeros(2*N)

ϵ=(x0max-x0min)/10^3
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(βmin,βmax))
    y0=rand(Uniform(x0min,x0max))
    y1=rand(Uniform(x0min,x0max))
    
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
    
        βsamples[count]=x;
        x0samples[count]=y1;
        αsamples[count], y0samples[count]=f(x,y1)[2]
    end
end 
    
ϵ=(βmax-βmin)/10^3
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(x0min,x0max))
    x0=rand(Uniform(βmin,βmax))
    x1=rand(Uniform(βmin,βmax))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)
    
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)
    
        βsamples[N+count]=x1;
        x0samples[N+count]=y;
        αsamples[N+count]=f(x1,y)[2][1]
        y0samples[N+count]=f(x1,y)[2][2]
    end
end 

# Plot the MLE and the 2N points identified on the boundary
a4 = plot2DBoundaryNoTicks((βsamples, x0samples), (βmle, x0mle), N,
                    xlims=(βmin,βmax), ylims=(x0min,x0max), xlabel="β", ylabel="x(0)", legend=false)
display(a4)
    
#############################################################################################
# Compute model for parameter values on the boundary
xtrace4 = zeros(length(tt),2*N);
ytrace4 = zeros(length(tt),2*N);

for i in 1:2*N
    (xtrace4[:,i], ytrace4[:,i]) = odesolver(tt, αsamples[i], βsamples[i], 
                                        x0samples[i], y0samples[i])
end

xU4 = maximum(xtrace4, dims=2)
xL4 = minimum(xtrace4, dims=2)
yU4 = maximum(ytrace4, dims=2)
yL4 = minimum(ytrace4, dims=2)

pp10 = plotPredictionNoMLE(tt, xtrace4, (xU4, xL4), confColor=:red,
                            xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp11 = plotPredictionNoMLE(tt, ytrace4, (yU4, yL4), confColor=:red,
                            xlabel="t", ylabel="y(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp12=plot(a4,pp10,pp11,layout=(1,3))
display(pp12)
savefig(pp12, joinpath(fileDirectory, "bivariateBx0.pdf"))
    
#############################################################################################
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter beta and y0
function bivariateβy0(β,y0)
    function funβy0(a); return loglhood(datax,datay,[a[1],β,a[2],y0],σ) end
    
    θG = [αmle,x0mle]
    lb=[αmin,x0min]
    ub=[αmax,x0max]
    (xopt,fopt)  = optimise(funβy0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
end 
    
f(x,y) = bivariateβy0(x,y)
g(x,y)=f(x,y)[1]-llstar

αsamples=zeros(2*N)
βsamples=zeros(2*N)
x0samples=zeros(2*N)
y0samples=zeros(2*N)

ϵ=(y0max-y0min)/10^3
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(βmin,βmax))
    y0=rand(Uniform(y0min,y0max))
    y1=rand(Uniform(y0min,y0max))
    
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)

        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
    
        βsamples[count]=x;
        y0samples[count]=y1;
        αsamples[count], x0samples[count]=f(x,y1)[2]
    end
end 
    
ϵ=(βmax-βmin)/10^3
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(y0min,y0max))
    x0=rand(Uniform(βmin,βmax))
    x1=rand(Uniform(βmin,βmax))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)
    
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)
    
        βsamples[N+count]=x1;
        y0samples[N+count]=y;
        αsamples[N+count], x0samples[N+count]=f(x1,y)[2]
    end
end 

# Plot the MLE and the 2N points identified on the boundary   
a5 = plot2DBoundaryNoTicks((βsamples, y0samples), (βmle, y0mle), N,
                            xlims=(βmin,βmax), ylims=(y0min,y0max), xlabel="β", ylabel="y(0)",legend=false)

display(a5)
    
#############################################################################################
# Compute model for parameter values on the boundary  
xtrace5 = zeros(length(tt),2*N);
ytrace5 = zeros(length(tt),2*N);
    
for i in 1:2*N
    (xtrace5[:,i],ytrace5[:,i]) = odesolver(tt, αsamples[i], βsamples[i],
                                            x0samples[i], y0samples[i]);
end

xU5 = maximum(xtrace5, dims=2)
xL5 = minimum(xtrace5, dims=2)
yU5 = maximum(ytrace5, dims=2)
yL5 = minimum(ytrace5, dims=2)
    
pp13 = plotPredictionNoMLE(tt, xtrace5, (xU5, xL5), confColor=:red,
                            xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp14 = plotPredictionNoMLE(tt, ytrace5, (yU5, yL5), confColor=:red,
                            xlabel="t", ylabel="y(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)  

pp15=plot(a5,pp13,pp14,layout=(1,3))
display(pp15)
savefig(pp15, joinpath(fileDirectory, "bivariateBy0.pdf"))

#############################################################################################
# Compute and propogate uncertainty forward from the bivariate likelihood for parameter x0 and y0
function bivariatex0y0(x0,y0)
    function funx0y0(a); return loglhood(datax,datay,[a[1],a[2],x0,y0],σ) end
    
    θG = [αmle,βmle]
    lb=[αmin,βmin]
    ub=[αmax,βmax]
    (xopt,fopt) = optimise(funx0y0,θG,lb,ub)
    llb=fopt-fmle
    return llb,xopt
end 
    
f(x,y) = bivariatex0y0(x,y)
g(x,y)=f(x,y)[1]-llstar

αsamples=zeros(2*N)
βsamples=zeros(2*N)
x0samples=zeros(2*N)
y0samples=zeros(2*N)

ϵ=(y0max-y0min)/10^3
h(y,p)=g(p,y)
count=0
while count < N
    x=rand(Uniform(x0min,x0max))
    y0=rand(Uniform(y0min,y0max))
    y1=rand(Uniform(y0min,y0max))
    
    if g(x,y0)*g(x,y1) < 0 
        global count+=1
        println(count)
        
        y1 = find_zero(h, (y0, y1), atol=ϵ, Roots.Brent(); p=x)
    
        x0samples[count]=x;
        y0samples[count]=y1;
        αsamples[count], βsamples[count]=f(x,y1)[2]
    end
end 
    
ϵ=(x0max-x0min)/10^3
h(x,p)=g(x,p)
count=0
while count < N
    y=rand(Uniform(y0min,y0max))
    x0=rand(Uniform(x0min,x0max))
    x1=rand(Uniform(x0min,x0max))
        
    if g(x0,y)*g(x1,y) < 0 
        global count+=1
        println(count)
    
        x1 = find_zero(h, (x0, x1), atol=ϵ, Roots.Brent(); p=y)
    
        x0samples[N+count]=x1;
        y0samples[N+count]=y;
        αsamples[N+count], βsamples[N+count]=f(x1,y)[2]
    end
end 
        
# Plot the MLE and the 2N points identified on the boundary
a6 = plot2DBoundaryNoTicks((x0samples, y0samples), (x0mle, y0mle), N,
                    xlims=(x0min,x0max), ylims=(y0min,y0max), xlabel="x(0)", ylabel="y(0)", legend=false)

display(a6)

#############################################################################################
# Compute model for parameter values on the boundary
xtrace6 = zeros(length(tt),2*N);
ytrace6 = zeros(length(tt),2*N);

for i in 1:2*N
    (xtrace6[:,i],ytrace6[:,i]) = odesolver(tt, αsamples[i], βsamples[i], 
                                            x0samples[i], y0samples[i])
end

xU6 = maximum(xtrace6, dims=2)
xL6 = minimum(xtrace6, dims=2)
yU6 = maximum(ytrace6, dims=2)
yL6 = minimum(ytrace6, dims=2)
        
    
pp16 = plotPredictionNoMLE(tt, xtrace6, (xU6, xL6), confColor=:red,
                            xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)

pp17 = plotPredictionNoMLE(tt, ytrace6, (yU6, yL6), confColor=:red,
                            xlabel="t", ylabel="y(t)", xlims=(0,10), ylims=(0,2.5),
                            legend=false)  

pp18=plot(a6,pp16,pp17,layout=(1,3))
display(pp18)
savefig(pp18, joinpath(fileDirectory, "bivariatex0y0.pdf"))

###########################################################################################
# Construct approximate prediction intervals from union of bivariate profile intervals
XU = max.(xU1, xU2, xU3, xU4, xU5, xU6)
YU = max.(yU1, yU2, yU3, yU4, yU5, yU6)
XL = max.(xL1, xL2, xL3, xL4, xL5, xL6)
YL = min.(yL1, yL2, yL3, yL4, yL5, yL6)

pp19 = plot(tt,XL,color=:red,lw=2,legend=false)
pp19 = plot!(tt,XU,color=:red,lw=2,xlabel="t",ylabel="x(t)",legend=false,xlims=(0,10),ylims=(0,2.5))

pp20 = plot(tt,YL,color=:red,lw=2,legend=false)
pp20 = plot!(tt,YU,color=:red,lw=2,xlabel="t",ylabel="y(t)",legend=false,xlims=(0,10),ylims=(0,2.5))

pp21=plot(pp19,pp20,layout=(1,2))
display(pp21)
savefig(pp21, joinpath(fileDirectory, "Unionprofiles.pdf"))

###########################################################################################
# Let's compute and push forward from the full 4D likelihood function
df=4
llstar=-quantile(Chisq(df),0.95)/2

N=50000
αs=rand(Uniform(αmin,αmax),N);
βs=rand(Uniform(βmin,βmax),N);
x0s=rand(Uniform(x0min,x0max),N);
y0s=rand(Uniform(y0min,y0max),N);

lls=zeros(N)
for i in 1:N
    lls[i]=loglhood(datax, datay, [αs[i],βs[i],x0s[i],y0s[i]],σ)-fmle
end

q1=scatter(lls,legend=false)
q1=hline!([llstar],lw=2)
display(q1)

# determine the number of log-likelihoods greater than the 95% confidence interval threshold
M=0
for i in 1:N
    if lls[i] >= llstar
       global M+=1
    end
end

# evaluate model for sets of parameters that give the required log-likelihood
αsampled=zeros(M)
βsampled=zeros(M)
x0sampled=zeros(M)
y0sampled=zeros(M)
xtraceF = zeros(length(tt),M)
ytraceF = zeros(length(tt),M)
j=0
for i in 1:N
    if lls[i] > llstar
        global j = j + 1
        αsampled[j]=αs[i]
        βsampled[j]=βs[i]
        x0sampled[j]=x0s[i]
        y0sampled[j]=y0s[i]
        (xtraceF[:,j],ytraceF[:,j]) = odesolver(tt,αs[i],βs[i],x0s[i],y0s[i])
    end
end

# evaluate the lower and upper bounds of the confidence intervals
xUF = maximum(xtraceF, dims=2)
xLF = minimum(xtraceF, dims=2)
yUF = maximum(ytraceF, dims=2)
yLF = minimum(ytraceF, dims=2)

# Plot the family of predictions made using the boundary tracing method, the MLE and the prediction intervals defined by the full log-liklihood and the union of the 6 bivariate profile likelihoods 
qq1 = plotPredictionComparisonNoTicks(tt, xtraceF, (xLF, xUF), (XU, XL), xxmle,
                                xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5),
                                legend=false)

qq2 = plotPredictionComparisonNoTicks(tt, ytraceF, (yLF, yUF), (YU, YL), yymle,
                                xlabel="t", ylabel="x(t)", xlims=(0,10), ylims=(0,2.5),
                                legend=false)


ww1=plot(qq1, qq2, layout=(1,2),legend=false)
display(ww1)
savefig(ww1, joinpath(fileDirectory, "Comparison.pdf"))