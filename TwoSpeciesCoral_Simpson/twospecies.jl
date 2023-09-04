using DifferentialEquations
using LinearAlgebra
using Distributions
gr();

function DE!(dC, C, p, t) #function to define process model
    λ1, λ2, δ, KK = p #parameter vector for the process model
    S = C[1] + C[2]
    dC[1] = λ1 * C[1] * (1.0 - S / KK) #define differential equations
    dC[2] = λ2 * C[2] * (1.0 - S / KK) - δ * C[2] * C[1] / KK #define differential equations
end

function odesolver(t1, λ1, λ2, δ, KK, C01, C02) #function to solve the process model
    p = (λ1, λ2, δ, KK) # parameter vector for the process model
    C0 = [C01, C02]
    tspan = (0.0, maximum(t1))  #time horizon
    prob = ODEProblem(DE!, C0, tspan, p) #define ODE model
    sol = solve(prob, saveat=t1)  #solve and save solutions
    return sol[1, :], sol[2, :]
end

function model(t1, θ) #function to solve the process model
    (y1, y2) = odesolver(t1, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6]) #solve the process model
    return y1, y2
end

function error(data11, data12, θ) #function to evaluate the loglikelihood for the data given parameters a
    (y1, y2) = model(t1, θ)
    dist1 = Normal(0.0, θ[7])
    e = loglikelihood(dist1, data11 - y1) + loglikelihood(dist1, data12 - y2)
    return e
end

function loglikelihood(θ) #function to evaluate the loglikelihood as a function of the parameters a 
    return error(data11, data12, θ)
end


λ1g=0.002; λ2g=0.002; δg=0.0; KKg=80.0; C0g=[1.0,1.0]; σg=1.0; # initial parameter estimates 
θG = [λ1g, λ2g, δg, KKg, C0g[1], C0g[2], σg] #parameter estimates
lb = [0.00, 0.00, 0.00, 60.0, 0.0, 0.0, 0.0]; #lower bound
ub = [0.01, 0.01, 0.01, 90.0, 1.0, 1.0, 3.0]; #upper bound

# observation times
t1 = [0
    769
    1140
    1488
    1876
    2233
    2602
    2889
    3213
    3621
    4028
]

data11 = [0.748717949
    0.97235023
    5.490243902
    17.89100529
    35
    56.38256703
    64.55087666
    66.61940299
    71.67362453
    80.47179487
    79.88291457
]; #data 1


data12 = [1.927065527
    0.782795699
    1.080487805
    2.113227513
    3.6
    2.74790376
    2.38089652
    1.8
    0.604574153
    1.305128205
    1.700502513]; #data 2
