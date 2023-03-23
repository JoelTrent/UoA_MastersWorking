using ForwardDiff

f(x::Vector) = x[1]^3 + 2*x[2] 

ForwardDiff.hessian(f, [1.0, 2.0])