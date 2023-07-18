using Roots

# f(x) = -x^2 + 1000
f(x) = exp(-x) - 4

find_zero(f, [-200, 1], Roots.Brent(), atol=0.01, verbose=true)