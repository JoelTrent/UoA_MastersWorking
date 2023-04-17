using Distributed

addprocs(2)

@everywhere struct test; a::Float64; b::Vector; end

@everywhere function my_func(a); return (rand(), a*2.0) end
@everywhere function my_func1(a); return test(a, []) end

x = @distributed (vcat) for i in 1:10 
    my_func(rand())
end

x1 = @distributed (vcat) for i in 1:10 
    my_func1(i)
end
x1[1]


for (i,(a,b)) in enumerate(x)
    println(i)
    println(a,"  ", b)
end