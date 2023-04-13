function bivariate_confidenceprofile_fix1axis(bivariate_optimiser::Function, 
                                                model::LikelihoodModel, 
                                                num_points::Int, 
                                                consistent::NamedTuple, 
                                                ind1::Int, 
                                                ind2::Int,
                                                atol::Float64)

    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateΨ_ellipse_analytical

    if biv_opt_is_ellipse_analytical
        boundarySamples = zeros(2, num_points)
    else
        boundarySamples = zeros(model.core.num_pars, num_points)
    end

    count=0
    for (i, j, N) in [[ind1, ind2, div(num_points,2)], [ind2, ind1, (div(num_points,2) + rem(num_points,2))]]

        indexesSorted = i < j

        if biv_opt_is_ellipse_analytical
            p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, Ψ_x=[0.0],
                θranges=θranges, λranges=λranges, consistent=consistent)
        else
            p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, Ψ_x=[0.0],
                θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-2))
        end

        for _ in 1:N
            count +=1
            Ψ_y0, Ψ_y1 = 0.0, 0.0

            # do-while loop
            while true
                p.Ψ_x[1] = rand(Uniform(lb[i], ub[i]))
                Ψ_y0 = rand(Uniform(lb[j], ub[j]))
                Ψ_y1 = rand(Uniform(lb[j], ub[j])) 

                (( bivariate_optimiser(Ψ_y0, p) * bivariate_optimiser(Ψ_y1, p) ) ≥ 0) || break
            end

            Ψ_y1 = find_zero(bivariate_optimiser, (Ψ_y0, Ψ_y1), atol=atol, Roots.Brent(); p=p)

            if biv_opt_is_ellipse_analytical
                if indexesSorted
                    boundarySamples[:, count] .= p.Ψ_x[1], Ψ_y1
                else
                    boundarySamples[:, count] .= Ψ_y1, p.Ψ_x[1]
                end
            else
                boundarySamples[i, count] = p.Ψ_x[1]
                boundarySamples[j, count] = Ψ_y1

                variablemapping2d!(@view(boundarySamples[:, count]), p.λ_opt, θranges, λranges)
            end
        end
    end
    return boundarySamples
end
