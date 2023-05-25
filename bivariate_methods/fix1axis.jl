function findNpointpairs_fix1axis!(p::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    i::Int, 
                                    j::Int,
                                    mle_targetll::Float64,
                                    save_internal_points::Bool,
                                    biv_opt_is_ellipse_analytical::Bool)

    x_vec, y_vec = zeros(num_points), zeros(2, num_points)
    local internal_all = zeros(model.core.num_pars, save_internal_points ? num_points : 0)
    local ll_values = zeros(save_internal_points ? num_points : 0)

    Ψ_y0, Ψ_y1 = 0.0, 0.0
    
    if biv_opt_is_ellipse_analytical

        for k in 1:num_points
            # do-while loop
            while true
                p.Ψ_x[1] = rand(Uniform(model.core.θlb[i], model.core.θub[i]))
                Ψ_y0 = rand(Uniform(model.core.θlb[j], model.core.θub[j]))
                Ψ_y1 = rand(Uniform(model.core.θlb[j], model.core.θub[j])) 

                f0 = bivariate_optimiser(Ψ_y0, p)
                f1 = bivariate_optimiser(Ψ_y1, p)

                if f0 * f1 < 0
                    x_vec[k] = p.Ψ_x[1]
                    y_vec[:,k] .= Ψ_y0, Ψ_y1

                    if save_internal_points
                        internal_all[i,k] = p.Ψ_x[1]

                        if f0 ≥ 0 
                            ll_values[k] = f0
                            internal_all[j,k] = Ψ_y0
                        else
                            ll_values[k] = f1
                            internal_all[j,k] = Ψ_y1
                        end
                    end
                    break
                end
            end
        end

        if save_internal_points
            get_λs_bivariate_ellipse_analytical!(@view(internal_all[[i, j], :]), num_points,
                                                    p.consistent, i, j, 
                                                    model.core.num_pars, p.initGuess,
                                                    p.θranges, p.λranges, internal_all)
        end

    else    
        λ_opt0, λ_opt1 = zeros(model.core.num_pars-2), zeros(model.core.num_pars-2)

        for k in 1:num_points
            # do-while loop
            while true
                p.Ψ_x[1] = rand(Uniform(model.core.θlb[i], model.core.θub[i]))
                Ψ_y0 = rand(Uniform(model.core.θlb[j], model.core.θub[j]))
                Ψ_y1 = rand(Uniform(model.core.θlb[j], model.core.θub[j])) 

                f0 = bivariate_optimiser(Ψ_y0, p)
                λ_opt0 .= p.λ_opt
                f1 = bivariate_optimiser(Ψ_y1, p)
                λ_opt1 .= p.λ_opt

                if f0 * f1 < 0
                    x_vec[k] = p.Ψ_x[1]
                    y_vec[:,k] .= Ψ_y0, Ψ_y1

                    if save_internal_points
                        internal_all[i,k] = p.Ψ_x[1]
                        if f0 ≥ 0 
                            ll_values[k] = f0
                            internal_all[j,k] = Ψ_y0
                            variablemapping2d!(@view(internal_all[:, k]), λ_opt0, p.θranges, p.λranges)
                        else
                            ll_values[k] = f1
                            internal_all[j,k] = Ψ_y1
                            variablemapping2d!(@view(internal_all[:, k]), λ_opt1, p.θranges, p.λranges)
                        end
                    end
                    break
                end
            end
        end
    end

    if save_internal_points; ll_values .= ll_values .+ mle_targetll end

    return x_vec, y_vec, internal_all, ll_values
end

function bivariate_confidenceprofile_fix1axis(bivariate_optimiser::Function, 
                                                model::LikelihoodModel, 
                                                num_points::Int, 
                                                consistent::NamedTuple, 
                                                ind1::Int, 
                                                ind2::Int,
                                                mle_targetll::Float64,
                                                save_internal_points::Bool)

    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateΨ_ellipse_analytical

    boundary = zeros(model.core.num_pars, num_points)
    internal_all = zeros(model.core.num_pars, 0)
    ll_values = zeros(0)

    count=0
    for (i, j, N) in [[ind1, ind2, div(num_points,2)], [ind2, ind1, (div(num_points,2) + rem(num_points,2))]]

        if biv_opt_is_ellipse_analytical
            p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, Ψ_x=[0.0],
                θranges=θranges, λranges=λranges, consistent=consistent)
        else
            p=(ind1=i, ind2=j, newLb=newLb, newUb=newUb, initGuess=initGuess, Ψ_x=[0.0],
                θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars-2))
        end


        x_vec, y_vec, internal, ll = findNpointpairs_fix1axis!(p, bivariate_optimiser, model,
                                                            N, i, j, mle_targetll, save_internal_points,
                                                            biv_opt_is_ellipse_analytical)
        
        for k in 1:N
            count +=1

            p.Ψ_x[1] = x_vec[k]

            Ψ_y1 = find_zero(bivariate_optimiser, (y_vec[1,k], y_vec[2,k]), Roots.Brent(); p=p)

            boundary[i, count] = x_vec[k]
            boundary[j, count] = Ψ_y1
            
            if !biv_opt_is_ellipse_analytical
                variablemapping2d!(@view(boundary[:, count]), p.λ_opt, θranges, λranges)
            end
        end

        if save_internal_points
            internal_all = hcat(internal_all, internal)
            ll_values = vcat(ll_values, ll) 
        end
    end

    if biv_opt_is_ellipse_analytical
        return get_λs_bivariate_ellipse_analytical!(@view(boundary[[ind1, ind2], :]), num_points,
                                                    consistent, ind1, ind2, 
                                                    model.core.num_pars, initGuess,
                                                    θranges, λranges, boundary), PointsAndLogLikelihood(internal_all, ll_values)
    end

    return boundary, PointsAndLogLikelihood(internal_all, ll_values)
end
