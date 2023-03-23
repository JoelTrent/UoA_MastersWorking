




function bivariate_confidenceprofiles(model::LikelihoodModel, θcombinations::Vector{<:Int64}, num_points_inside_boundary::Int; 
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood, use_unsafe_optimiser::Bool=false,
     method=(:Bracketing, :Fix1axis))



    valid_profile_type = profile_type in [:EllipseApprox, :EllipseApproxAnalytical, :LogLikelihood]
    @assert valid_profile_type "Specified `profile_type` is invalid. Allowed values are :EllipseApprox, :EllipseApproxAnalytical, :LogLikelihood."

    if profile_type in [:EllipseApprox, :EllipseApproxAnalytical]
        check_ellipse_approx_exists!(model)
    end


    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)







    df = 2
    llstar = -quantile(Chisq(df), confLevel)/2
    num_vars = length(θnames)

    # Find confidence intervals for each parameter in θ
    # Search between [lb[i], θmle[i]] for the left side and [θmle[i], ub[i]] for the right side
    # If it doesn't exist in either range, then the parameter is locally unidentifiable in that range for 
    # that confidence level.

    confidenceDict = Dict{Tuple{Symbol,Symbol}, BivariateConfidenceStruct}()
    p = Dict(:ind1=>1,
            :ind2=>2, 
            :data=>data, 

            :num_vars=>num_vars,
            :targetll=>(fmle+llstar),
            :likelihoodFunc=>likelihoodFunc,
            :θranges=>(0:0, 0:0, 0:0),
            :λranges=>(0:0, 0:0, 0:0),
            :Ψ_x=>0.0)



    if method[1] == :Brent && method[2]==:fix1axis
        g(x,p) = bivariateΨ(x,p)[1]
    else
        g(x,p) = bivariateΨ_vectorsearch(x,p)[1]
    end


    for (ind1, ind2) in θcombinations

        println(ind1)
        println(ind2)

        newLb=>zeros(num_vars-2), 
        newUb=>zeros(num_vars-2),
        initGuess=>zeros(num_vars-2),

        boundsmapping2d!(newLb, model.core.θlb, θi)
        boundsmapping2d!(newUb, model.core.θub, θi)
        boundsmapping2d!(initGuess, model.core.θmle, θi)

        p[:θranges], p[:λranges] = variablemapping2dranges(p[:num_vars], ind1, ind2)

        boundarySamples = zeros(num_vars, 2*num_points)

        
        # find 2*num_points using some procedure 
        
        if method[1] == :Bracketing
            
            if method[2] == :Fix1axis
                
                count=0
                for (i, j, N) in [[ind1, ind2, div(num_points,2)], [ind2, ind1, num_points]]

                    p[:ind1]=i
                    p[:ind2]=j
                    ϵ=(ub[i]-lb[i])/10^6

                    while count < N

                        p[:Ψ_x] = rand(Uniform(lb[i], ub[i]))
                        Ψ_y0 = rand(Uniform(lb[j], ub[j]))
                        Ψ_y1 = rand(Uniform(lb[j], ub[j]))
                        
                        if ( g(Ψ_y0, p) * g(Ψ_y1, p) ) < 0 

                            count+=1
                            println(count)

                            Ψ_y1 = find_zero(g, (Ψ_y0, Ψ_y1), atol=ϵ, Roots.Brent(); p=p)

                            boundarySamples[i, count] = p[:Ψ_x]
                            boundarySamples[j, count] = Ψ_y1

                            variablemapping2d!(@view(boundarySamples[:, count]), bivariateΨ(Ψ_y1, p)[2], p[:θranges], p[:λranges])
                        end
                    end
                end

            elseif method[2] == :Vectorsearch

                insidePoints, outsidePoints = findNpointpairs_vectorsearch(p, num_points, lb, ub, ind1, ind2)

                p[:pointa] = [0.0,0.0]
                p[:uhat]   = [0.0,0.0]

                for i in 1:num_points
                    p[:pointa] .= insidePoints[:,i]
                    v_bar = outsidePoints[:,i] - insidePoints[:,i]

                    v_bar_norm = norm(v_bar, 2)
                    p[:uhat] = v_bar / v_bar_norm

                    ϵ=v_bar_norm/10^6

                    Ψ_y1 = find_zero(g, (0.0, v_bar_norm), atol=ϵ, Roots.Brent(); p=p)
                    
                    boundarySamples[[ind1, ind2], i] .= p[:pointa] + Ψ_y1*p[:uhat]
                    variablemapping2d!(@view(boundarySamples[:, i]), bivariateΨ_vectorsearch(Ψ_y1, p)[2], p[:θranges], p[:λranges])
                end
            end
        end
        
        confidenceDict[(θnames[ind1], θnames[ind2])] = BivariateConfidenceStruct((θmle[ind1],θmle[ind2]), boundarySamples, lb[[ind1, ind2]], ub[[ind1, ind2]], confLevel)
    end

    return confidenceDict, p
end




function bivariate_confidenceprofiles(model::LikelihoodModel, num_points_inside_boundary::Int; 
    confidence_level::Float64=0.95, profile_type::Symbol=:LogLikelihood, use_unsafe_optimiser::Bool=false,
     method=(:Brent,:fix1axis))


    θcombinations = combinations(1:model.core.num_pars, 2)

    return bivariate_confidenceprofiles(model, θcombinations, num_points_inside_boundary, 
            confidence_level=confidence_level, profile_type=profile_type, 
            use_unsafe_optimiser=use_unsafe_optimiser, method=method)

end