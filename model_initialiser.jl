function initialiseLikelihoodModel(loglikefunction::Function,
    data::Union{Tuple, NamedTuple},
    θnames::Vector{<:Symbol},
    θinitialGuess::Vector{<:Float64},
    θlb::Vector{<:Float64},
    θub::Vector{<:Float64})

    # Initialise CoreLikelihoodModel, finding the MLE solution
    θnameToIndex = Dict{Symbol,Int}(name=>i for (i, name) in enumerate(θnames))
    num_pars = length(θnames)

    function funmle(θ); return loglikefunction(θ, data) end
    (θmle, maximisedmle) = optimise(funmle, θinitialGuess, θlb, θub)

    corelikelihoodmodel = CoreLikelihoodModel(loglikefunction, data, θnames, θnameToIndex,
                                        θlb, θub, θmle, maximisedmle, num_pars)


    conf_levels_evaluated = DefaultDict{Float64, Bool}(false)
    # When initialising a new confidence level, the first line should be written as: 
    # conf_ints_evaluated[conflevel] = DefaultDict{Union{Int, Symbol}, Bool}(false)
    conf_ints_evaluated = Dict{Float64, DefaultDict{Union{Int, Symbol}, Bool}}()

    likelihoodmodel = LikelihoodModel(corelikelihoodmodel,
                                    missing,
                                    conf_levels_evaluated, conf_ints_evaluated)


    return likelihoodmodel    
end