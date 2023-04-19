function init_uni_profile_row_exists!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Int},
                                        profile_type::AbstractProfileType)
    for θi in θs_to_profile
        if !haskey(model.uni_profile_row_exists, (θi, profile_type))
            model.uni_profile_row_exists[(θi, profile_type)] = DefaultDict{Float64, Int}(0)
        end
    end
    return nothing
end

function init_biv_profile_row_exists!(model::LikelihoodModel, 
                                        θcombinations::Vector{Vector{Int}},
                                        profile_type::AbstractProfileType,
                                        method::AbstractBivariateMethod)
    for (ind1, ind2) in θcombinations
        if !haskey(model.biv_profile_row_exists, ((ind1, ind2), profile_type, method))
            model.biv_profile_row_exists[((ind1, ind2), profile_type, method)] = DefaultDict{Float64, Int}(0)
        end
    end
    return nothing
end

function init_uni_profiles_df(num_rows; existing_largest_row=0)
   
    uni_profiles_df = DataFrame()
    uni_profiles_df.row_ind = collect(1:num_rows) .+ existing_largest_row
    uni_profiles_df.θindex = zeros(Int, num_rows)
    uni_profiles_df.not_evaluated_internal_points = trues(num_rows)
    uni_profiles_df.not_evaluated_predictions = trues(num_rows)
    uni_profiles_df.conf_level = zeros(num_rows)
    uni_profiles_df.profile_type = Vector{AbstractProfileType}(undef, num_rows)
    uni_profiles_df.num_points = zeros(Int, num_rows)

    return uni_profiles_df
end

function init_biv_profiles_df(num_rows; existing_largest_row=0)
   
    biv_profiles_df = DataFrame()
    biv_profiles_df.row_ind = collect(1:num_rows) .+ existing_largest_row
    biv_profiles_df.θindexes = fill((0,0), num_rows)
    biv_profiles_df.not_evaluated_internal_points = trues(num_rows)
    biv_profiles_df.not_evaluated_predictions = trues(num_rows)
    biv_profiles_df.conf_level = zeros(num_rows)
    biv_profiles_df.profile_type = Vector{AbstractProfileType}(undef, num_rows)
    biv_profiles_df.method = Vector{AbstractBivariateMethod}(undef, num_rows)
    biv_profiles_df.num_points = zeros(Int, num_rows)

    return biv_profiles_df
end

"""
TO DO: Add support for user defined size of preallocation of the uni and biv profiles dataframes
"""
function initialiseLikelihoodModel(loglikefunction::Function,
    predictfunction::Union{Function, Missing},
    data::Union{Tuple, NamedTuple},
    θnames::Vector{<:Symbol},
    θinitialGuess::Vector{<:Float64},
    θlb::Vector{<:Float64},
    θub::Vector{<:Float64};
    uni_prealloaction_size=NaN,
    biv_preallocation_size=NaN)

    # Initialise CoreLikelihoodModel, finding the MLE solution
    θnameToIndex = Dict{Symbol,Int}(name=>i for (i, name) in enumerate(θnames))
    num_pars = length(θnames)

    function funmle(θ); return loglikefunction(θ, data) end
    (θmle, maximisedmle) = optimise(funmle, θinitialGuess, θlb, θub)

    corelikelihoodmodel = CoreLikelihoodModel(loglikefunction, predictfunction, data, θnames, θnameToIndex,
                                        θlb, θub, θmle, maximisedmle, num_pars)


    # conf_levels_evaluated = DefaultDict{Float64, Bool}(false)
    # When initialising a new confidence level, the first line should be written as: 
    # conf_ints_evaluated[conflevel] = DefaultDict{Union{Int, Symbol}, Bool}(false)
    # conf_ints_evaluated = Dict{Float64, DefaultDict{Union{Int, Symbol}, Bool}}()

    num_uni_profiles = 0
    num_biv_profiles = 0

    uni_profiles_df = init_uni_profiles_df(num_pars)    
    # if zero, is invalid row
    uni_profile_row_exists = Dict{Tuple{Int, AbstractProfileType}, DefaultDict{Float64, Int}}()    
    # uni_profile_row_exists = DefaultDict{Tuple{Int, Float64, AbstractProfileType}, Int}(0)
    uni_profiles_dict = Dict{Int, AbstractUnivariateConfidenceStruct}()

    num_combinations = binomial(num_pars, 2)
    biv_profiles_df = init_biv_profiles_df(num_combinations)
    # if zero, is invalid row
    biv_profile_row_exists = Dict{Tuple{Tuple{Int, Int}, AbstractProfileType, AbstractBivariateMethod}, DefaultDict{Float64, Int}}()
    biv_profiles_dict = Dict{Int, AbstractBivariateConfidenceStruct}()

    uni_predictions_dict = Dict{Int, AbstractPredictionStruct}()
    biv_predictions_dict = Dict{Int, AbstractPredictionStruct}()


    likelihoodmodel = LikelihoodModel(corelikelihoodmodel,
                                    missing, 
                                    num_uni_profiles, num_biv_profiles,
                                    uni_profiles_df, biv_profiles_df,
                                    uni_profile_row_exists, biv_profile_row_exists, 
                                    uni_profiles_dict, biv_profiles_dict,
                                    uni_predictions_dict, biv_predictions_dict)

    return likelihoodmodel
end

"""
Can be called without a prediction function
"""
function initialiseLikelihoodModel(loglikefunction::Function,
    data::Union{Tuple, NamedTuple},
    θnames::Vector{<:Symbol},
    θinitialGuess::Vector{<:Float64},
    θlb::Vector{<:Float64},
    θub::Vector{<:Float64};
    uni_prealloaction_size=NaN,
    biv_preallocation_size=NaN)

    return initialiseLikelihoodModel(loglikefunction, missing, data, θnames,
                                        θinitialGuess, θlb, θub)
end