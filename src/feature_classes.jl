abstract type AbstractFeatureClass end

struct CategoricalFeature <: AbstractFeatureClass end
struct LinearFeature <: AbstractFeatureClass end
struct QuadraticFeature <: AbstractFeatureClass end
struct ProductFeature <: AbstractFeatureClass end
struct ThresholdFeature <: AbstractFeatureClass end
struct HingeFeature <: AbstractFeatureClass end

char_to_feature = Dict(
    'l' => [LinearFeature(), CategoricalFeature()],
    'q' => QuadraticFeature(),
    'p' => ProductFeature(),
    't' => ThresholdFeature(),
    'h' => HingeFeature()
)
function features_from_string(s::AbstractString)
    mapreduce(vcat, collect(s)) do c 
        if !haskey(char_to_feature, c) 
            throw(error("$c is not a feature class, use one of $(keys(char_to_feature))"))
        end
        char_to_feature[c]
    end
end

# The whole thing without statsmodels - translate namedtuple directly into matrices
feature_cols(cont_vars, cat_vars, ::LinearFeature, nk) = reduce(hcat, cont_vars)
feature_cols(cont_vars, cat_vars, ::CategoricalFeature, nk) = mapreduce(x -> permutedims(levels(x)) .== x, hcat, cat_vars)
feature_cols(cont_vars, cat_vars, ::QuadraticFeature, nk) = mapreduce(x -> x.^2, hcat, cont_vars)

function feature_cols(continuous_vars, cat_vars, ::ProductFeature, nk)
    # loop over all combinations, generate an interaction term, and add these together
    product_terms = mapreduce(hcat, 1:(length(continuous_vars)-1)) do i
        mapreduce(hcat, i+1:length(continuous_vars)) do j
            continuous_vars[i] .* continuous_vars[j]
        end
    end
    return product_terms
end

function feature_cols(continuous_predictors, cat_pred, ::HingeFeature, nknots = 50)
    mapreduce(hcat, continuous_predictors) do pred
        Maxnet.hinge(pred)
    end
end

function feature_cols(continuous_predictors, cat_pred, ::ThresholdFeature, nknots = 50)
    mat = mapreduce(hcat, continuous_predictors) do pred
        Maxnet.hinge(pred, nknots)
    end

    return mat
end
