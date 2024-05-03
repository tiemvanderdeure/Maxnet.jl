abstract type AbstractFeatureClass end

struct CategoricalFeature <: AbstractFeatureClass end
struct LinearFeature <: AbstractFeatureClass end
struct QuadraticFeature <: AbstractFeatureClass end
struct ProductFeature <: AbstractFeatureClass end
struct ThresholdFeature <: AbstractFeatureClass end
struct HingeFeature <: AbstractFeatureClass end

char_to_feature = Dict(
    'l' => [LinearFeature(), CategoricalFeature()],
    'q' => [QuadraticFeature()],
    'p' => [ProductFeature()],
    't' => [ThresholdFeature()],
    'h' => [HingeFeature()]
)

# Parse a string of features to feature classes
function features_from_string(s::AbstractString)
    mapreduce(vcat, collect(s)) do c 
        if !haskey(char_to_feature, c) 
            throw(error("$c is not a feature class, use one of $(keys(char_to_feature))"))
        end
        char_to_feature[c]
    end
end

# Default features based on number of presences

"""
    default_features(np)

Takes the number of presences `np` and returns a `Vector` of `AbstractFeatureClass`s that are used my maxent as default.

If `np` is less than ten, then only `LinearFeature` and `CategoricalFeature` are used.
If it is at least 10, then `QuadraticFeature` is additionally used.
If it is at least 15, then `HingeFeature` is additionally used.
If it is at least 80, then `ProductFeature` is additionally used.
"""
function default_features(np)
    features = [LinearFeature(), CategoricalFeature()]
    if np >= 10
        append!(features, [QuadraticFeature()])
    end
    if np >= 15
        append!(features, [HingeFeature()])
    end
    if np >= 80
        append!(features, [ProductFeature()])
    end
    return features
end
