abstract type AbstractFeatureClass end

struct CategoricalFeature <: AbstractFeatureClass end
struct LinearFeature <: AbstractFeatureClass end
struct QuadraticFeature <: AbstractFeatureClass end
struct ProductFeature <: AbstractFeatureClass end
struct ThresholdFeature <: AbstractFeatureClass end
struct HingeFeature <: AbstractFeatureClass end

coef_features(t::StatsModels.ContinuousTerm) = LinearFeature()
coef_features(t::StatsModels.InteractionTerm) = ProductFeature()
coef_features(t::StatsModels.CategoricalTerm) = [CategoricalFeature() for _ in 1:length(t.contrasts.levels)]
coef_features(t::ThresholdTerm) = fill(ThresholdFeature(), 1:StatsModels.width(t))
coef_features(t::HingeTerm) = fill(HingeFeature(), StatsModels.width(t))
coef_features(t::StatsModels.FunctionTerm{typeof(^), Vector{StatsModels.AbstractTerm}}) = QuadraticFeature()

coef_features(t::StatsModels.TupleTerm) = reduce(vcat, coef_features.(t))

# How should inputs be translated to terms for each feature class
feature_terms(terms, continuous_vars, ::LinearFeature) = terms
function feature_terms(terms, continuous_vars, ::ProductFeature)
    continuous_terms = terms[continuous_vars]
    # loop over all combinations, generate an interaction term, and add these together
    product_terms = mapreduce(+, 1:(length(continuous_terms)-1)) do i
        mapreduce(+, i+1:length(continuous_terms)) do j
            StatsModels.InteractionTerm((continuous_terms[i], continuous_terms[j]))
        end
    end
    return product_terms
end

function feature_terms(terms, continuous_idx, ::QuadraticFeature)
    fun_expr = :(x^2)
    exponent_term = term(2)

    return mapreduce(+, continuous_idx) do i
        term_ = terms[i]
        return StatsModels.FunctionTerm(^, [term_, exponent_term], fun_expr)
    end
end

feature_terms(terms, continuous_idx, ::HingeFeature, nknots = 50) = HingeTerm.(terms[continuous_idx], term(nknots))
feature_terms(terms, continuous_idx, ::ThresholdFeature, nknots = 50) = ThresholdTerm.(terms[continuous_idx], term(nknots))