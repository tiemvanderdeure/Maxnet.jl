function addsamples(presences, predictors) # this should be a general sdm thing
    predictors = Tables.rowtable(predictors)
    @assert length(presences) == length(predictors) 

    to_add = setdiff(predictors[presences], predictors[.~presences]);

    predictors_ = Tables.columntable([predictors; to_add])
    presences_= vcat(presences, fill(false, length(to_add)))

    return presences_, predictors_
end

function lambdas(reg, p, weights; λmax = 4, n = 200)
    c = Statistics.mean(reg) * sum(p) / sum(weights)
    10 .^ range(λmax, 0; length = n) .* c
end

function maxnet(presences, predictors; features = [LinearFeature(), ProductFeature()],
                regularization_multiplier = 1.0,
                regularization_function = default_regularization,
                addsamplestobackground = true, dofit = false)

    #disallowmissing!(predictors)
    if addsamplestobackground
        presences, predictors = addsamples(presences, predictors)
    else
        predictors = Tables.columntable(predictors)
    end

    predictor_keys = keys(predictors)
    terms = term.(predictor_keys)
    continuous_indices = findall(predictor_keys .!= :ecoreg) # MLJ scitype system

    # Get all the terms
    all_terms = mapreduce(+, features) do fe
        feature_terms(terms, continuous_indices, fe)
    end

    # Generate the model matrix
    sch = StatsModels.schema(predictors, Dict(:ecoreg => StatsModels.FullDummyCoding()))
    ts = StatsModels.apply_schema(all_terms, sch)
    mm = StatsModels.modelcols(StatsModels.collect_matrix_terms(ts), predictors)

    # Model matrix metadata
    column_feature_classes = coef_features(ts)

    # Generate regularization
    reg = regularization_function(mm, column_feature_classes, presences)

    # Generate weights
    weights = presences .* 1. .+ (1 .- presences) .* 100.

    # generate lambdas
    λ = lambdas(reg, presences, weights; λmax = 4, n = 200)

    machine = Lasso.fit(
        Lasso.LassoPath, mm, presences, Lasso.Distributions.Binomial(); 
        wts = weights, penalty_factor = reg, standardize = false, λ = λ,
        dofit = dofit
    )

    return machine
end
