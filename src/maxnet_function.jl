function addsamples(presences, predictors) # this should be a general sdm thing
    predictors = Tables.rowtable(predictors)

    to_add = setdiff(predictors[presences], predictors[.~presences]);

    predictors_ = Tables.columntable([predictors; to_add])
    presences_= vcat(presences, fill(false, length(to_add)))

    return presences_, predictors_
end

function lambdas(reg, p, weights; λmax = 4, n = 200)
    c = Statistics.mean(reg) * sum(p) / sum(weights)
    10 .^ range(λmax, 0; length = n) .* c
end

function model_matrix(continuous_predictors, categorical_predictors, features)
    ms = map(features) do fe
        Maxnet.feature_cols(continuous_predictors, categorical_predictors, fe, 10)
    end
    
    # combine all into one model matrix
    mm = reduce(hcat, ms)

    # Get the feature class for each column in the model matrix
    column_feature_classes = mapreduce(vcat, ms, features) do m, fe
        repeat([fe], size(m, 2))
    end

    return (mm, column_feature_classes)
end

function maxnet(presences, predictors; features = "lq",
                regularization_multiplier = 1.0,
                regularization_function = default_regularization,
                addsamplestobackground = true, weight_factor::Float64 = 100.)

    #disallowmissing!(predictors)
    if addsamplestobackground
        presences, predictors = addsamples(presences, predictors)
    else
        predictors = Tables.columntable(predictors)
    end

    features = features_from_string(features) 

    categoricals_keys = Tuple(key for key in keys(predictors) if predictors[key] isa CategoricalArrays.CategoricalArray)
    continuous_keys = Tuple(key for key in keys(predictors) if ~(predictors[key] isa CategoricalArrays.CategoricalArray))
    
    continuous_predictors = predictors[continuous_keys]
    categorical_predictors = predictors[categoricals_keys]

    # Get a matrix for each feature
    (mm, column_feature_classes) = model_matrix(continuous_predictors, categorical_predictors, features)
    
    # Generate regularization
    reg = regularization_function(mm, column_feature_classes, presences) .* regularization_multiplier

    # Generate weights, 1 for presences, weightfactor for absences
    weights = presences .* 1. .+ (1 .- presences) .* weight_factor

    # generate lambdas
    λ = lambdas(reg, presences, weights; λmax = 4, n = 200)

    machine = Lasso.fit(
        Lasso.LassoPath, mm, presences, Lasso.Distributions.Binomial(); 
        wts = weights, penalty_factor = reg, standardize = false, λ = λ,
        dofit = true, irls_maxiter = 1_000
    )

    return mm, machine
end
