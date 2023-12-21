struct MaxnetModel 
    path
    features
    columns
    coefs
    alpha
    entropy
    predictor_data
    categorical_predictors
    continuous_predictors
end

function Base.show(io::IO, mime::MIME"text/plain", m::MaxnetModel)
    vars_selected = mapreduce(Maxnet._var_keys, (x, y) -> unique(vcat(x, y)), selected_features(m))

    println(io, "Fit Maxnet model")
    
    println(io, "Features classes: $(m.features)")
    println(io, "Entropy: $(m.entropy)")
    println(io, "Model complexity: $(length(m.coefs.nzval))")
    println(io, "Variables selected: $vars_selected")
end

"""
    maxnet(
        presences, predictors, [features];
        regularization_multiplier, regularization_function,
        addsamplestobackground, weight_factor, backend, 
        kw...
    )

# Arguments
- `presences`: A `BitVector` where presences are `true` and background samples are `false`
- `predictors`: A Tables.jl-compatible table of predictors. Categorical predictors should be `CategoricalVector`s

# Keywords
- `features`: Either:
    - A `Vector` of `AbstractFeatureClass` type features; or
    - A string where "l" = linear and categorical, "q" = quadratic, "p" = product, "t" = threshold, "h" = hinge; or
    - Nothing, in which case the default features based on the number of presences are used
- `regularization_multiplier`: A constant to adjust regularization, where a higher `regularization_multiplier` results in a higher penalization for features
- `regularization_function`: A function to compute a regularization for each feature. A default `regularization_function` is built in.
- `addsamplestobackground`: A boolean, where `true` adds the background samples to the predictors. Defaults to `true`.
- `n_knots`: the number of knots used for Threshold and Hinge features. Defaults to 50. Ignored if there are neither Threshold nor Hinge features
- `weight_factor`: A `Float64` to adjust the weight of the background samples. Defaults to 100.0.
- `backend`: Either `LassoBackend()` or `GLMNetBackend()`, to use Lasso.jl or GLMNet.jl fit the model.
Lasso.jl is written in pure julia, but can be slower with large model matrices (e.g. when hinge is enabled). Defaults to `LassoBackend`.
- `kw...`: Further arguments to be passed to Lasso.fit or GLMNet.glmnet

# Returns
- `model`: A model of type `MaxnetModel`

"""
function maxnet(
    presences::BitVector, predictors; 
    features = default_features(sum(presences)),
    regularization_multiplier::Float64 = 1.0,
    regularization_function = default_regularization,
    addsamplestobackground::Bool = true, weight_factor::Float64 = 100.,
    n_knots::Int = 50,
    backend::MaxnetBackend = LassoBackend(),
    kw...)
    
    _maxnet(
        presences, 
        predictors, 
        features,
        regularization_multiplier, 
        regularization_function,
        addsamplestobackground,
        weight_factor,
        n_knots,
        backend;
        kw...
    )
end
#maxnet(presences, predictors; kw...) = maxnet(presences, predictors, features; kw...)

### internal methods where features is not a keyword

# If features is a string, parse it
function _maxnet(presences::BitArray, predictors, features::String, args...; kw...)
    # automatically select if features is an empty string
    if features == ""
        _maxnet(presences, predictors, default_features(length(presences)), args...; kw...)
    else
        _maxnet(
            presences, predictors, features_from_string(features),
            args...; kw...
        )
    end
end

function _maxnet(
    presences::BitVector, 
    predictors, 
    features::Vector{<:AbstractFeatureClass},
    regularization_multiplier::Float64,
    regularization_function,
    addsamplestobackground::Bool, 
    weight_factor::Float64,
    n_knots::Int,
    backend::MaxnetBackend;
    kw...)

    # check if predictors is a table
    Tables.istable(predictors) || throw(ArgumentError("predictors must be a Tables.jl-compatible table"))

    if addsamplestobackground
        presences, predictors = addsamples(presences, predictors) # this returns a column table
    else
        predictors = Tables.columntable(predictors) # otherwise just convert to column table
    end

    # divide predictors into continuous and categorical
    keys_categorical = Tables.schema(predictors).names[findall(Tables.schema(predictors).types .<: CategoricalArrays.CategoricalValue)]
    keys_continuous = Tuple(key for key in keys(predictors) if ~(key in keys_categorical))
    continuous_predictors = predictors[keys_continuous]
    categorical_predictors = predictors[keys_categorical]

    # remove categoricalfeature if there are no categorical features
    if length(categorical_predictors) == 0
        filter!(f -> f !== CategoricalFeature(), features)
    end

    # keep only categorical feature if there are no continuous features
    if length(continuous_predictors) == 0
        filter!(f -> f == CategoricalFeature(), features)
    end

    # Specify each column
    columns = mapreduce(vcat, features) do feature
        _feature_columns(continuous_predictors, categorical_predictors, feature, n_knots)
    end
    
    # combine all into one model matrix
    mm = _model_matrix(predictors, columns)

    # Generate regularization
    reg = regularization_function(mm, getfield.(columns, :feature), presences) .* regularization_multiplier

    # Generate weights, 1 for presences, weightfactor for absences
    weights = presences .* 1. .+ (1 .- presences) .* weight_factor

    # generate lambdas
    λ = lambdas(reg, presences, weights; λmax = 4, n = 200)

    # Fit the model
    lassopath = fit_lasso_path(backend, mm, presences, wts = weights, penalty_factor = reg, λ = λ)
    
    # get the coefficients out
    coefs = SparseArrays.sparse(get_coefs(lassopath)[:, end])

    # calculate alpha, entropy
    bg = view(mm, .~presences, :) # matrix for background points
    
    rr = exp.(bg * coefs) # get the exponent (no intersect)

    raw = rr ./ sum(rr)

    entropy = -sum(raw .* log.(raw)) # entropy
    alpha = -log(sum(rr)) # intersect

    return MaxnetModel(
        lassopath,
        features,
        columns,
        coefs,
        alpha,
        entropy,
        predictors,
        keys_categorical,
        keys_continuous
    )
end

