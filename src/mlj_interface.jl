mutable struct MaxnetBinaryClassifier <: MMI.Probabilistic
    features::Union{String, Vector{<:AbstractFeatureClass}}
    regularization_multiplier::Float64
    regularization_function
    addsamplestobackground::Bool
    n_knots::Integer
    weight_factor::Float64
    link::GLM.Link
    clamp::Bool
    kw
end

function MaxnetBinaryClassifier(; 
    features="", 
    regularization_multiplier = 1.0, regularization_function = default_regularization, 
    addsamplestobackground = true, n_knots = 50, weight_factor = 100., 
    link = CloglogLink(), clamp = false,
    kw...
)

    MaxnetBinaryClassifier(
        features, regularization_multiplier, regularization_function, 
        addsamplestobackground, n_knots, weight_factor, link, clamp, kw
    )
end

MMI.metadata_pkg(
    MaxnetBinaryClassifier;
    name = "Maxnet",
    uuid = "81f79f80-22f2-4e41-ab86-00c11cf0f26f",
    url = "https://github.com/tiemvanderdeure/Maxnet.jl",
    is_pure_julia = false,
    package_license = "MIT",
    is_wrapper = false    
)

MMI.metadata_model(
    MaxnetBinaryClassifier;
    input_scitype = MMI.Table(MMI.Continuous, MMI.Finite),
    target_scitype = AbstractVector{<:MMI.Finite{2}},
    load_path = "Maxnet.MaxnetBinaryClassifier",
    human_name = "Maxnet",
    reports_feature_importances=false
)

"""
$(MMI.doc_header(MaxnetBinaryClassifier))

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous` or `<:Multiclass`.

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Binary`. The first class should refer to background values,
  and the second class to presence values.

# Hyper-parameters

- `features`: Specifies which features classes to use in the model, e.g. "lqh" for linear, quadratic and hinge features. 
    See also [Maxnet.maxnet](@ref)
- `regularization_multiplier = 1.0`: 'Adjust how tight the model will fit. Increasing this will reduce overfitting.
- `regularization_function`: A function to compute the regularization of each feature class. Defaults to `Maxnet.default_regularization`
- `addsamplestobackground = true`: Controls wether to add presence values to the background.
- `n_knots = 50`: The number of knots used for Threshold and Hinge features. A higher number gives more flexibility for these features.
- `weight_factor = 100.0`: A `Float64` value to adjust the weight of the background samples.
- `link = CloglogLink()`: The link function to use when predicting. See `Maxnet.predict` 
- `clamp = false`: Clamp values passed to `MLJBase.predict` to the range the model was trained on.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions are 
  probabilistic and can be interpreted as the probability of presence.

# Fitted Parameters

The fields of `fitted_params(mach)` are:

- `fitresult`: A `Tuple` where the first entry is the `Maxnet.MaxnetModel` returned by the Maxnet algorithm
    and the second the entry is the classes of `y`

# Report

The fields of `report(mach)` are:

- `selected_variables`: A `Vector` of `Symbols` of the variables that were selected.
- `selected_features`: A `Vector` of `Maxnet.ModelMatrixColumn` with the features that were selected.
- `complexity`: the number of selected features in the model.


# Example

```@example
using MLJBase, Maxnet
p_a, env = Maxnet.bradypus()
y = coerce(p_a, Binary)
X = coerce(env, Count => Continuous)

mach = machine(MaxnetBinaryClassifier(features = "lqp"), X, y)
fit!(mach)
yhat = MLJBase.predict(mach, env)

```

"""
MaxnetBinaryClassifier

function MMI.fit(m::MaxnetBinaryClassifier, verbosity::Int, X, y)
    # convert categorical to boolean
    y_boolean = Bool.(MMI.int(y) .- 1)

    fitresult = maxnet(
        y_boolean, X; m.features,
        regularization_multiplier = m.regularization_multiplier,
        regularization_function = m.regularization_function,
        weight_factor = m.weight_factor,
        m.kw...)

    decode = MMI.classes(y)
    cache = nothing
    
    features = selected_features(fitresult)
    vars_included = mapreduce(_var_keys, (x, y) -> unique(vcat(x, y)), features; init = Symbol[])

    report = Dict(
        :complexity => length(features),
        :selected_variables => vars_included,
        :selected_features => features       
    )

    return (fitresult, decode), cache, report
end

function MMI.predict(m::MaxnetBinaryClassifier, (fitresult, decode), Xnew; 
    link = CloglogLink(), clamp = false)
    p = predict(fitresult, Xnew; link = link, clamp = clamp)
    MMI.UnivariateFinite(decode, [1 .- p, p])
end