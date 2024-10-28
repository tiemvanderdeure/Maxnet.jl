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

The keywords `link`, and `clamp` are passed to [`Maxnet.predict`](@ref), while all other keywords are passed to [`maxnet`](@ref).
See the documentation of these functions for the meaning of these parameters and their defaults.

# Example
```jldoctest
using Maxnet, MLJBase
p_a, env = Maxnet.bradypus()

mach = machine(MaxnetBinaryClassifier(features = "lqp"), env, categorical(p_a))
fit!(mach)
yhat = MLJBase.predict(mach, env)
# output
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