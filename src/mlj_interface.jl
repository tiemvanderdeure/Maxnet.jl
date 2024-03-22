mutable struct MaxnetBinaryClassifier <: MMI.Probabilistic
    features::Union{String, Vector{<:AbstractFeatureClass}}
    regularization_multiplier::Float64
    regularization_function
    weight_factor::Float64
    backend::MaxnetBackend
    link::GLM.Link
    clamp::Bool
    kw
end

function MaxnetBinaryClassifier(; 
    features="", 
    regularization_multiplier = 1.0, regularization_function = default_regularization, 
    weight_factor = 100., backend = LassoBackend(), 
    link = CloglogLink(), clamp = false,
    kw...
)

    MaxnetBinaryClassifier(
        features, regularization_multiplier, regularization_function, 
        weight_factor, backend, link, clamp, kw
    )
end

"""
    MaxnetBinaryClassifier

    A model type for fitting a maxnet model using `MLJ`.
        
    Use `MaxnetBinaryClassifier()` to create an instance with default parameters, or use keyword arguments to specify parameters.
    
    All keywords are passed to `maxnet` when calling `fit!` on a machine of this model type.
    See the documentation of [`maxnet`](@ref) for the parameters and their defaults.

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

MMI.input_scitype(::Type{<:MaxnetBinaryClassifier}) =
        MMI.Table{<:Union{<:AbstractVector{<:Continuous}, <:AbstractVector{<:Multiclass}}} #{<:Union{<:Continuous <:Multiclass}}
    
MMI.target_scitype(::Type{<:MaxnetBinaryClassifier}) = AbstractVector{Multiclass{2}}# AbstractVector{<:MMI.Finite}

MMI.load_path(::Type{<:MaxnetBinaryClassifier}) = "Maxnet.MaxnetBinaryClassifier"

function MMI.fit(m::MaxnetBinaryClassifier, verbosity::Int, X, y)
    # convert categorical to boolean
    y_boolean = Bool.(MMI.int(y) .- 1)

    fitresult = maxnet(
        y_boolean, X; m.features,
        regularization_multiplier = m.regularization_multiplier,
        regularization_function = m.regularization_function,
        weight_factor = m.weight_factor,
        backend = m.backend,
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