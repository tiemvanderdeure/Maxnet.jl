#=MMI.@mlj_model mutable struct MaxnetBinaryClassifier <: MMI.Deterministic
    features = ""
    regularization_multiplier = 1.0
    regularization_function = default_regularization
    weight_factor = 100.
    backend = LassoBackend
    kw...
end
=#
mutable struct MaxnetBinaryClassifier <: MMI.Probabilistic
    features::Union{String, Vector{<:AbstractFeatureClass}}
    regularization_multiplier::Float64
    regularization_function
    weight_factor::Float64
    backend::MaxnetBackend
    link::GLM.Link
    kw
end

function MaxnetBinaryClassifier(; 
    features="", 
    regularization_multiplier = 1.0, regularization_function = default_regularization, 
    weight_factor = 100., backend = LassoBackend(), 
    link = CloglogLink(),
    kw...
)
    MaxnetBinaryClassifier(
        features, 
        regularization_multiplier, regularization_function, weight_factor, backend, link, kw)
end

MMI.input_scitype(::Type{<:MaxnetBinaryClassifier}) =
        MMI.Table{<:Union{<:AbstractVector{<:Continuous}, <:AbstractVector{<:Multiclass}}} #{<:Union{<:Continuous <:Multiclass}}
    
MMI.target_scitype(::Type{<:MaxnetBinaryClassifier}) = AbstractVector{Multiclass{2}}# AbstractVector{<:MMI.Finite}

MMI.load_path(::Type{<:MaxnetBinaryClassifier}) = "Maxnet.MaxnetBinaryClassifier"

function MMI.fit(m::MaxnetBinaryClassifier, verbosity::Int, X, y)
    # convert categorical to boolean
    y_boolean = Bool.(MMI.int(y) .- 1)

    # Find names of categorical columns
    keys_categorical = MMI.schema(X).names[findall(MMI.schema(X).scitypes .<: Union{Multiclass, Binary})]

    fitresult = maxnet(
        y_boolean, X, m.features; 
        keys_categorical = keys_categorical, 
        regularization_multiplier = m.regularization_multiplier,
        regularization_function = m.regularization_function,
        weight_factor = m.weight_factor,
        backend = m.backend,
        m.kw...)

    decode = y[1]
    report = nothing
    cache = nothing

    return (fitresult, decode), cache, report
end

function MMI.predict(m::MaxnetBinaryClassifier, (fitresult, decode), Xnew)
    p = predict(fitresult, Xnew; link = m.link)
    MMI.UnivariateFinite(MMI.classes(decode), [1 .- p, p])
end