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
    println(io, "Model complexity: $(complexity(m))")
    println(io, "Variables selected: $vars_selected")
end

complexity(m::MaxnetModel) = length(m.coefs.nzval)