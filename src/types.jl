struct MaxnetModel 
    path::Union{GLMNet.GLMNetPath, Lasso.LassoPath}
    features::Vector{<:AbstractFeatureClass}
    columns::Vector{ModelMatrixColumn}
    coefs::AbstractVector
    alpha::Float64
    entropy::Float64
    predictor_data
    categorical_predictors::NTuple{<:Any, Symbol}
    continuous_predictors::NTuple{<:Any, Symbol}
end

function Base.show(io::IO, mime::MIME"text/plain", m::MaxnetModel)
    vars_selected = mapreduce(Maxnet._var_keys, (x, y) -> unique(vcat(x, y)), selected_features(m); init = Symbol[])

    println(io, "Fit Maxnet model")
    
    println(io, "Features classes: $(m.features)")
    println(io, "Entropy: $(m.entropy)")
    println(io, "Model complexity: $(complexity(m))")
    println(io, "Variables selected: $vars_selected")
end

"Get the number of non-zero coefficients in the model"
complexity(m::MaxnetModel) = length(m.coefs.nzval)