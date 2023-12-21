# A type with all metadata needed to (re)construct a column from data
struct ModelMatrixColumn
    feature::AbstractFeatureClass
    key::Symbol
    args
end

ModelMatrixColumn(feature, key, args...) = ModelMatrixColumn(feature, key, args)

 # Get ModelMatrixColumn objects from each feature, given some input data
_feature_columns(cont_vars, cat_vars, f::LinearFeature, nk) = [ModelMatrixColumn(f, k) for k in keys(cont_vars)]
_feature_columns(cont_vars, cat_vars, f::QuadraticFeature, nk) = [ModelMatrixColumn(f, k) for k in keys(cont_vars)]
function _feature_columns(cont_vars, cat_vars, f::CategoricalFeature, nk)
    mapreduce(hcat, keys(cat_vars)) do k
        [ModelMatrixColumn(f, k, x) for x in CategoricalArrays.levels(cat_vars[k])]
    end
end

function _feature_columns(cont_vars, cat_vars, f::ProductFeature, nk)
    ks = keys(cont_vars)
    n = length(cont_vars)
    mapreduce(vcat, 1:(n-1)) do i
        mapreduce(vcat, i+1:n) do j
            ModelMatrixColumn(f, ks[i], (ks[j], ))
        end
    end
end
   
function _feature_columns(cont_vars, cat_vars, f::HingeFeature, nk)
    mapreduce(vcat, keys(cont_vars)) do k
        ranges = hinge_ranges(cont_vars[k], nk)
        [ModelMatrixColumn(f, k, r) for r in ranges]
    end
end

function _feature_columns(cont_vars, cat_vars, f::ThresholdFeature, nk)
    mapreduce(vcat, keys(cont_vars)) do k
        thresholds = range(extrema(cont_vars[k])...; length = nk + 2)[2:nk + 1]
        [ModelMatrixColumn(f, k, t) for t in thresholds]
    end
end

# Generate the actual column from the data and the column specification
_get_column(data, ::LinearFeature, key) = data[key]
_get_column(data, ::CategoricalFeature, key, x) = data[key] .== x
_get_column(data, ::QuadraticFeature, key) = data[key].^2
_get_column(data, ::ProductFeature, key, key2) = data[key] .* data[key2]
_get_column(data, ::HingeFeature, key, mi, ma) = hingeval.(data[key], mi, ma)
_get_column(data, ::ThresholdFeature, key, x) = data[key] .>= x

_get_column(data, c::ModelMatrixColumn) = _get_column(data, c.feature, c.key, c.args...)

function _model_matrix(data, cols::Vector{<:ModelMatrixColumn})
    # pre-allocate memory
    A = zeros(Float64, length(first(data)), length(cols))

    for (i, c) in enumerate(cols)
        A[:, i] .= _get_column(data, c)
    end
    return A
end

function _var_keys(c::ModelMatrixColumn)
    if c.feature == ProductFeature()
        [c.key, c.args[1]]
    else
        [c.key]
    end
end