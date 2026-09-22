# A type with all metadata needed to (re)construct a column from data - but not actual data
struct ModelMatrixColumn{F<:AbstractFeatureClass}
    feature::F
    key::Symbol
    args::Tuple
end

ModelMatrixColumn(feature, key, args...) = ModelMatrixColumn(feature, key, args)

### Get ModelMatrixColumn objects from each feature, given some input data
# fallback method - get columns one continuous variable at a time and vcat
_feature_columns(cont_vars, cat_vars, f, nk) = mapreduce(k -> __columns(f, k, cont_vars[k], nk), vcat, keys(cont_vars))
# for categorical variable map over categorical variables instead
function _feature_columns(cont_vars, cat_vars, f::CategoricalFeature, nk)
    mapreduce(vcat, keys(cat_vars)) do k
        [ModelMatrixColumn(f, k, (x,)) for x in CategoricalArrays.levels(cat_vars[k])]
    end
end
# product needs its own implementation as it combines variables
function _feature_columns(cont_vars, cat_vars, f::ProductFeature, nk)
    ks = keys(cont_vars)
    n = length(cont_vars)
    mapreduce(vcat, 1:(n-1); init = Maxnet.ModelMatrixColumn[]) do i
        mapreduce(vcat, i+1:n) do j
            ModelMatrixColumn(f, ks[i], (ks[j], ))
        end
    end
end
__columns(f, k, args...) = ModelMatrixColumn(f, k) # fallback method for linear and quadratic
__columns(f::HingeFeature, k, var, nk) = [ModelMatrixColumn(f, k, r) for r in hinge_ranges(var, nk)]
__columns(f::ThresholdFeature, k, var, nk) = 
    [ModelMatrixColumn(f, k, t) for t in range(extrema(var)...; length = nk + 2)[2:nk + 1]]

# Generate the actual column from the data and the column specification
_get_column!(A, data, ::LinearFeature, key) = A .= data[key]
_get_column!(A, data, ::CategoricalFeature, key, x) = A .= data[key] .== (x)
_get_column!(A, data, ::QuadraticFeature, key) = A .= data[key].^2
_get_column!(A, data, ::ProductFeature, key, key2) = A .= data[key] .* data[key2]
_get_column!(A, data, ::HingeFeature, key, mi, ma) = A .= hingeval.(data[key], mi, ma)
_get_column!(A, data, ::ThresholdFeature, key, x) = A .= data[key] .>= x

function _get_column!(A, data, c::ModelMatrixColumn)
    _get_column!(A, data, c.feature, c.key, c.args...)
    return A
end

function _model_matrix(data, cols::Vector{<:ModelMatrixColumn})
    # pre-allocate memory
    A = zeros(Float64, length(first(data)), length(cols))

    for (i, c) in enumerate(cols)
        _get_column!(view(A, :, i), data, c)
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