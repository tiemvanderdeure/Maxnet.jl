"""
    predict(m, x; link, clamp)

    Use a maxnet model to predict on new data.

# Arguments
- `m`: a MaxnetModel as returned by `maxnet`
- `x`: a `Tables.jl`-compatible table of predictors. All columns that were used to fit `m` should be present in `x`

# Keywords
- `link`: the link function used. Defaults to CloglogLink(), which is the default on the Maxent Java appliation since version 4.3.
    Alternatively, LogitLink() was the Maxent default on earlier versions. 
    To get exponential output, which can be interpreted as predicted abundance, use LogLink()
    IdentityLink() returns the exponent without any transformation.
- `clamp`: If `true`, values in `x` will be clamped to the range the model was trained on. Defaults to `false`.

# Returns
A `Vector` with the resulting predictions.

"""
function predict(m::MaxnetModel, x; link = CloglogLink(), clamp = false)
    predictors = Tables.columntable(x)
    for k in keys(predictors)
        k in keys(m.predictor_data) || error("$k is not found in the predictors")
    end

    # clamp the predictors
    if clamp
        for k in m.continuous_predictors
            predictors[k] .= Base.clamp.(predictors[k], extrema(m.predictor_data[k])...)
        end
    end

    # build the model matrix, but only the columns that have non-0 coefficients
    mm = _model_matrix(predictors, m.columns[m.coefs.nzind])

    exponent = mm * m.coefs.nzval .+ m.alpha .+ m.entropy

    GLM.linkinv.(Ref(link), exponent)
end

selected_features(m::MaxnetModel) = m.columns[m.coefs.nzind]