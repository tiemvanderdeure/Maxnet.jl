function predict(m::MaxnetModel, x; link = CloglogLink())
    predictors = Tables.columntable(x)
    continuous_predictors = predictors[m.continuous_keys]
    categorical_predictors = predictors[m.categoricals_keys]

    # build the model matrix - need to figure out how to not build too many unnecessary columns
    mm, _ = model_matrix(continuous_predictors, categorical_predictors, m.features)

    exponent = mm * m.coefs .+ m.alpha

    if link in [CloglogLink(), LogitLink()]
        exponent .+= m.entropy
    end

    GLM.linkinv.(Ref(link), exponent)
end