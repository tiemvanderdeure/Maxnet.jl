function predict(m::MaxnetModel, x; link = CloglogLink())
    
    predictors = Tables.columntable(x)
    continuous_predictors = predictors[m.keys_continuous]
    categorical_predictors = predictors[m.keys_categorical]

    # build the model matrix - need to figure out how to not build too many unnecessary columns
    mm = mapreduce(hcat, m.features) do fe
        feature_cols(continuous_predictors, categorical_predictors, fe, 10)
    end

    exponent = mm * m.coefs .+ m.alpha

    GLM.linkinv.(Ref(link), exponent)
end