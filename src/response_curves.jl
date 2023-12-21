function _get_range_or_levels(m, var, length) 
    if var in m.continuous_predictors
        range(extrema(m.predictor_data[var]), length)
    elseif var in m.categorical_predictors
        CategoricalArrays.levels(m.predictor_data[var])
    end
end

function _predictors_mean_or_mode(m::MaxnetModel)
    cont = m.predictor_data[m.continuous_predictors]
    cat = m.predictor_data[m.categorical_predictors]
    merge(
        map(Statistics.mean, cont),
        map(StatsBase.mode, cat)
    ) 
end


function response_curve(
    m::MaxnetModel, var; 
    link = IdentityLink(),
    reference_values = _predictors_mean_or_mode(m), 
    range_length = 100
)
    _response_curve(m, var, _get_range_or_levels(m, var, range_length), link, reference_values)
end
   
response_curves(m::MaxnetModel; kw...) =
    NamedTuple(key => response_curve(m, key, ; kw...) for key in keys(m.predictor_data))

function _response_curve(
    m::MaxnetModel,
    varname,
    values,
    link,
    reference_values
)
    model_input = map(reference_values) do v
        fill(v, length(values))
    end

    model_input[varname] .= values
    predict(m, model_input; link = link)
end