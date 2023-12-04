function addsamples(presences, predictors) # this should be a general sdm thing
    ret = Int[]
    for i in eachindex(presences)
        if presences[i]
            add = true
            for j in eachindex(presences)
                if (!presences[j]) && predictors[j, :] == predictors[i, :]
                    add = false
                    break
                end
            end
            add && push!(ret, i)
        end
    end
    vcat(presences, zeros(length(ret))), vcat(predictors, predictors[ret, :])
end

function maxnet(presences, predictors, 
                formula = maxnet_formula(presences, predictors),
                regularization_multiplier = 1.0,
                regularization_function = maxnet_default_regularization,
                addsamplestobackground = true)

    disallowmissing!(predictors)
    if addsamplestobackground
        presences, predictors = addsamples(presences, predictors)
    end

    #... TODO: This function is not yet finished

end