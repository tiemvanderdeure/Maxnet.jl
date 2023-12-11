using DelimitedFiles
function bradypus() 
    data, headers = DelimitedFiles.readdlm("data/bradypus.tsv", '\t', Int; header = true)
    env = NamedTuple{Tuple(Symbol.(headers[2:end]))}(collect(eachcol(data)[2:end]))
    env = merge(env, (; ecoreg = CategoricalArrays.categorical(env.ecoreg)))
    p_a = Bool.(data[:, 1])
    return p_a, env
end
