using Maxnet
using Test

@testset "Maxnet.jl" begin
    # Write your tests here.
end

# Test stub based on the R package
using DataFrames, CSV, CategoricalArrays
bradypus = CSV.read("data/bradypus.tsv", DataFrame)
bradypus.ecoreg = categorical(bradypus.ecoreg)
p = bradypus.presence
data = bradypus[:, 2:end]
mod = maxnet(p, data)
plot(mod, type = "cloglog")
mod = amxnet(p, data, maxnet_formula(p, data, classes = "lq"))
plot(mod, "tmp6190_ann")
