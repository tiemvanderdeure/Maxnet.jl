module MaxNet
using Lasso
using StatsModels
using Missings

# Write your package code here.

include("utils.jl")
include("terms.jl")
include("maxnet_function.jl")
end
