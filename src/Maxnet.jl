module Maxnet

import Tables, Lasso, Interpolations
import StatsAPI
import StatsBase, Statistics

using StatsModels: term

# Write your package code here.

include("utils.jl")
#include("terms.jl")
include("feature_classes.jl")
include("regularization.jl")
include("maxnet_function.jl")
include("data.jl")

end
