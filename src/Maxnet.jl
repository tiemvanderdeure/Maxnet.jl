module Maxnet

import Tables, Lasso, GLMNet, Interpolations, CategoricalArrays, GLM
import StatsAPI, StatsBase, Statistics
import MLJModelInterface

using GLM: IdentityLink, CloglogLink, LogLink, LogitLink

export IdentityLink, CloglogLink, LogLink, LogitLink # re-export relevant links
export LassoBackend, GLMNetBackend
export maxnet, predict

# Write your package code here.

include("utils.jl")
include("lasso.jl")
include("feature_classes.jl")
include("regularization.jl")
include("maxnet_function.jl")
include("predict.jl")
include("data.jl")
include("mlj_interface.jl")

end
