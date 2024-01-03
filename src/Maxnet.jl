module Maxnet

import Tables, Lasso, GLMNet, Interpolations, CategoricalArrays, GLM, SparseArrays
import StatsAPI, StatsBase, Statistics
import MLJModelInterface as MMI

using GLM: IdentityLink, CloglogLink, LogitLink, LogLink
using MLJModelInterface: Continuous, Binary, Multiclass, Count

export IdentityLink, CloglogLink, LogitLink, LogLink # re-export relevant links
export LassoBackend, GLMNetBackend
export maxnet, predict, complexity
export LinearFeature, CategoricalFeature, QuadraticFeature, ProductFeature, ThresholdFeature, HingeFeature
export MaxnetBinaryClassifier

# Write your package code here.

include("utils.jl")
include("MaxnetModel.jl")
include("lasso.jl")
include("feature_classes.jl")
include("model_matrix.jl")
include("regularization.jl")
include("maxnet_function.jl")
include("predict.jl")
include("response_curves.jl")
include("data.jl")
include("mlj_interface.jl")

end
