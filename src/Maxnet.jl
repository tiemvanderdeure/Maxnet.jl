module Maxnet

import Tables, Lasso, GLMNet, Interpolations, CategoricalArrays, GLM, SparseArrays
import StatsAPI, StatsBase, Statistics
import MLJModelInterface as MMI

using StatsAPI: predict
using GLM: IdentityLink, CloglogLink, LogitLink, LogLink
using MLJModelInterface: Continuous, Binary, Multiclass, Count

export IdentityLink, CloglogLink, LogitLink, LogLink # re-export relevant links
export maxnet, predict, complexity
export LinearFeature, CategoricalFeature, QuadraticFeature, ProductFeature, ThresholdFeature, HingeFeature, AbstractFeatureClass
export MaxnetBinaryClassifier


include("utils.jl")
include("lasso.jl")
include("feature_classes.jl")
include("model_matrix.jl")
include("regularization.jl")
include("types.jl")
include("maxnet_function.jl")
include("predict.jl")
include("data.jl")
include("mlj_interface.jl")

end
