```@meta
CurrentModule = Maxnet
```

## Installation
Maxnet.jl is not yet registered - install by running
```julia
]
add https://github.com/tiemvanderdeure/Maxnet.jl 
```

## Basic usage
### Fit a model
Use the `maxnet` function to generate a model. `maxnet` takes a boolean vector (where `true` encodes presences and `false` background points) and a `Tables.jl`-compatible data structure as arguments.

Maxnet.jl comes with a sample dataset of presences and background points for the sloth species _Bradypus variegatus_ (see [Philips et al., 2006](https://doi.org/10.1016/j.ecolmodel.2005.03.026) for details).

The following code fits a maxnet model for _Bradypus variegatus_ with default settings and generates the predicted suitability at each point.

```julia
using Maxnet
p_a, env = Maxnet.bradypus()
bradypus_model = maxnet(p_a, env)
prediction = Maxnet.predict(bradypus_model, env)
```

There are numerous settings that can be tweaked to change the model fit. These are documentated in the documentatoin for the `maxnet`(@ref) and `Maxnet.predict`(@ref) functions.

### Response curves
Use the `response_curve` or `response_curves` function to generate response curves for one or all predictor variables. 

## Backends
Lasso regression is the workhorse of the maxnet algorithm. By default, Maxnet.jl uses Lasso.jl to perform lasso regression. 

Users can choose to use GLMNet.jl instead of Lasso.jl. GLMNet.jl is a wrapper around the Fortran code from glmnet, whereas Lasso.jl is a pure Julia implementation of glmnet.

Lasso.jl and GLMNet.jl sometimes give slightly different solutions - see [this open issue](https://github.com/JuliaStats/Lasso.jl/issues/78). GLMNet.jl is sometimes also slighltly faster, particularly for big model matrices (e.g. when using maxnet with many predictor variables and with Hinge or Threshold features enabled).

Use `GLMNetBackend()` or `LassoBackend()` to specifiy the backend when fitting a model.

```julia
using Maxnet
p_a, env = Maxnet.bradypus()
model = maxnet(p_a, env; backend = GLMNetBackend())
```
