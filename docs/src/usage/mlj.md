```@meta
CurrentModule = Maxnet
```

# Integration with MLJ
Maxnet.jl integrates with the MLJ ecosystem.

See [MLJs project page](https://github.com/alan-turing-institute/MLJ.jl) for more info about MLJ.

To use Maxnet with MLJ, initialise a model by calling [`MaxnetBinaryClassifier`](@ref), which accepts any arguments otherwise passed to [`maxnet`](@ref). The model can then be used with MLJ's `machine`.

For example:

```julia
using Maxnet: MaxnetBinaryClassifier, bradypus
using MLJBase

# sample data
y, X = bradypus()

# define a model
model = MaxnetBinaryClassifier(features = "lq")

# construct a machine
mach = machine(model, X, categorical(y))

# partition data
train, test = partition(eachindex(y), 0.7, shuffle=true)

# fit the machine to the data
fit!(mach; rows = train)

# predict on test data
pred_test = predict(mach; rows = test)

# predict on some new dataset
pred = predict(mach, X)
```

