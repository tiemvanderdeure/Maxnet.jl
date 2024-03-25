# Maxnet

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tiemvanderdeure.github.io/Maxnet.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tiemvanderdeure.github.io/Maxnet.jl/dev/)
[![Build Status](https://github.com/tiemvanderdeure/Maxnet.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/tiemvanderdeure/Maxnet.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/tiemvanderdeure/Maxnet.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/tiemvanderdeure/Maxnet.jl)

This is a Julia implementation of the [maxnet algorithm](https://github.com/mrmaxent/maxnet), with all core functionality in the original R package.

Maxnet transforms input data in various ways and then uses the GLMnet algorithm to fit a lasso path, selecting the best variables and transformations.

Maxnet is closely related to the Java MaxEnt application, which is widely used in species distribution modelling. Maxnet was introduced in [this publication](https://doi.org/10.1111/ecog.03049), which goes more into details about the algorithm.

Also see the Maxent page on the site of the [American Museum for Natural History](https://biodiversityinformatics.amnh.org/open_source/maxent/).

A basic example, using a toy dataset included in the package:
```julia
using Maxnet
p_a, env = Maxnet.bradypus()
bradypus_model = maxnet(p_a, env)
prediction = predict(bradypus_model, env)
```

Maxnet integrates with [MLJ](https://github.com/alan-turing-institute/MLJ.jl) through the `MaxnetBinaryClassifier` type.
