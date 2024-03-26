var documenterSearchIndex = {"docs":
[{"location":"reference/api/","page":"API reference","title":"API reference","text":"CurrentModule = Maxnet","category":"page"},{"location":"reference/api/#API","page":"API reference","title":"API","text":"","category":"section"},{"location":"reference/api/#Index","page":"API reference","title":"Index","text":"","category":"section"},{"location":"reference/api/","page":"API reference","title":"API reference","text":"","category":"page"},{"location":"reference/api/#Functions","page":"API reference","title":"Functions","text":"","category":"section"},{"location":"reference/api/","page":"API reference","title":"API reference","text":"Modules = [Maxnet]\nOrder = [:function]","category":"page"},{"location":"reference/api/#Maxnet.complexity-Tuple{Maxnet.MaxnetModel}","page":"API reference","title":"Maxnet.complexity","text":"Get the number of non-zero coefficients in the model\n\n\n\n\n\n","category":"method"},{"location":"reference/api/#Maxnet.default_features-Tuple{Any}","page":"API reference","title":"Maxnet.default_features","text":"default_features(np)\n\nTakes the number of presences np and returns a Vector of AbstractFeatureClasss that are used my maxent as default.\n\nIf np is less than ten, then only LinearFeature and CategoricalFeature are used. If it is at least 10, then QuadraticFeature is additionally used. If it is at least 15, then HingeFeature is additionally used. If it is at least 80, then ProductFeature is additionally used.\n\n\n\n\n\n","category":"method"},{"location":"reference/api/#Maxnet.maxnet-Tuple{BitVector, Any}","page":"API reference","title":"Maxnet.maxnet","text":"maxnet(\n    presences, predictors; \n    features, regularization_multiplier, regularization_function,\n    addsamplestobackground, weight_factor, \n    kw...\n)\n\nFit a model using the maxnet algorithm.\n\nArguments\n\npresences: A BitVector where presences are true and background samples are false\npredictors: A Tables.jl-compatible table of predictors. Categorical predictors should be CategoricalVectors\n\nKeywords\n\nfeatures: Either a Vector of AbstractFeatureClass to be used in the model,    or a String where \"l\" = linear and categorical, \"q\" = quadratic, \"p\" = product, \"t\" = threshold, \"h\" = hinge (e.g. \"lqh\"); or   By default, the features are based on the number of presences are used. See default_features\nregularization_multiplier: A constant to adjust regularization, where a higher regularization_multiplier results in a higher penalization for features\nregularization_function: A function to compute a regularization for each feature. A default regularization_function is built in.\naddsamplestobackground: A boolean, where true adds the background samples to the predictors. Defaults to true.\nn_knots: the number of knots used for Threshold and Hinge features. Defaults to 50. Ignored if there are neither Threshold nor Hinge features\nweight_factor: A Float64 value to adjust the weight of the background samples. Defaults to 100.0.\nkw...: Further arguments to be passed to GLMNet.glmnet\n\nReturns\n\nmodel: A model of type MaxnetModel\n\nExamples\n\nusing Maxnet\np_a, env = Maxnet.bradypus();\nbradypus_model = maxnet(p_a, env; features = \"lq\")\n\nFit Maxnet model\nFeatures classes: Maxnet.AbstractFeatureClass[LinearFeature(), CategoricalFeature(), QuadraticFeature()]\nEntropy: 6.114650341746531\nModel complexity: 21\nVariables selected: [:frs6190_ann, :h_dem, :pre6190_l1, :pre6190_l10, :pre6190_l4, :pre6190_l7, :tmn6190_ann, :vap6190_ann, :ecoreg, :cld6190_ann, :dtr6190_ann, :tmx6190_ann]\n\n\n\n\n\n","category":"method"},{"location":"reference/api/#StatsAPI.predict-Tuple{Maxnet.MaxnetModel, Any}","page":"API reference","title":"StatsAPI.predict","text":"predict(m, x; link, clamp)\n\nUse a maxnet model to predict on new data.\n\nArguments\n\nm: a MaxnetModel as returned by maxnet\nx: a Tables.jl-compatible table of predictors. All columns that were used to fit m should be present in x\n\nKeywords\n\nlink: the link function used. Defaults to CloglogLink(), which is the default on the Maxent Java appliation since version 4.3.   Alternatively, LogitLink() was the Maxent default on earlier versions.    To get exponential output, which can be interpreted as predicted abundance, use LogLink()   IdentityLink() returns the exponent without any transformation.\nclamp: If true, values in x will be clamped to the range the model was trained on. Defaults to false.\n\nReturns\n\nA Vector with the resulting predictions.\n\nExample\n\nusing Maxnet\np_a, env = Maxnet.bradypus();\nbradypus_model = maxnet(p_a, env; features = \"lq\")\nprediction = predict(bradypus_model, env)\n\n\n\n\n\n","category":"method"},{"location":"reference/api/#Types","page":"API reference","title":"Types","text":"","category":"section"},{"location":"reference/api/","page":"API reference","title":"API reference","text":"Modules = [Maxnet]\nOrder = [:type]","category":"page"},{"location":"reference/api/#Maxnet.MaxnetBinaryClassifier","page":"API reference","title":"Maxnet.MaxnetBinaryClassifier","text":"MaxnetBinaryClassifier\n\nA model type for fitting a maxnet model using `MLJ`.\n    \nUse `MaxnetBinaryClassifier()` to create an instance with default parameters, or use keyword arguments to specify parameters.\n\nThe keywords `link`, and `clamp` are passed to [`Maxnet.predict`](@ref), while all other keywords are passed to [`maxnet`](@ref).\nSee the documentation of these functions for the meaning of these parameters and their defaults.\n\n# Example\n```jldoctest\nusing Maxnet, MLJBase\np_a, env = Maxnet.bradypus()\n\nmach = machine(MaxnetBinaryClassifier(features = \"lqp\"), env, categorical(p_a))\nfit!(mach)\nyhat = MLJBase.predict(mach, env)\n# output\n```\n\n\n\n\n\n","category":"type"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"CurrentModule = Maxnet","category":"page"},{"location":"usage/quickstart/#Installation","page":"Quick start","title":"Installation","text":"","category":"section"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"Maxnet.jl is not yet registered - install by running","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"]\nadd https://github.com/tiemvanderdeure/Maxnet.jl ","category":"page"},{"location":"usage/quickstart/#Basic-usage","page":"Quick start","title":"Basic usage","text":"","category":"section"},{"location":"usage/quickstart/#Fit-a-model","page":"Quick start","title":"Fit a model","text":"","category":"section"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"Use the maxnet function to generate a model. maxnet takes a BitVector as its first arguments, where true encodes presences points and false background points. As its second argument, it takes any Tables.jl-compatible data structure with predictor variables. Categorical variables are treated differently than numeric variables and must be a CategoricalVector. Keyword arguments are used to tweak model settings.","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"predict takes a model generated by maxnet and any Tables.jl-compatible data structure.","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"Maxnet.jl comes with a sample dataset of presences and background points for the sloth species Bradypus variegatus (see Philips et al., 2006 for details).","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"The following code fits a maxnet model for Bradypus variegatus with default settings and generates the predicted suitability at each point.","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"using Maxnet\np_a, env = Maxnet.bradypus()\nbradypus_model = maxnet(p_a, env)\nprediction = predict(bradypus_model, env)","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"There are numerous settings that can be tweaked to change the model fit. These are documentated in the documentation for the maxnet and predict functions.","category":"page"},{"location":"usage/quickstart/#Model-settings","page":"Quick start","title":"Model settings","text":"","category":"section"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"The two most important settings to change when running Maxnet is the feature classes selected and the regularization factor.","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"By default, the feature classes selected depends on the number of presence points, see Maxnet.default_features. To set them manually, specify the features keyword using either a Vector of AbstractFeatureClass, or a string, where l represents LinearFeature and CategoricalFeature, q represents QuadraticFeature, p represents ProductFeature, t represents ThresholdFeature and h represents HingeFeature. ","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"For example:","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"model1 = maxnet(p_a, env; features = [LinearFeature(), CategoricalFeature(), QuadraticFeature()])\nmodel2 = maxnet(p_a, env; features = \"lqph\")","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"The regularization multiplier controls how much the algorithms penalizes complex models. A higher regularization multiplier will result in a simpler model with fewer features.","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"model3 = maxnet(p_a, env; features = \"lqph\", regularization_multiplier = 10.0)","category":"page"},{"location":"usage/quickstart/","page":"Quick start","title":"Quick start","text":"The number of features selected is shown when a model is printed in the REPL and can be accessed using complexity. Here complexity(model2) gives 48 and complexity(model3) gives 13.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Maxnet","category":"page"},{"location":"#Maxnet","page":"Home","title":"Maxnet","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status) (Image: Coverage)","category":"page"},{"location":"","page":"Home","title":"Home","text":"This is a Julia implementation of the maxnet algorithm, with all core functionality in the original R package.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Maxnet transforms input data in various ways and then uses the GLMnet algorithm to fit a lasso path, selecting the best variables and transformations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Maxnet is closely related to the Java MaxEnt application, which is widely used in species distribution modelling. It was developped by Steven Philips. See this publication for more details about maxnet.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Also see the Maxent page on the site of the American Museum for Natural History.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Documentation for Maxnet.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"usage/mlj/","page":"MLJ","title":"MLJ","text":"CurrentModule = Maxnet","category":"page"},{"location":"usage/mlj/#Integration-with-MLJ","page":"MLJ","title":"Integration with MLJ","text":"","category":"section"},{"location":"usage/mlj/","page":"MLJ","title":"MLJ","text":"Maxnet.jl integrates with the MLJ ecosystem.","category":"page"},{"location":"usage/mlj/","page":"MLJ","title":"MLJ","text":"See MLJs project page for more info about MLJ.","category":"page"},{"location":"usage/mlj/","page":"MLJ","title":"MLJ","text":"To use Maxnet with MLJ, initialise a model by calling MaxnetBinaryClassifier, which accepts any arguments otherwise passed to maxnet. The model can then be used with MLJ's machine.","category":"page"},{"location":"usage/mlj/","page":"MLJ","title":"MLJ","text":"For example:","category":"page"},{"location":"usage/mlj/","page":"MLJ","title":"MLJ","text":"using Maxnet: MaxnetBinaryClassifier, bradypus\nusing MLJBase\n\n# sample data\ny, X = bradypus()\n\n# define a model\nmodel = MaxnetBinaryClassifier(features = \"lq\")\n\n# construct a machine\nmach = machine(model, X, categorical(y))\n\n# partition data\ntrain, test = partition(eachindex(y), 0.7, shuffle=true)\n\n# fit the machine to the data\nfit!(mach; rows = train)\n\n# predict on test data\npred_test = predict(mach; rows = test)\n\n# predict on some new dataset\npred = predict(mach, X)","category":"page"}]
}
