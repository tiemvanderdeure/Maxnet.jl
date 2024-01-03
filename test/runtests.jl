using Maxnet, Test, MLJBase, Statistics

p_a, env = Maxnet.bradypus()
env1 = map(e -> [e[1]], env) # just the first row

@testset "utils" begin
    @test_throws ErrorException Maxnet.features_from_string("a")
    # test each feature class is returned correctly
    @test Maxnet.features_from_string("l") == [LinearFeature(), CategoricalFeature()]
    @test Maxnet.features_from_string("q") == [QuadraticFeature()]
    @test Maxnet.features_from_string("lq") == [LinearFeature(), CategoricalFeature(), QuadraticFeature()]
    @test Maxnet.features_from_string("lqp") == [LinearFeature(), CategoricalFeature(), QuadraticFeature(), ProductFeature()]
    @test Maxnet.features_from_string("lqph") == [LinearFeature(), CategoricalFeature(), QuadraticFeature(), ProductFeature(), HingeFeature()]
    @test Maxnet.features_from_string("lqpt") == [LinearFeature(), CategoricalFeature(), QuadraticFeature(), ProductFeature(), ThresholdFeature()]

    @test Maxnet.default_features(100) == [LinearFeature(), CategoricalFeature(), QuadraticFeature(), HingeFeature(), ProductFeature()]
end

@testset "Maxnet" begin
    # test both backends work
    model_glmnet = Maxnet.maxnet((p_a), env; features = "lq", backend = GLMNetBackend());
    model_lasso = Maxnet.maxnet((p_a), env; features = "lq", backend = LassoBackend());

    # test both backends come up with approximately the same result
    @test all(isapprox.(model_glmnet.coefs, model_lasso.coefs; rtol = 0.1, atol = 0.1))
    @test Statistics.cor(model_glmnet.coefs, model_lasso.coefs) > 0.99

    # test the result
    @test model_glmnet.entropy ≈ 6.114650341746531
    @test complexity(model_glmnet) == 21

    # some class combinations and keywords
    Maxnet.maxnet(p_a, env; features = "lq", backend = LassoBackend());
    Maxnet.maxnet(p_a, env; features = "lqp", regularization_multiplier = 2., backend = LassoBackend());
    Maxnet.maxnet(p_a, env; features = "lqh", regularization_multiplier = 5., nknots = 10, backend = LassoBackend());
    Maxnet.maxnet(p_a, env; features = "lqph", weight_factor = 10., backend = LassoBackend());

    # predictions
    prediction = Maxnet.predict(model_lasso, env)
    @test Statistics.mean(prediction[p_a]) > Statistics.mean(prediction[.~p_a])
    @test minimum(prediction) > 0.
    @test maximum(prediction) < 1.
    @test mean(prediction) ≈ 0.243406167194403

    # check that clamping works
    # clamp shouldn't change anything in this case
    @test prediction == Maxnet.predict(model_lasso, env; clamp = true)
    
    # predict with a crazy extrapolation
    env1_extrapolated = merge(env1, (;cld6190_ann = [100_000]))
    env1_max_cld = merge(env1, (;cld6190_ann = [maximum(env.cld6190_ann)]))

    # using clamp the prediction uses the highest cloud
    @test Maxnet.predict(model_lasso, env1_extrapolated; link = IdentityLink(), clamp = true) == 
        Maxnet.predict(model_lasso, env1_max_cld; link = IdentityLink()) 
end

@testset "MLJ" begin
    mn = Maxnet.MaxnetBinaryClassifier

    # convert to continuous
    cont_keys = collect(key => Continuous for key in keys(env) if key !== :ecoreg)
    env_typed = MLJBase.coerce(env, cont_keys...)

    # make a machine
    mach1 = machine(mn(features = "lq", backend = LassoBackend()), env_typed, categorical(p_a))
    fit!(mach1)
    
    mach2 = machine(mn(features = "lqph", backend = GLMNetBackend()), env_typed, categorical(p_a))
    fit!(mach2)
    
    # make the equivalent model without mlj
    model = Maxnet.maxnet((p_a), env_typed; features = "lqph", backend = GLMNetBackend());


    # predict via MLJBase
    mljprediction = MLJBase.predict(mach2, env_typed)
    mlj_true_probability = pdf.(mljprediction, true)

    # test that this predicts the same as the equivalent model without mlj
    @test all(Maxnet.predict(model, env_typed) .≈ mlj_true_probability)

    @test Statistics.mean(mlj_true_probability[p_a]) > Statistics.mean(mlj_true_probability[.~p_a])
    @test minimum(mlj_true_probability) > 0.
    @test maximum(mlj_true_probability) < 1.
end

