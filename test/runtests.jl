using Maxnet, Test, MLJBase, Statistics

p_a, env = Maxnet.bradypus()
env1 = map(e -> [e[1]], env)

@testset "utils" begin
    @test_throws ErrorException Maxnet.features_from_string("a")
    @test Maxnet.features_from_string("l") == [LinearFeature(), CategoricalFeature()]
    @test Maxnet.features_from_string("q") == [QuadraticFeature()]
end

@testset "Maxnet" begin
    # test both backends work
    model_glmnet = Maxnet.maxnet((p_a), env; features = "lq", backend = GLMNetBackend());
    model_lasso = Maxnet.maxnet((p_a), env; features = "lq", backend = LassoBackend());

    # test both backends come up with approximately the same result
    @test all(isapprox.(model_glmnet.coefs, model_lasso.coefs; rtol = 0.1, atol = 0.1))
    @test Statistics.cor(model_glmnet.coefs, model_lasso.coefs) > 0.99

    # select classes automatically
    Maxnet.maxnet(p_a, env; backend = LassoBackend());

    # some class combinations
    Maxnet.maxnet(p_a, env; features = "lq", backend = LassoBackend());
    Maxnet.maxnet(p_a, env; features = "lqp", regularization_multiplier = 2., backend = LassoBackend());
    Maxnet.maxnet(p_a, env; features = "lqh", regularization_multiplier = 5., backend = LassoBackend());
    Maxnet.maxnet(p_a, env; features = "lqph", backend = LassoBackend());
    Maxnet.maxnet(p_a, env; features = "lqpt", backend = LassoBackend());

    # predictions
    prediction = Maxnet.predict(model_lasso, env)
    @test Statistics.mean(prediction[p_a]) > Statistics.mean(prediction[.~p_a])
    @test minimum(prediction) > 0.
    @test maximum(prediction) < 1.

    # clamp shouldn't change anything in this case
    @test all(prediction .== Maxnet.predict(model_lasso, env; clamp = true))
    
    # predict with a crazy extrapolation
    env1_extrapolated = merge(env1, (;cld6190_ann = [100_000]))
    # without clamp the prediction is crazy
    @test abs(Maxnet.predict(model_lasso, env1_extrapolated; link = IdentityLink())[1]) > 100_000.
    # without clamp the prediction is reasonable
    @test abs(Maxnet.predict(model_lasso, env1_extrapolated; link = IdentityLink(), clamp = true)[1]) < 5.
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
    
    # predict via MLJBase
    mljprediction = MLJBase.predict(mach2, env_typed)
    mlj_true_probability = pdf.(mljprediction, true)

    @test Statistics.mean(mlj_true_probability[p_a]) > Statistics.mean(mlj_true_probability[.~p_a])
    @test minimum(mlj_true_probability) > 0.
    @test maximum(mlj_true_probability) < 1.
end

