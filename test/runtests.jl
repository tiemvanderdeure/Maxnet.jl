using Maxnet, Test, MLJBase

p_a, env = Maxnet.bradypus()

@testset "utils" begin
    @test_throws ErrorException Maxnet.features_from_string("a")
end

@testset "Maxnet" begin
    model_glmnet = Maxnet.maxnet(Bool.(p_a), env, "lq"; backend = GLMNetBackend());
    model_lasso = Maxnet.maxnet(Bool.(p_a), env, "lq"; backend = LassoBackend());

    pred_glmnet = Maxnet.predict(model_glmnet, env)
    pred_lasso = Maxnet.predict(model_lasso, env)

    # no classes
    model_lasso = Maxnet.maxnet(Bool.(p_a), env; backend = LassoBackend());

    # each class
    Maxnet.maxnet(Bool.(p_a), env, "l"; backend = LassoBackend());
    Maxnet.maxnet(Bool.(p_a), env, "q"; backend = LassoBackend());
    Maxnet.maxnet(Bool.(p_a), env, "p"; backend = LassoBackend());
    Maxnet.maxnet(Bool.(p_a), env, "h"; backend = LassoBackend());
    Maxnet.maxnet(Bool.(p_a), env, "t"; backend = LassoBackend());

    # some class combinations
    Maxnet.maxnet(Bool.(p_a), env, "lq"; backend = LassoBackend());
    Maxnet.maxnet(Bool.(p_a), env, "lqp"; backend = LassoBackend());
    Maxnet.maxnet(Bool.(p_a), env, "lqh"; backend = LassoBackend());
    Maxnet.maxnet(Bool.(p_a), env, "lqph"; backend = LassoBackend());
    Maxnet.maxnet(Bool.(p_a), env, "lqpt"; backend = LassoBackend());
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
    
    @test mach2.fitresult[1].path isa Maxnet.GLMNet.GLMNetPath

    # predict via MLJBase
    t = pdf.(MLJBase.predict(mach2, env_typed), true)
end

