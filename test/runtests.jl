using Maxnet, Statistics, CategoricalArrays, MLJTestInterface
using Test

# read in Bradypus data
p_a, env = Maxnet.bradypus()
# Make the levels in ecoreg string to make sure that that works
env = merge(env, (; ecoreg = recode(env.ecoreg, (unwrap(l) => string(l) for l in levels(env.ecoreg))...)))
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
    @test Maxnet.default_features(1) == [LinearFeature(), CategoricalFeature()]

    @test Maxnet.hinge(1:5, 3) ==    [ 
     #  1:5   3:5  1:3  1:5 
        0.0   0.0  0.0  0.0
        0.25  0.0  0.5  0.25
        0.5   0.0  1.0  0.5
        0.75  0.5  1.0  0.75
        1.0   1.0  1.0  1.0
    ]
    @test size(Maxnet.hinge(1:200)) == (200, 98)

    presence, predictors = Maxnet.addsamples([true, false, false], (a = [1,2,3], b = [1,2,3]))
    @test presence == [true, false, false, false]
    @test predictors == (a = [1,2,3,1], b = [1,2,3,1])
end


@testset "Maxnet" begin
    # some class combinations and keywords
    m = maxnet(p_a, env; features = "lq");
    m2 = maxnet(p_a, env)
    m3 = maxnet(p_a, env[(:cld6190_ann, :h_dem)])
    m4 = maxnet(p_a, env[(:ecoreg,)], addsamplestobackground =false)
    m5 = maxnet(p_a, env[(:cld6190_ann, :h_dem)]; features = "ht", n_knots = 3)

    # test that the model throws an error if there are no presences
    @test_throws "All data points are absences" maxnet(falses(length(p_a)), env)
    @test_throws "All data points are presences" maxnet(trues(length(p_a)), env)

    # test the results
    @test m.entropy ≈ 6.114650341746531
    @test complexity(m) == 21
    @test m2.features == [LinearFeature(), CategoricalFeature(), QuadraticFeature(), HingeFeature(), ProductFeature()]
    @test m3.features == [LinearFeature(), QuadraticFeature(), HingeFeature(), ProductFeature()]
    @test m4.features == [CategoricalFeature()]
    @test m5.features == [HingeFeature(), ThresholdFeature()]
    @test length(m5.columns) == 14 # (n-1)*2 hinge columns and n threshold columns for each variable
    
    # predictions
    prediction = predict(m, env)
    @test Statistics.mean(prediction[p_a]) > Statistics.mean(prediction[.~p_a])
    @test minimum(prediction) > 0.
    @test maximum(prediction) < 1.
    @test mean(prediction) ≈ 0.24375837576014572 atol=1e-4

    # check that clamping works
    # clamp shouldn't change anything in this case
    @test prediction == predict(m, env; clamp = true)
    
    # predict with a crazy extrapolation
    env1_extrapolated = merge(env1, (;cld6190_ann = [100_000]))
    env1_max_cld = merge(env1, (;cld6190_ann = [maximum(env.cld6190_ann)]))

    # using clamp the prediction uses the highest cloud
    @test predict(m, env1_extrapolated; link = IdentityLink(), clamp = true) == 
        predict(m, env1_max_cld; link = IdentityLink()) 

    # test that maxnet works if no features are selected
    empty_model = maxnet(p_a, env; regularization_multiplier = 1000);
    @test complexity(empty_model) == 0
    @test Maxnet.selected_features(empty_model) == Symbol[]
    @test length(unique(predict(empty_model, env))) == 1

    # test that keywords arguments are passed to glmnet
    weights = ifelse.(p_a, 1.0, 10.0)
    m_w = maxnet(p_a, env; features = "lq", addsamplestobackground = false, weights)
    m = maxnet(p_a, env; features = "lq", addsamplestobackground = false)
    @test m_w.entropy > m.entropy
end

@testset "MLJ" begin
    data = MLJTestInterface.make_binary()
    failures, summary = MLJTestInterface.test(
        [MaxnetBinaryClassifier],
        data...;
        mod=@__MODULE__,
        verbosity=0, # bump to debug
        throw=false, # set to true to debug
    )
    @test isempty(failures)

    using MLJBase
    mn = Maxnet.MaxnetBinaryClassifier

    # Test model metadata
    @test name(mn) == "MaxnetBinaryClassifier"
    @test human_name(mn) == "Maxnet"
    @test package_name(mn) == "Maxnet"
    @test !supports_weights(mn)
    @test !is_pure_julia(mn)
    @test is_supervised(mn)
    @test package_license(mn) == "MIT"
    @test prediction_type(mn) == :probabilistic
    @test input_scitype(mn) == Table{<:Union{AbstractVector{<:Continuous}, AbstractVector{<:Finite}}}
    @test hyperparameters(mn) == (:features, :regularization_multiplier, :regularization_function, :addsamplestobackground, :n_knots, :weight_factor, :link, :clamp, :kw)

    # convert to continuous
    cont_keys = collect(key => Continuous for key in keys(env) if key !== :ecoreg)
    env_typed = MLJBase.coerce(env, cont_keys...)

    # make a machine
    mach1 = machine(mn(features = "lq"), env_typed, categorical(p_a))
    fit!(mach1)
    
    mach2 = machine(mn(features = "lqph"), env_typed, categorical(p_a))
    fit!(mach2)
    
    # make the equivalent model without mlj
    model = Maxnet.maxnet((p_a), env_typed; features = "lqph");

    # predict via MLJBase
    mljprediction = MLJBase.predict(mach2, env_typed)
    mlj_true_probability = pdf.(mljprediction, true)

    # test that this predicts the same as the equivalent model without mlj

    @test all(predict(model, env_typed) .≈ mlj_true_probability)

    @test Statistics.mean(mlj_true_probability[p_a]) > Statistics.mean(mlj_true_probability[.~p_a])
    @test minimum(mlj_true_probability) > 0.
    @test maximum(mlj_true_probability) < 1.
end

