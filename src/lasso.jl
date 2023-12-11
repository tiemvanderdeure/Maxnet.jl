abstract type MaxnetBackend end
struct LassoBackend <: MaxnetBackend end
struct GLMNetBackend <: MaxnetBackend end

function fit_lasso_path(
    backend::LassoBackend, mm, presences;
    kw...) 

    Lasso.fit(
        Lasso.LassoPath, mm, presences, Lasso.Distributions.Binomial(); 
        standardize = false, irls_maxiter = 1_000, kw...)
end

function fit_lasso_path(
    backend::GLMNetBackend, mm, presences;
    wts, penalty_factor, λ, kw...) 

    presence_matrix = [presences 1 .- presences]
    GLMNet.glmnet(
        mm, presence_matrix, GLMNet.Binomial(); 
        weights = wts, penalty_factor = penalty_factor, lambda = λ, standardize = false)
end

get_coefs(path::GLMNet.GLMNetPath) = path.betas
get_coefs(path::Lasso.LassoPath) = path.coefs