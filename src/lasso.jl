function fit_lasso_path(
    mm, presences;
    wts, penalty_factor, λ, kw...) 

    presence_matrix = [1 .- presences presences]
    GLMNet.glmnet(
        mm, presence_matrix, GLMNet.Binomial(); 
        weights = wts, penalty_factor = penalty_factor, lambda = λ, standardize = false)
end

get_coefs(path::GLMNet.GLMNetPath) = path.betas
