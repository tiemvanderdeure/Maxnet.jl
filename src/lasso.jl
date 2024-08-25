function fit_lasso_path(
    mm, presences;
    weights, penalty_factor, lambda, kw...) 

    presence_matrix = [1 .- presences presences]
    GLMNet.glmnet(
        mm, presence_matrix, GLMNet.Binomial(); 
        weights, penalty_factor, lambda, standardize = false, kw...)
end

get_coefs(path::GLMNet.GLMNetPath) = path.betas
