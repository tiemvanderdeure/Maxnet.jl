# transformations
function hinge(x, nknots = 50)
    mi, ma = extrema(x)
    k = range(mi, ma; length = nknots)
    lh = hingeval.(x, k[1:end-1]', [ma])
    rh = hingeval.(x, [mi], k[2:end]')
    [lh rh]
end

function hinge_ranges(x, nknots = 50)
    mi, ma = extrema(x)
    k = range(mi, ma; length = nknots)
    [[(mi, ma) for mi in k[1:end-1]]; [(mi, ma) for ma in k[2:end]]]
end

hingeval(x, mi, ma) = clamp((x - mi) / (ma - mi), 0., 1.)

# adding presence samples to background
function addsamples(presences, predictors) # this should be a general sdm thing
    predictors = Tables.rowtable(predictors)

    to_add = setdiff(predictors[presences], predictors[.~presences]);

    predictors_ = Tables.columntable([predictors; to_add])
    presences_= vcat(presences, fill(false, length(to_add)))

    return presences_, predictors_
end

# generate lambdas
function lambdas(reg, p, weights; λmax = 4, n = 200)
    c = Statistics.mean(reg) * sum(p) / sum(weights)
    10 .^ range(λmax, 0; length = n) .* c
end