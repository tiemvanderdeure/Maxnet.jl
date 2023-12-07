# numbers taken from R package
regtable(::CategoricalFeature) = [[0, 10, 17], [0.65, 0.5, 0.25]]
regtable(::LinearFeature) = [[0, 10, 30, 100], [1., 1., 0.2, 0.05]]
regtable(::QuadraticFeature) = [[0, 10, 17, 30, 100], [1.3, 0.8, 0.5, 0.25, 0.05]]
regtable(::ProductFeature) = [[0, 10, 17, 30, 100], [2.6, 1.6, 0.9, 0.55, 0.05]]
regtable(::HingeFeature) = [[0, 1], [0.5, 0.5]]
regtable(::ThresholdFeature) = [[0, 100], [2., 1.]]

default_or_regtable(default, t::Union{LinearFeature, QuadraticFeature, ProductFeature}) = default
default_or_regtable(default, t::Union{CategoricalFeature, ThresholdFeature, HingeFeature}) = regtable(t)

function default_regtable(feature_classes)
    if ProductFeature() in feature_classes
        regtable(ProductFeature())
    elseif QuadraticFeature() in feature_classes
        regtable(QuadraticFeature())
    else
        regtable(LinearFeature())
    end
end

function interpolate_from_regtable(regtable, np)
    interpolator = Interpolations.linear_interpolation(regtable[1], regtable[2], extrapolation_bc=Interpolations.Flat())
    interpolator(np) / sqrt(np)
end

function default_regularization(mm, classes, p, np = sum(p); 
    floor_reg = 0.001 # lowest reg is this times diff between min and max
)
    @assert size(mm, 2) == length(classes)
    @assert size(mm, 1) == length(p)

    mm_p = view(mm, p, :)

    unique_classes = unique(classes)
    default_table = default_regtable(unique_classes)
    class_regs = Dict(class => interpolate_from_regtable(default_or_regtable(default_table, class), np) for class in unique_classes)

    regs = @inbounds @views map(enumerate(classes)) do (i, class)
        # some lower bound on regularization
        reg = floor_reg * (Statistics.maximum(mm[:, i]) - Statistics.minimum(mm[:, i]))

        # increase regularization for extreme hinges
        if class == HingeFeature()
            hinge_reg = max(Statistics.std(mm_p[:, i]), 1/sqrt(np)) * 0.5 / sqrt(np)
            reg = max(reg, hinge_reg)
        end

        # increase reg'n for threshold features that are all 1 or 0 on presences
        if class == ThresholdFeature() 
            if sum(mm_p[:, i]) == np || sum(mm_p[:, i]) == 0.
                reg = max(reg, 1.)
            end
        end

        # regularization based on class
        class_reg = class_regs[class] * Statistics.std(mm_p[:, i])
        if class_reg > reg
        end
        reg = max(reg, class_reg)

        return reg
    end
    return regs
end