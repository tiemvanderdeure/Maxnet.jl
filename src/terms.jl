# Implement a Threshold term
struct ThresholdTerm{T,N} <: StatsModels.AbstractTerm
        term::T
        nknots::N
end

thresholds(t::Symbol, nknots::Int) = ThresholdTerm(term(t), term(nknots))

Base.show(io::IO, t::ThresholdTerm) = print(io, "thresholds($(t.term), nknots = $(t.nknots))")

function StatsModels.apply_schema(t::StatsModels.FunctionTerm{typeof(thresholds)},
                                sch::StatsModels.Schema,
                                Mod::Type{<:Any})
    apply_schema(ThresholdTerm(t.args...), sch, Mod)
end

# apply_schema to internal Terms and check for proper types
function StatsModels.apply_schema(t::ThresholdTerm,
                                sch::StatsModels.Schema,
                                Mod::Type{<:Any})
    term = StatsModels.apply_schema(t.term, sch, Mod)
    isa(term, StatsModels.ContinuousTerm) ||
        throw(ArgumentError("ThresholdTerm only works with continuous terms (got $term)"))
    isa(t.nknots, StatsModels.ConstantTerm) ||
        throw(ArgumentError("ThresholdTerm nknots must be a number (got $t.nknots)"))
    ThresholdTerm(term, t.nknots.n)
end

StatsModels.terms(p::ThresholdTerm) = terms(p.term)
StatsModels.termvars(p::ThresholdTerm) = StatsModels.termvars(p.term)
StatsModels.width(p::ThresholdTerm) = p.nknots
StatsModels.coefnames(p::ThresholdTerm) = (coefnames(p.term) * "_threshold_") .* string.(1:width(p))  

function StatsModels.modelcols(p::ThresholdTerm, d::NamedTuple)
    col = StatsModels.modelcols(p.term, d)
    thresholds(col, p.nknots)
end

### Hinge
struct HingeTerm{T,N} <: StatsModels.AbstractTerm
    term::T
    nknots::N
end

hinge(t::Symbol, nknots::Int) = HingeTerm(term(t), term(nknots))

Base.show(io::IO, t::HingeTerm) = print(io, "hinge($(t.term), nknots = $(t.nknots))")

function StatsModels.apply_schema(t::StatsModels.FunctionTerm{typeof(hinge)},
                            sch::StatsModels.Schema,
                            Mod::Type{<:Any})
apply_schema(HingeTerm(t.args...), sch, Mod)
end

# apply_schema to internal Terms and check for proper types
function StatsModels.apply_schema(t::HingeTerm,
                            sch::StatsModels.Schema,
                            Mod::Type{<:Any})
term = StatsModels.apply_schema(t.term, sch, Mod)
isa(term, StatsModels.ContinuousTerm) ||
    throw(ArgumentError("HingeTerm only works with continuous terms (got $term)"))
isa(t.nknots, StatsModels.ConstantTerm) ||
    throw(ArgumentError("HingeTerm nknots must be a number (got $t.nknots)"))
    HingeTerm(term, t.nknots.n)
end

StatsModels.terms(p::HingeTerm) = terms(p.term)
StatsModels.termvars(p::HingeTerm) = StatsModels.termvars(p.term)
StatsModels.width(p::HingeTerm) = p.nknots * 2 - 2
StatsModels.coefnames(p::HingeTerm) = (StatsModels.coefnames(p.term) * "_hinge_") .* string.(1:StatsModels.width(p))  

function StatsModels.modelcols(p::HingeTerm, d::NamedTuple)
    col = StatsModels.modelcols(p.term, d)
    hinge(col, p.nknots)
end

