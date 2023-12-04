using StatsBase, StatsModels

struct ThresholdTerm{T,N} <: AbstractTerm
        term::T
        nknots::N
end

thresholds(t::Symbol, nknots::Int) = ThresholdTerm(term(t), term(nknots))

Base.show(io::IO, t::ThresholdTerm) = print(io, "thresholds($(t.term), nknots = $(t.nknots))")

function StatsModels.apply_schema(t::FunctionTerm{typeof(thresholds)},
                                sch::StatsModels.Schema,
                                Mod::Type{<:Any})
    apply_schema(ThresholdsTerm(t.args_parsed...), sch, Mod)
end

# apply_schema to internal Terms and check for proper types
function StatsModels.apply_schema(t::ThresholdTerm,
                                sch::StatsModels.Schema,
                                Mod::Type{<:Any})
    term = apply_schema(t.term, sch, Mod)
    isa(term, ContinuousTerm) ||
    throw(ArgumentError("ThresholdTerm only works with continuous terms (got $term)"))
    isa(t.nknots, ConstantTerm) ||
    throw(ArgumentError("ThresholdTerm nknots must be a number (got $t.nknots)"))
    ThresholdTerm(term, t.nknots.n)
end

StatsModels.terms(p::ThresholdTerm) = terms(p.term)
StatsModels.termvars(p::ThresholdTerm) = StatsModels.termvars(p.term)
StatsModels.width(p::ThresholdTerm) = 1

function StatsModels.modelcols(p::ThresholdTerm, d::NamedTuple)
    col = modelcols(p.term, d)
    reduce(hcat, [col.^n for n in 1:p.deg])
end