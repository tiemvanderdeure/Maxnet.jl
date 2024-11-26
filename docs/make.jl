using Maxnet
using Documenter

DocMeta.setdocmeta!(Maxnet, :DocTestSetup, :(using Maxnet; using MLJBase: Multiclass); recursive=true)

makedocs(;
    modules=[Maxnet],
    authors="Tiem van der Deure <tiemvanderdeure@gmail.com>, Michael Krabbe Borregaard <mkborregaard@sund.ku.dk>",
    repo="https://github.com/tiemvanderdeure/Maxnet.jl/blob/{commit}{path}#{line}",
    sitename="Maxnet.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tiemvanderdeure.github.io/Maxnet.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => Any[
            "Quick start" => "usage/quickstart.md",
            "MLJ" => "usage/mlj.md",
        ],
        "API reference" => "reference/api.md"
    ],
)

deploydocs(;
    repo="github.com/tiemvanderdeure/Maxnet.jl",
    devbranch="master",
)
