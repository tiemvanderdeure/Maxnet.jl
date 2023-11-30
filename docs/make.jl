using maxnet
using Documenter

DocMeta.setdocmeta!(maxnet, :DocTestSetup, :(using maxnet); recursive=true)

makedocs(;
    modules=[maxnet],
    authors="Tiem van der Deure <tiemvanderdeure@gmail.com>, Michael Krabbe Borregaard <mkborregaard@sund.ku.dk>",
    repo="https://github.com/tiemvanderdeure/maxnet.jl/blob/{commit}{path}#{line}",
    sitename="maxnet.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tiemvanderdeure.github.io/maxnet.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tiemvanderdeure/maxnet.jl",
    devbranch="master",
)
