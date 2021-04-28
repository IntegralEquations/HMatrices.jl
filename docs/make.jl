using HMatrices
using Documenter

DocMeta.setdocmeta!(HMatrices, :DocTestSetup, :(using HMatrices); recursive=true)

makedocs(;
    modules=[HMatrices],
    authors="Luiz M. Faria <maltezfaria@gmail.com> and contributors",
    repo="https://github.com/maltezfaria/HMatrices.jl/blob/{commit}{path}#{line}",
    sitename="HMatrices.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://maltezfaria.github.io/HMatrices.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/maltezfaria/HMatrices.jl",
)
