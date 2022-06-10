using HMatrices
using Documenter

DocMeta.setdocmeta!(HMatrices, :DocTestSetup, :(using HMatrices); recursive=true)

makedocs(;
    modules=[HMatrices],
    authors="Luiz M. Faria <maltezfaria@gmail.com> and contributors",
    repo="https://github.com/WaveProp/HMatrices.jl/blob/{commit}{path}#{line}",
    sitename="HMatrices.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://WaveProp.github.io/HMatrices.jl",
        assets=String[],
    ),
    pages=[
        "Getting started" => "index.md",
        "Kernel matrices" => "kernelmatrix.md",
        "Distributed HMatrix" => "dhmatrix.md",
        "Notebooks" => "notebooks.md",
        "Benchmarks" => ["benchs_v0.1.0.md","benchs_v0.1.1.md","benchs_v0.1.2.md"],
        "References" => "references.md"
    ],
)

deploydocs(;
    repo="github.com/WaveProp/HMatrices.jl",
    devbranch="main"
)
