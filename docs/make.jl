using HMatrices
using Documenter

DocMeta.setdocmeta!(HMatrices, :DocTestSetup, :(using HMatrices); recursive = true)

on_CI = get(ENV, "CI", "false") == "true"

makedocs(;
    modules = [HMatrices],
    authors = "Luiz M. Faria <maltezfaria@gmail.com> and contributors",
    repo = "",
    sitename = "HMatrices.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://WaveProp.github.io/HMatrices.jl",
        assets = String[],
    ),
    pages = [
        "Getting started" => "index.md",
        "Kernel matrices" => "kernelmatrix.md",
        "Distributed HMatrix" => "dhmatrix.md",
        "Notebooks" => "notebooks.md",
        "Benchmarks" => ["benchs.md"],
        "References" => "references.md",
    ],
    warnonly = on_CI ? false : Documenter.except(:linkcheck_remotes),
    pagesonly = true,
)

deploydocs(;
    repo = "github.com/WaveProp/HMatrices.jl",
    devbranch = "main",
    push_preview = false,
)
