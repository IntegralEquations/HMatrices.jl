using HMatrices
using Documenter
using LinearAlgebra

# the public API: the exported and `public` names, and the methods of `Base` and
# `LinearAlgebra` functions documented for an `HMatrix` (used by `references.md`)
const PUBLIC_API = Set{Any}(
    [
        (getfield(HMatrices, n) for n in names(HMatrices) if n !== :HMatrices)...,
        lu, lu!, cholesky, cholesky!, mul!, Matrix,
    ]
)
ispublicapi(x) = x in PUBLIC_API

DocMeta.setdocmeta!(HMatrices, :DocTestSetup, :(using HMatrices); recursive = true)

on_CI = get(ENV, "CI", "false") == "true"

makedocs(;
    modules = [HMatrices],
    repo = "",
    sitename = "HMatrices.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://IntegralEquations.github.io/HMatrices.jl",
        assets = String[],
    ),
    pages = [
        "Getting started" => "index.md",
        "Kernel matrices" => "kernelmatrix.md",
        "Reference" => "references.md",
        "Internals" => ["internals.md", "internals_arithmetic.md"],
    ],
    warnonly = on_CI ? false : Documenter.except(:linkcheck_remotes),
    pagesonly = true,
)

deploydocs(;
    repo = "github.com/IntegralEquations/HMatrices.jl",
    devbranch = "main",
    push_preview = true,
)
