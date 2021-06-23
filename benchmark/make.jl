using PkgBenchmark
using HMatrices
using LinearAlgebra

function postprocess(results,fname="benchs")
    # writeresults(joinpath(path,fname),results)
    dir = @__DIR__
    path = joinpath(dir,"../docs/src")
    export_markdown(joinpath(path,fname*".md"),results)
end

env = Dict("JULIA_NUM_THREADS" =>1,
           "OPEN_BLAS_NUM_THREADS" => 1
           )

config = BenchmarkConfig(;juliacmd=`/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia -O3`,env)

dir       = @__DIR__
retune    = false
results   = benchmarkpkg("HMatrices",config;retune)

postprocess(results)
