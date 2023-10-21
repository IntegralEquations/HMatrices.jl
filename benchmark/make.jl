using PkgBenchmark

function postprocess(results, fname = "benchs")
    # writeresults(joinpath(path,fname),results)
    dir = @__DIR__
    path = joinpath(dir, "../docs/src")
    return export_markdown(joinpath(path, fname * ".md"), results)
end

env = Dict("JULIA_NUM_THREADS" => 4, "OPEN_BLAS_NUM_THREADS" => 1)

config = BenchmarkConfig(; juliacmd = `julia -O3`, env)

dir = @__DIR__
retune = false
results = benchmarkpkg("HMatrices", config; retune)

postprocess(results)
