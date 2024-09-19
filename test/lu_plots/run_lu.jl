using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays
using BenchmarkTools
using JLD2

using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

const FILE_PATH = joinpath(ARGS[1], "lu_results.jld2")

Random.seed!(1)

threads = Threads.nthreads() > 1

results = Dict{String,Dict{String,Float64}}()
if isfile(FILE_PATH)
    results = load(FILE_PATH)["results"]
end

thread_key = string(Threads.nthreads())
if !haskey(results, thread_key)
    results[thread_key] = Dict{String,Float64}()
end

#####################################################################
# 3D Laplace on sphere
#####################################################################
problem_size = [10000, 100000]
atol_list = (1e-1)
nmax_list = (100)
for size in problem_size
    for atol in atol_list
        for nmax in nmax_list
            m = size
            T = Float64
            X = points_on_sphere(m)
            Y = X
            K = laplace_matrix(X, X)

            println("Problem size=$(size), atol=$atol, nmin=$nmax")

            splitter = CardinalitySplitter(; nmax = nmax)
            Xclt = ClusterTree(X, splitter)
            Yclt = ClusterTree(Y, splitter)
            adm = StrongAdmissibilityStd(3)
            comp = PartialACA(; atol = atol)

            H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads, distributed = false)
            time = @belapsed lu($H; atol = $atol, threads = $threads)
            results[thread_key]["$(size)_$(atol)_$(nmax)_sphere"] = time
            println("Elapsed time=$time\n")
        end
    end
end

#####################################################################
# 3D Laplace on airplane
#####################################################################
problem_size = [(345, 10000), (92.2, 100000)]
atol_list = (1e-1)
nmax_list = (100)
for size in problem_size
    for atol in atol_list
        for nmax in nmax_list
            m = size[1]
            T = Float64
            X = points_on_airplain(m)
            Y = X
            K = laplace_matrix(X, X)

            println("Problem size=$(size[2]), atol=$atol, nmin=$nmax")

            splitter = CardinalitySplitter(; nmax = nmax)
            Xclt = ClusterTree(X, splitter)
            Yclt = ClusterTree(Y, splitter)
            adm = StrongAdmissibilityStd(3)
            comp = PartialACA(; atol = atol)

            H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads, distributed = false)
            time = @belapsed lu($H; atol = $atol, threads = $threads)
            results[thread_key]["$(size[2])_$(atol)_$(nmax)_airplane"] = time
            println("Elapsed time=$time\n")
        end
    end
end

JLD2.@save FILE_PATH results
