# The benchmark suite, run by AirspeedVelocity on pull requests labeled `run benchmark`
# (see `.github/workflows/benchmark.yml`). The script of the pull request is run on both
# the base and the head revisions, so it must only use functions which exist in both.
#
# Run it locally with e.g.
#   benchpkg HMatrices --path=. --rev=main,dirty --script=benchmark/benchmarks.jl --add=StaticArrays --exeflags="--threads=4"
# With `HMATRICES_BENCHMARK_BLAS=AppleAccelerate` (or `MKL`), and that package added with
# `--add`, it is loaded to replace the default BLAS.
using HMatrices
using HMatrices: ClusterTree, CardinalitySplitter, StrongAdmissibilityStd, PartialACA, TSVD
using BenchmarkTools
using LinearAlgebra
using Random
using StaticArrays

let blas = get(ENV, "HMATRICES_BENCHMARK_BLAS", "")
    isempty(blas) || @eval using $(Symbol(blas))
end
# the parallelism comes from the blocks, not from BLAS
BLAS.set_num_threads(1)
@info "Benchmarking on $(Threads.nthreads()) threads" BLAS.get_config()

const SUITE = BenchmarkGroup()

"""
    fibonacci_sphere(n)

Quasi-uniform distribution of `n` points on the unit sphere.
"""
function fibonacci_sphere(n)
    ϕ = (1 + sqrt(5)) / 2
    return map(1:n) do i
        z = 1 - (2i - 1) / n
        r = sqrt(1 - z^2)
        θ = 2π * i / ϕ
        return SVector(r * cos(θ), r * sin(θ), z)
    end
end

"""
    single_layer_sphere(n; k = 0)

Nyström-like discretization of the Laplace (`k = 0`) or Helmholtz single-layer operator
on the unit sphere, with `n` quasi-uniform points of equal weights. The diagonal is the
integral of the static part of the kernel over a disk of the area of a point.
"""
function single_layer_sphere(n; k = 0)
    X = fibonacci_sphere(n)
    w = 4π / n
    a = sqrt(w / π)
    f = if iszero(k)
        (x, y) -> x == y ? a / 2 : w / (4π * norm(x - y))
    else
        (x, y) -> begin
            x == y && return complex(a / 2, k * w / (4π))
            d = norm(x - y)
            return w * exp(im * k * d) / (4π * d)
        end
    end
    return KernelMatrix(f, X, X)
end

const ATOL = 1.0e-6 # tolerance of the factorizations
const ADM = StrongAdmissibilityStd(3)
const COMP = PartialACA(; rtol = 1.0e-8) # compressor of the assembly

cluster(X) = ClusterTree(X, CardinalitySplitter(; nmax = 50))

#===========================================================================================
Compression of a single block of well-separated clusters
===========================================================================================#

let N = 1000, rtol = 1.0e-6
    Random.seed!(1)
    X = rand(SVector{3, Float64}, N)
    Y = map(x -> x + SVector(3, 3, 3), X)
    K = KernelMatrix((x, y) -> 1 / (4π * norm(x - y)), X, Y)
    compressors = Dict("PartialACA" => PartialACA(; rtol), "TSVD" => TSVD(; rtol))
    if isdefined(HMatrices, :RandomizedSVD)
        compressors["RandomizedSVD"] = HMatrices.RandomizedSVD(; rtol)
    end
    G = SUITE["compressors"] = BenchmarkGroup()
    for (name, comp) in compressors
        G[name] = @benchmarkable $comp($K, 1:$N, 1:$N)
    end
end

#===========================================================================================
Assembly, products and factorizations of the single-layer operator on the sphere
===========================================================================================#

let n = 20_000
    K = single_layer_sphere(n)
    clt = cluster(K.X)
    H = assemble_hmatrix(K, clt, clt; adm = ADM, comp = COMP)
    Hh = assemble_hmatrix(Hermitian(K), clt, clt; adm = ADM, comp = COMP)
    F = lu(H; atol = ATOL)
    x, y = rand(n), zeros(n)
    G = SUITE["Laplace"] = BenchmarkGroup()
    for threads in (false, true)
        G["assemble threads=$threads"] = @benchmarkable assemble_hmatrix($K, $clt, $clt; adm = ADM, comp = COMP, threads = $threads) samples = 3 evals = 1
        G["mul! threads=$threads"] = @benchmarkable mul!($y, $H, $x, 1, 0; threads = $threads)
        G["lu threads=$threads"] = @benchmarkable lu!(M; atol = ATOL, threads = $threads) setup = (M = deepcopy($H)) samples = 3 evals = 1
        G["cholesky threads=$threads"] = @benchmarkable cholesky!(M; atol = ATOL, threads = $threads) setup = (M = deepcopy($Hh)) samples = 3 evals = 1
    end
    G["solve"] = @benchmarkable $F \ $x
end

let n = 8_000
    K = single_layer_sphere(n; k = 8)
    clt = cluster(K.X)
    H = assemble_hmatrix(K, clt, clt; adm = ADM, comp = COMP)
    G = SUITE["Helmholtz"] = BenchmarkGroup()
    for threads in (false, true)
        G["lu threads=$threads"] = @benchmarkable lu!(M; atol = ATOL, threads = $threads) setup = (M = deepcopy($H)) samples = 3 evals = 1
    end
end
