# Time the hierarchical LU and Cholesky factorizations, serial and threaded, on the
# single-layer operator of the sphere (Laplace, or Helmholtz with wavenumber k > 0,
# for which only the LU applies). Run with
#   julia --project=test -t 6 benchmark/factorization_bench.jl [n] [k]
# The tolerances can be set with e.g. `ATOLS=1e-6,1e-8`.
using HMatrices, LinearAlgebra
using HMatrices: ClusterTree, CardinalitySplitter, StrongAdmissibilityStd, PartialACA, RandomizedSVD
BLAS.set_num_threads(1)
include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

n = parse(Int, get(ARGS, 1, "16000"))
k = parse(Float64, get(ARGS, 2, "0"))
K = single_layer_sphere(n; k)
clt = ClusterTree(K.X, CardinalitySplitter(; nmax = 50))
assemble(K) = assemble_hmatrix(K, clt, clt; adm = StrongAdmissibilityStd(3), comp = PartialACA(; rtol = 1.0e-10), threads = false)
H = assemble(K)
Hh = iszero(k) ? assemble(Hermitian(K)) : nothing
y = rand(eltype(H), n)
relres(F) = (x = F \ y; norm(mul!(similar(y), H, x; threads = false) - y) / norm(y))

println("n = $n, k = $k, threads = $(Threads.nthreads())")
for atol in (parse(Float64, a) for a in split(get(ENV, "ATOLS", "1e-4,1e-6,1e-8"), ","))
    variants = Pair{String, Function}[]
    for comp in (RandomizedSVD(; atol), PartialACA(; atol))
        c = nameof(typeof(comp))
        push!(variants, "lu, $c" => t -> lu(H, comp; threads = t))
        isnothing(Hh) || push!(variants, "cholesky, $c" => t -> cholesky(Hh, comp; threads = t))
    end
    for (name, f) in variants, threads in (false, true)
        threads && Threads.nthreads() == 1 && continue
        f(threads) # warm up
        t = minimum(_ -> @elapsed(f(threads)), 1:3)
        println(
            "atol=$atol  ", rpad("$name (threads = $threads)", 42), "time = ", rpad(round(t, digits = 3), 6),
            " residual = ", round(relres(f(threads)), sigdigits = 2),
        )
    end
end
