# import Pkg
# Pkg.activate(@__DIR__)

using HMatrices
using BenchmarkTools
using Random
using LinearAlgebra
using LoopVectorization # for vectorized kernels
using StaticArrays

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

const SUITE = BenchmarkGroup()

BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
BLAS.set_num_threads(1)

#===========================================================================================

Benchmark the compression using:
    - aca
    - partial aca
    - tsvd

===========================================================================================#

SUITE["Compressors"] = BenchmarkGroup(["aca", "compression", "rkmatrix"])

rtol = 1.0e-6
N = 1000
X = rand(SVector{3, Float64}, N)
Y = map(x -> x + SVector(3, 3, 3), X)
K = laplace_matrix(X, Y)
irange = 1:N
jrange = 1:N

compressors = [PartialACA(; rtol = rtol), TSVD(; rtol = rtol)]

for comp in compressors
    SUITE["Compressors"][string(comp)] = @benchmarkable $comp($K, $irange, $jrange)
end

#===========================================================================================

Problem parameters

===========================================================================================#

N = 50_000
nmax = 200
eta = 3
rtol = 1.0e-4
radius = 1

X = points_on_cylinder(N, radius)
splitter = HMatrices.CardinalitySplitter(nmax)
Xclt = HMatrices.ClusterTree(X, splitter)
Xp = HMatrices.root_elements(Xclt)
adm = HMatrices.StrongAdmissibilityStd(eta)
comp = HMatrices.PartialACA(; rtol)

step = 1.75 * π * radius / sqrt(N)
k = 2 * π / (10 * step) # 10 pts per wavelength

kernels = Dict(
    "Laplace" => (kernel = laplace_matrix(X, X), global_index = true),
    "Laplace permuted" => (kernel = laplace_matrix(Xp, Xp), global_index = false),
    "Laplace vectorized" => (kernel = LaplaceMatrixVec(Xp, Xp), global_index = false),
)

for (name, (K, p)) in kernels
    SUITE[name] = BenchmarkGroup([name, N])
    for threads in (true, false)
        # assemble
        SUITE[name]["assemble threads=$threads"] = @benchmarkable assemble_hmatrix(
            $K,
            $Xclt,
            $Xclt;
            adm = $adm,
            comp = $comp,
            threads = $threads,
            distributed = false,
            global_index = $p,
        ) samples = 4 evals = 1
        # gemv and LU times for the same kernel should be the same (it only
        # cares about the hmatrix), so benchmark only one of them
        if occursin("vectorized", name)
            # Matrix vector product. The assembly is considered in a
            # setup-phase. A single sample but multiple evaluations to amortize
            # the setup cost
            SUITE[name]["gemv threads=$threads"] =
                @benchmarkable mul!(y, H, x; threads = $threads) setup = (
                H = assemble_hmatrix(
                    $K,
                    $Xclt,
                    $Xclt;
                    adm = $adm,
                    comp = $comp,
                    threads = true,
                    distributed = false,
                    global_index = $p,
                );
                x = randn($N);
                y = zeros($N)
            ) samples = 1 evals = 50
            # LU factorization. The assemble is considered in a setup-phase.
            SUITE[name]["LU threads=$threads"] =
                @benchmarkable lu!(H, $comp; threads = $threads) setup = (
                H = assemble_hmatrix(
                    $K,
                    $Xclt,
                    $Xclt;
                    adm = $adm,
                    comp = $comp,
                    threads = true,
                    distributed = false,
                    global_index = $p,
                )
            ) samples = 4 evals = 1
        end
    end
end
