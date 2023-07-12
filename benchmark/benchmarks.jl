using BenchmarkTools
using PkgBenchmark
using HMatrices
using StaticArrays
using LinearAlgebra

using HMatrices: PartialACA, ACA, TSVD

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

# declare global const shared by all _benchfiles
const SUITE = BenchmarkGroup()

BLAS.set_num_threads(1)

SUITE["Compressors"] = BenchmarkGroup(["aca", "compression", "rkmatrix"])

rtol = 1e-6
N = 1000
X = rand(SVector{3,Float64}, N)
Y = map(x -> x + SVector(3, 3, 3), X)
K = laplace_matrix(X, Y)
irange = 1:N
jrange = 1:N

compressors = [PartialACA(; rtol=rtol), ACA(; rtol=rtol), TSVD(; rtol=rtol)]

for comp in compressors
    SUITE["Compressors"][string(comp)] = @benchmarkable $comp($K, $irange, $jrange)
end

# In what follows, we benchmark Laplace and Helmholtz single-layer operators for
# points on a cylinder. For each kernel, we benchmark:
# - time to assemble
# - time to do a gemv operation
# - time to do an lu factorization

ENV["JULIA_DEBUG"] = "HMatrices"

N = 100_000
nmax = 200
eta = 3
rtol = 1e-5
radius = 1

X = points_on_cylinder(radius, N)
splitter = HMatrices.CardinalitySplitter(nmax)
Xclt = HMatrices.ClusterTree(X, splitter)
Xp = HMatrices.root_elements(Xclt)
adm = HMatrices.StrongAdmissibilityStd(eta)
comp = HMatrices.PartialACA(; rtol)

step = 1.75 * π * radius / sqrt(N)
k = 2 * π / (10 * step) # 10 pts per wavelength

kernels = [("Laplace", laplace_matrix(X, X), true),
           ("Helmholtz", helmholtz_matrix(X, X, k), true),
           ("LaplaceVec", LaplaceMatrixVec(Xp, Xp), false),
           ("HelmholtzVec", HelmholtzMatrixVec(Xp, Xp, k), false)]

for (name, K, p) in kernels
    SUITE[name] = BenchmarkGroup([name, N])
    # bench assemble
    SUITE[name]["assemble cpu"] = @benchmarkable assemble_hmatrix($K, $Xclt, $Xclt;
                                                                  adm=$adm,
                                                                  comp=$comp, threads=false,
                                                                  distributed=false,
                                                                  global_index=$p)
    SUITE[name]["assemble threads"] = @benchmarkable assemble_hmatrix($K, $Xclt, $Xclt;
                                                                      adm=$adm, comp=$comp,
                                                                      threads=true,
                                                                      distributed=false,
                                                                      global_index=$p)
    # SUITE[name]["assemble procs"] = @benchmarkable assemble_hmatrix($K, $Xclt, $Xclt; adm = $adm, comp = $comp, threads = false, distributed = true, global_index = $p)
    # bench gemv only for regular case since the vectorized case should be the same
    if p
        x = rand(eltype(K), N)
        y = zero(x)
        H = assemble_hmatrix(K, Xclt, Xclt; adm, comp, threads=true, distributed=false,
                             global_index=p)
        SUITE[name]["gemv cpu"] = @benchmarkable mul!($y, $H, $x, $1, $0; threads=false,
                                                      global_index=$p)
        SUITE[name]["gemv threads"] = @benchmarkable mul!($y, $H, $x, $1, $0; threads=true,
                                                          global_index=$p)
        SUITE[name]["lu"] = @benchmarkable lu!($H; rank=5)
    end
end
