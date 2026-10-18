using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

Random.seed!(1)

m = 5000
T = Float64
# well-conditioned and positive definite, without a dominant diagonal which
# would hide errors
K = single_layer_sphere(m)
X = Y = K.X

splitter = CardinalitySplitter(; nmax = 50)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(3)
comp = PartialACA(; atol = 1.0e-10)
for threads in (false, true)
    H = assemble_hmatrix(Hermitian(K), Xclt, Yclt; adm, comp, threads, distributed = false)
    hchol = cholesky(H; atol = 1.0e-10)
    y = rand(m)
    M = Matrix(K)
    exact = M \ y
    approx = hchol \ y
    @test norm(exact - approx) < 1.0e-7 * norm(exact)
    # test multiplication by checking if the solution is correct
    @test hchol.L * (hchol.U * approx) ≈ y
    @test startswith(sprint(show, hchol.L), "Lower triangular part of Hermitian HMatrix")
    @test startswith(sprint(show, hchol.U), "Upper triangular part of Hermitian HMatrix")
    # a compressor with its own buffer type
    hchol = cholesky(H, ACAWithRecompression(PartialACA(; atol = 1.0e-10)))
    @test norm(exact - hchol \ y) < 1.0e-7 * norm(exact)
end
