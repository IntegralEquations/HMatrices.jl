using Test
using HMatrices
using HMatrices: ClusterTree, CardinalitySplitter, StrongAdmissibilityStd, PartialACA, ACAWithRecompression
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

Random.seed!(1)

m = 5000
T = Float64
# well-conditioned, without a dominant diagonal which would hide errors
K = single_layer_sphere(m)
X = Y = K.X

splitter = CardinalitySplitter(; nmax = 50)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(3)
comp = PartialACA(; atol = 1.0e-10)
for threads in (false, true)
    H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads)
    hlu = lu(H; atol = 1.0e-10)
    y = rand(m)
    M = Matrix(K)
    exact = M \ y
    approx = hlu \ y
    @test norm(exact - approx) < 1.0e-7 * norm(exact)
    # test multiplication by checking if the solution is correct
    # the factors act in the ordering of the trees
    @test hlu.L * (hlu.U * approx[hlu.q]) ≈ y[hlu.p]
    @test propertynames(hlu) == (:L, :U, :p, :q)
    @test startswith(sprint(show, hlu.L), "Lower triangular part of HMatrix")
    @test startswith(sprint(show, hlu.U), "Upper triangular part of HMatrix")
    for t in (true, false)
        z = mul!(zero(y), hlu.U, approx[hlu.q], 1, 0; threads = t)
        @test mul!(zero(y), hlu.L, z, 1, 0; threads = t) ≈ y[hlu.p]
    end
    # a compressor with its own buffer type
    hlu = lu(H, ACAWithRecompression(PartialACA(; atol = 1.0e-10)))
    @test norm(exact - hlu \ y) < 1.0e-7 * norm(exact)
end

@testset "different row and column trees" begin
    # same points in a different order: the row and column permutations differ
    n = 1500
    S = single_layer_sphere(n)
    X, Y = S.X, S.X[randperm(n)]
    K = KernelMatrix(S.f, X, Y)
    H = assemble_hmatrix(K, ClusterTree(X, splitter), ClusterTree(Y, splitter); adm, comp, threads = false)
    y = rand(n)
    exact = Matrix(K) \ y
    F = lu(H; atol = 1.0e-10)
    @test norm(F \ y - exact) < 1.0e-6 * norm(exact)
    # the factors are in the ordering of the trees, which differ for rows and columns
    @test F.p != F.q
    z = rand(n)
    M = Matrix(K)
    @test norm(F.L * (F.U * z) - M[F.p, F.q] * z) < 1.0e-6 * norm(M[F.p, F.q] * z)
end
