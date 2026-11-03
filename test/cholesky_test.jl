using Test
using HMatrices
using HMatrices: ClusterTree, CardinalitySplitter, StrongAdmissibilityStd, PartialACA
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
    H = assemble_hmatrix(Hermitian(K), Xclt, Yclt; adm, comp, threads)
    hchol = cholesky(H; atol = 1.0e-10, threads)
    y = rand(m)
    M = Matrix(K)
    exact = M \ y
    approx = hchol \ y
    B = rand(m, 3)
    @test norm(hchol \ B - M \ B) < 1.0e-7 * norm(M \ B)
    b = rand(ComplexF64, m)
    @test norm(hchol \ b - M \ b) < 1.0e-7 * norm(M \ b)
    @test norm(exact - approx) < 1.0e-7 * norm(exact)
    # test multiplication by checking if the solution is correct
    # the factors act in the ordering of the tree
    @test hchol.L * (hchol.U * approx[hchol.p]) ≈ y[hchol.p]
    @test propertynames(hchol) == (:L, :U, :p)
    @test startswith(sprint(show, hchol.L), "Lower triangular part of Hermitian HMatrix")
    @test startswith(sprint(show, hchol.U), "Upper triangular part of Hermitian HMatrix")
    # recompression after ACA
    hchol = cholesky(H, PartialACA(; atol = 1.0e-10, recompress = true))
    @test norm(exact - hchol \ y) < 1.0e-7 * norm(exact)
end

@testset "shift, single precision and complex entries" begin
    n = 1500
    S = single_layer_sphere(n)
    X = S.X
    y = rand(n)
    # a real shift of a `Hermitian` is a `Hermitian`
    H = assemble_hmatrix(Hermitian(S); rtol = 1.0e-10, threads = false)
    Hs = H + 2I
    @test Hs isa HMatrices.HermitianHMatrix && 2I + H isa HMatrices.HermitianHMatrix
    M = Matrix(S)
    @test norm(Matrix(Hs) - (M + 2I)) < 1.0e-8 * norm(M)
    @test norm(Matrix(H) - M) < 1.0e-8 * norm(M)
    exact = (M + 2I) \ y
    @test norm(cholesky(Hs; atol = 1.0e-10) \ y - exact) < 1.0e-7 * norm(exact)
    # single precision
    K32 = KernelMatrix((x, y) -> Float32(S.f(x, y)), X, X)
    H32 = assemble_hmatrix(Hermitian(K32); rtol = 1.0e-5, threads = false)
    y32 = Float32.(y)
    exact = Matrix(K32) \ y32
    for threads in (false, true)
        F = cholesky(H32; rtol = 1.0e-5, threads)
        @test eltype(F.factors) == Float32
        @test norm(F \ y32 - exact) < 1.0e-3 * norm(exact)
    end
    # complex entries: `D*S*D'` with `D` diagonal and unitary is Hermitian positive definite
    Kc = KernelMatrix((x, y) -> S.f(x, y) * cis(x[1] - y[1]), X, X)
    Hc = assemble_hmatrix(Hermitian(Kc); rtol = 1.0e-10, threads = false)
    b = rand(ComplexF64, n)
    exact = Matrix(Kc) \ b
    for threads in (false, true)
        F = cholesky(Hc; atol = 1.0e-10, threads)
        @test norm(F \ b - exact) < 1.0e-7 * norm(exact)
    end
end
