using Test
using Logging
using HMatrices
using HMatrices: ClusterTree, CardinalitySplitter, StrongAdmissibilityStd, PartialACA
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

Random.seed!(1)

m = 2000
n = 2000

X = rand(SVector{3, Float64}, m)
Y = [rand(SVector{3, Float64}) for _ in 1:n]
splitter = CardinalitySplitter(; nmax = 50)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(; eta = 3)
atol = 1.0e-5
comp = PartialACA(; atol)
K = laplace_matrix(X, Y)
H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false)

H_full = HMatrices._tree_matrix(H)

R = HMatrices.RkMatrix(rand(m, 10), rand(n, 10))
P = HMatrices.RkMatrix(rand(m, 10), rand(n, 10))

α = rand() - 0.5
β = rand() - 0.5

@testset "hmul!" begin
    # the rows and columns of `H` come from different points, so the products are
    # those of the stored blocks (as in the factorizations)
    C = deepcopy(H)
    tmp = β * H_full + α * H_full * H_full
    HMatrices._hmul_tree!(C, H, H, α, β, PartialACA(; atol = 1.0e-6))
    @test HMatrices._tree_matrix(C) ≈ tmp
    # adjoint
    C = deepcopy(H)
    tmp = β * H_full + α * adjoint(H_full) * H_full
    HMatrices._hmul_tree!(C, adjoint(H), H, α, β, PartialACA(; atol = 1.0e-6))
    @test HMatrices._tree_matrix(C) ≈ tmp
    # recompression after ACA
    C = deepcopy(H)
    tmp = β * H_full + α * H_full * H_full
    HMatrices._hmul_tree!(C, H, H, α, β, PartialACA(; atol = 1.0e-6, recompress = true))
    @test HMatrices._tree_matrix(C) ≈ tmp
end

@testset "hmul! with adjoints" begin
    # kernel without the large regularized diagonal, which would hide errors in
    # the other entries of the products
    Xs = X[1:800]
    Xsclt = ClusterTree(Xs, CardinalitySplitter(; nmax = 30))
    f = (x, y) -> x == y ? 0.0im : exp(2im * norm(x - y)) / (4π * norm(x - y))
    Hs = assemble_hmatrix(KernelMatrix(f, Xs, Xs), Xsclt, Xsclt; comp = PartialACA(; atol = 1.0e-10), threads = false)
    Ms = HMatrices._tree_matrix(Hs)
    for opA in (identity, adjoint), opB in (identity, adjoint)
        C = deepcopy(Hs)
        HMatrices.hmul!(C, opA(Hs), opB(Hs), -1, 1, PartialACA(; atol = 1.0e-10))
        exact = Ms - opA(Ms) * opB(Ms)
        @test norm(HMatrices._tree_matrix(C) - exact) < 1.0e-8 * norm(exact)
    end
end

@testset "hmul! with zero rows" begin
    # compactly supported kernel: rows of the products sampled by the compressor
    # are zero up to rounding errors, and must not be used as pivots
    Xc = X[1:1000]
    Xcclt = ClusterTree(Xc, CardinalitySplitter(; nmax = 30))
    f = (x, y) -> (d = norm(x - y); x == y ? 10.0 : d < 0.3 ? exp(-d) / (4π * d) : 0.0)
    for comp in (PartialACA(; atol = 1.0e-8, recompress = false), PartialACA(; atol = 1.0e-8))
        Hc = assemble_hmatrix(KernelMatrix(f, Xc, Xc), Xcclt, Xcclt; comp, threads = false)
        Mc = HMatrices._tree_matrix(Hc)
        C = deepcopy(Hc)
        # the sharp cutoff of the kernel limits the accuracy of the partial ACA, which
        # may warn that it failed
        with_logger(() -> HMatrices.hmul!(C, Hc, Hc, -1, 1, comp), NullLogger())
        exact = Mc - Mc * Mc
        @test norm(HMatrices._tree_matrix(C) - exact) < 1.0e-2 * norm(exact)
    end
end

@testset "gemv" begin
    T = eltype(H)
    m, n = size(H)
    x = rand(T, n)
    y = rand(T, m)

    @testset "serial" begin
        exact = β * y + α * Matrix(H) * x
        approx = mul!(copy(y), H, x, α, β; threads = false)
        @test exact ≈ approx

        # multiply by adjoint
        adjH = adjoint(deepcopy(H))
        exact = β * y + α * adjoint(Matrix(H)) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = false)
        @test exact ≈ approx
    end

    @testset "threads" begin
        exact = β * y + α * Matrix(H) * x
        approx = mul!(copy(y), H, x, α, β; threads = true)
        @test exact ≈ approx

        # multiply by adjoint
        adjH = adjoint(deepcopy(H))
        exact = β * y + α * adjoint(Matrix(H)) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = true)
        @test exact ≈ approx
    end

    @testset "hermitian" begin
        threads = false
        for threads in (true, false)
            K = laplace_matrix(X, X)
            Hsym = assemble_hmatrix(Hermitian(K), Xclt, Xclt; adm, comp, threads)
            H = assemble_hmatrix(K, Xclt, Xclt; adm, comp, threads)
            x = rand(n)
            y1 = mul!(zero(x), H, x, 1, 0; threads)
            y2 = mul!(zero(x), Hsym, x, 1, 0; threads)
            @test y1 ≈ y2
        end
    end
end

@testset "gemm" begin
    α = rand() - 0.5
    β = rand() - 0.5
    T = eltype(H)
    m, n = size(H)
    k = 10
    x = rand(T, n, k)
    y = rand(T, m, k)

    @testset "serial" begin
        exact = β * y + α * Matrix(H) * x
        approx = mul!(copy(y), H, x, α, β; threads = false)
        @test exact ≈ approx
    end

    @testset "threads" begin
        exact = β * y + α * Matrix(H) * x
        approx = mul!(copy(y), H, x, α, β; threads = true)
        @test exact ≈ approx
    end

    @test_throws DimensionMismatch mul!(zeros(T, m, 2), H, rand(T, n, 3))
end

@testset "columns and products of a product operator" begin
    # complex entries, a complex multiplier, two products and a low-rank term
    K = single_layer_sphere(1200; k = 4)
    clt = ClusterTree(K.X, CardinalitySplitter(; nmax = 40))
    H = assemble_hmatrix(K, clt, clt; adm = StrongAdmissibilityStd(3), comp = PartialACA(; atol = 1.0e-8), threads = false)
    tm = HMatrices._tree_matrix
    chd = HMatrices.children(H)
    X1, Y1, X2, Y2 = chd[2, 1], chd[1, 2], chd[2, 2], chd[2, 2]
    m, n = size(X1, 1), size(Y1, 2)
    A, B = rand(ComplexF64, m, 5), rand(ComplexF64, n, 5)
    a = 0.3 - 0.8im
    L = HMatrices.LeafOperator{ComplexF64}(A, B, [(X1, Y1), (X2, Y2)], a)
    M = a * (tm(X1) * tm(Y1) + tm(X2) * tm(Y2)) + A * B'
    for j in (1, 7, 41)
        @test HMatrices.getblock!(zeros(ComplexF64, m), L, 1:m, j) ≈ M[:, j]
        @test HMatrices.getblock!(zeros(ComplexF64, n), adjoint(L), 1:n, j) ≈ M'[:, j]
    end
    # a whole block (e.g. for `TSVD`)
    @test HMatrices.getblock!(zeros(ComplexF64, m, 3), L, 1:m, [1, 7, 41]) ≈ M[:, [1, 7, 41]]
    V, W = rand(ComplexF64, n, 3), rand(ComplexF64, m, 3)
    @test mul!(zeros(ComplexF64, m, 3), L, V) ≈ M * V
    @test mul!(zeros(ComplexF64, n, 3), adjoint(L), W) ≈ M' * W
end
